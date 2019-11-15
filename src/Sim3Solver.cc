/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{

// QXC：Sim3Solver对象构造函数，一一对应的存储了匹配上的地图点的指针、相机系坐标、像素坐标、最大容许误差等。并设置了RANSAC参数
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size();		// QXC：实际上就是第一个关键帧的特征点数目

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    for(int i1=0; i1<mN1; i1++)		// QXC：对每一个匹配上的地图点对进行遍历
    {
        if(vpMatched12[i1])
        {
            MapPoint* pMP1 = vpKeyFrameMP1[i1];
            MapPoint* pMP2 = vpMatched12[i1];

            if(!pMP1)					// QXC：在匹配中非地图点和坏点其实都判断过了，但是LocalMapping可能会对地图点做改变，所以这里保险起见再判断一次
                continue;

            if(pMP1->isBad() || pMP2->isBad())		// QXC：在匹配中非地图点和坏点其实都判断过了，但是LocalMapping可能会对地图点做改变，所以这里保险起见再判断一次
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);	// QXC：获得两个地图点对应的特征ID		// QXC：实际上indexKF1等于i1
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0)
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1);		// QXC：最大允许误差——关键帧1
            mvnMaxError2.push_back(9.210*sigmaSquare2);		// QXC：最大允许误差——关键帧2

            mvpMapPoints1.push_back(pMP1);			// QXC：地图点指针——关键帧1
            mvpMapPoints2.push_back(pMP2);			// QXC：地图点指针——关键帧2
            mvnIndices1.push_back(i1);				// QXC：第一个关键帧中地图点的特征ID

            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);			// QXC：相机系下的地图点坐标——关键帧1

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);			// QXC：相机系下的地图点坐标——关键帧2

            mvAllIndices.push_back(idx);			// QXC：当前匹配对的序号
            idx++;
        }
    }

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);	// QXC：计算地图点像素坐标
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();	// QXC：设置RANSAC参数
}

// QXC：设置RANSAC参数，包括RANSAC概率、最小内点数、最大迭代次数等
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}

// QXC：进行迭代计算相似变换，每次迭代随机的挑选三个匹配点对进行相似变换计算，根据计算结果筛选内点（两个方向的像素投影误差都很小的点认为是内点），
//      当有内点数大于阈值（nIterations）的情况时才会返回一个非空的相似变换结果。
//      只要不是匹配点对总数目小于最小内点数量这种情况发生，vbInliers（按照第一帧的特征点ID索引）和nInliers都会返回最好迭代情况的内点标志位和数目
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);	// QXC：vbInliers维数为第一帧的特征点总数
    nInliers=0;

    if(N<mRansacMinInliers)	// QXC：匹配对数目小于最小内点数时直接返回
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)		// QXC：mnIterations记录本Sim3Solver对象总的迭代次数，nCurrentIterations记录本次调用本函数（iterate）的迭代次数
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < 3; ++i)		// QXC：不重复的随机抽取三对匹配对
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back();	// QXC：将被选中的位置用尾部填充，再把尾部pop掉，这样可以避免重复抽样，并且保持各位置机会均等
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i,P3Dc2i);	// QXC：利用三对匹配点分别在两个相机系中的坐标，计算两个相机系的相似变换

        CheckInliers();			// QXC：利用前面得到的相似变换计算两个方向的投影像素误差，认为两个误差都很小的点为内点

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)	// QXC：当有一次迭代内点数目达标时，就直接返回
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;	// QXC：由筛选后特征点的内点标识，得到原始特征点内点标识
                return mBestT12;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)	// QXC：当本Sim3Solver对象总的迭代次数超出阈值时
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

// QXC：求P中所有列向量的质心C，并求P去质心后的结果Pr
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);	// QXC：对P的所有列（由第三个参数取1决定）求和（由第四个参数取CV_REDUCE_SUM决定），输出结果放到C中
    C = C/P.cols;	// QXC：C的每个元素为P的每行元素的均值，即C为P中所有列向量的质心

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;	// QXC：将P的每个列向量去质心，存储到Pr中
    }
}

// QXC：根据文献“Horn 1987, Closed-form solution of absolute orientataion using unit quaternions”给出的方法，利用三对匹配点分别在两个相机系中的坐标，计算两个相机系的相似变换
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    // QXC：分别求出两个关键帧中的地图点相机系坐标质心，并将两个关键帧中的地图点相机系坐标去质心
    ComputeCentroid(P1,Pr1,O1);		// QXC：P1为原始的三个相机系坐标（三个三维列向量组成的3*3矩阵），O1为它们的质心（三维列向量），Pr1为P1的去质心结果
    ComputeCentroid(P2,Pr2,O2);		// QXC：同上

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t();

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3,3,P1.type());

    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis			// QXC：获得相机系2到相机系1的方向余弦阵

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;	// QXC：注意这里是个纯旋转

    // Step 6: Scale

    if(!mbFixScale)	// QXC：单目情况mbFixScale始终为false，因此会进入这个if；但单目+IMU情况不会进入
    {
        double nom = Pr1.dot(P3);		// QXC：将Pr1和P3分别存储成所有列向量排成一列（顺序从左至右），再做内积
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);			// QXC：aux_P3存储了P3的二次幂
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);	// QXC：den为aux_P3所有元素之和
            }
        }

        ms12i = nom/den;			// ？QXC：获得相机系2到相机系1转换时使用的尺度（？不太理解其计算方法的含义）
    }
    else		// QXC：非单目情况，或者单目+IMU情况
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;		// ？QXC：根据尺度ms12i，计算相机系1原点到相机系2原点的矢量在相机系1下的坐标（？计算方法不理解，为何用质心可以算这个？？）

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));		// QXC：相机系2到相机系1的转换矩阵

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));		// QXC：相机系1到相机系2的转换矩阵
}

// QXC：利用两个方向的相似变换及两帧的内参，分别把相机系1坐标和相机系2坐标转换成帧2和帧1的像素坐标，并求像素误差，将两个方向投影像素误差都在阈值内的点认为是内点
void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);		// QXC：将相机系2下的匹配点坐标转换到相机系1下，并投影成像素坐标，存到vP2im1中
    Project(mvX3Dc1,vP1im2,mT21i,mK2);		// QXC：类似上面

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])	// QXC：内点要求在两帧的投影误差都很小
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}

// QXC：获得最佳（iterate函数中内点数最多）相似变换的旋转部分（相机系2到相机系1的方向余弦阵）
cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

// QXC：获得最佳（iterate函数中内点数最多）相似变换的平移部分（相机系1原点到相机系2原点的矢量在相机系1下的坐标）
cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

// QXC：获得最佳（iterate函数中内点数最多）相似变换的变换尺度
float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

// QXC：根据w系点坐标vP3Dw、w系到c系的做表转换阵Tcw、c系的内参矩阵，求取点在c系代表的相机的像素坐标
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

// QXC：根据相机系坐标和内参矩阵，计算像素坐标
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
