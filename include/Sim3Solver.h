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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2
{

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:

    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

    void CheckInliers();

    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    KeyFrame* mpKF1;
    KeyFrame* mpKF2;

    std::vector<cv::Mat> mvX3Dc1;		// QXC：一一对应的存储两个关键帧中匹配上的两个地图点在各关键帧相机系下的坐标（已筛除坏点）
    std::vector<cv::Mat> mvX3Dc2;
    
    std::vector<MapPoint*> mvpMapPoints1;	// QXC：一一对应的存储两个关键帧中匹配上的两个地图点的指针（已筛除坏点）
    std::vector<MapPoint*> mvpMapPoints2;
    
    std::vector<MapPoint*> mvpMatches12;	// QXC：尺寸和关键帧1中特征数相同，各元素存储了匹配上的（如果有的话）关键帧2中对应的地图点的指针（原始数据，未经过坏点筛选）
    
    std::vector<size_t> mvnIndices1;		// QXC：用到的第一个关键帧中地图点对应的特征点ID（已筛除坏点）
    
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    
    std::vector<size_t> mvnMaxError1;		// QXC：一一对应的存储两个关键帧中匹配上的两个地图点的最大允许误差（已筛除坏点）
    std::vector<size_t> mvnMaxError2;

    int N;	// QXC：有效的地图点匹配对数目
    int mN1;	// QXC：第一个关键帧中的特征数目

    // Current Estimation
    cv::Mat mR12i;
    cv::Mat mt12i;
    float ms12i;
    cv::Mat mT12i;		// QXC：该相似变换是由旋转量mR12i、平移量mt12i和尺度ms12i共同计算得到的，旋转部分为(ms12i×mR12i)，平移部分为mt12i（参见ComputeSim3）
    cv::Mat mT21i;		// QXC：旋转部分为R=(1/ms12i×mR12i')，平移部分为-R*mt12i（参见ComputeSim3）
    std::vector<bool> mvbInliersi;	// QXC：维数和mvpMapPoints1一样，每个元素标识该点是不是内点
    int mnInliersi;			// QXC：记录内点数目

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;	// QXC：记录内点数最多的mvbInliersi
    int mnBestInliers;			// QXC：mvbBestInliers中内点的数目（即内点数最多的情况时的内点数）
    cv::Mat mBestT12;
    cv::Mat mBestRotation;
    cv::Mat mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;		// QXC：单目情况该参数始终为false

    // Indices for random selection
    std::vector<size_t> mvAllIndices;		// QXC：从0到N存储了当前匹配对的顺序ID

    // Projections
    std::vector<cv::Mat> mvP1im1;		// QXC：一一对应的存储两个关键帧中匹配上的两个地图点的像素坐标
    std::vector<cv::Mat> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;		// QXC：Sim3Solver对象总迭代次数阈值

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;	// QXC：两个关键帧的内参矩阵
    cv::Mat mK2;

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
