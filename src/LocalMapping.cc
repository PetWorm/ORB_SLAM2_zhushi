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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

// QXC：LocalMapping线程入口函数
void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);	// QXC：设置mbAcceptKeyFrames的标志位为false，使得Tracking不能立刻添加新的关键帧

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())		// QXC：当有新关键帧加入时返回true，否则返回false
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();	// QXC：处理队列中最早添加的关键帧，包括计算BoW信息、被该帧观测到的地图点的信息更新、共视图维护以及将该关键帧加入全局图

            // Check recent MapPoints
            MapPointCulling();		// QXC：筛选最近添加的地图点，剔除mlpRecentAddedMapPoints中一些质量不好的点

            // Triangulate new MapPoints
            CreateNewMapPoints();	// QXC：将当前新关键帧和其covisibility graph中的关键帧的特征点（还未当做地图点的特征点）进行匹配，尝试建立新的地图点（建立了会放入mlpRecentAddedMapPoints）
									//      该函数会调用CheckNewKeyFrames检测是否还有未处理的新关键帧，如果有则直接返回

            if(!CheckNewKeyFrames())	// QXC：mlNewKeyFrames中可能有多个新添加的关键帧，进入这个if表示没有更多的新关键帧了
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();	// QXC：在最新添加的关键帧和其在covisibility graph中附近关键帧之间，fuse地图点，并更新connection
            }

            mbAbortBA = false;		// QXC：将新添加的关键帧都处理之后就可以继续BA

            if(!CheckNewKeyFrames() && !stopRequested())	// QXC：没有未处理的新添加关键帧，并且线程未被要求停止时
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);	// QXC：以最新添加的关键帧及其covisibility graph（包括其中所有关键帧，
																							//      所有观测到的地图点，其他能观测到这些地图点的关键帧【优化中固定】）
																							//      为BA对象，会根据优化结果解除不适合的地图点和关键帧之间的联系。

                // Check redundant local Keyframes
                KeyFrameCulling();	// QXC：挑出冗余关键帧，并将其设置为坏帧
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);	// QXC：向回环线程的关键帧队列中添加新关键帧
        }
        else if(Stop())		// QXC：当Tracking进入OnlyTracking模式时，会请求LocalMapping停止（stop），此时Stop函数将（应当）返回True；LocalClosing在修正回环时也会造成Stop
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())	// QXC：如果Stop返回true则isStopped一定也返回true
            {
                usleep(3000);		// QXC：当LocalMapping被要求stop，且没有终结（finish）时，Run函数会一直在这个while循环中等待，且此时LocalMapping禁止新的关键帧插入
            }
            if(CheckFinish())		// QXC：查询线程是否终结，终结则跳出while大循环
                break;
        }

        ResetIfRequested();	// QXC：当Tracking请求了reset时，清空“最近添加关键帧”和“最近添加地图点”两个容器

        // Tracking will see that Local Mapping is idle
        SetAcceptKeyFrames(true);	// QXC：设置mbAcceptKeyFrames的标志位为true，使得Tracking可以立刻添加新的关键帧

        if(CheckFinish())		// QXC：查询线程是否终结，终结则跳出while大循环
            break;

        usleep(3000);
    }

    SetFinish();	// QXC：示意线程已终结
}

// QXC：由Tracking线程调用，向LocalMapping线程添加新的关键帧，并发出打断BA标志
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

// QXC：查询Tracking线程没有加入新关键帧，有返回true，无返回false
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

// QXC：取出Tracking添加的最早的关键帧：计算其BoW信息；更新被其观测到的地图点的相关信息（包括添加观测、更新平均单位矢量和深度、计算最优描述子等）；
//      更新covisibility graph信息；将该关键帧加入全局地图
void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();	// QXC：取出最早添加的新关键帧
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))	// QXC：新添加的关键帧应当是不存在于任何MapPoint的观测（Observation）中的，因此IsInKeyFrame应返回false
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

// QXC：对最近添加的地图点进行一遍筛选
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // QXC：对最近加入局部地图的地图点进行一遍筛选，剔除坏点、连续跟踪比例低的点、观测很少的点以及首次观测间隔较远（但现在才添加）的点
    while(lit!=mlpRecentAddedMapPoints.end())	// QXC：当mlpRecentAddedMapPoints中没有元素时（最近没有添加地图点），begin将等于end，while循环一次都进不去
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())			// QXC：Bad地图点舍弃
        {
            lit = mlpRecentAddedMapPoints.erase(lit);	// QXC：擦除mlpRecentAddedMapPoints中lit所指向的地图点指针，并返回指向下一个地图点指针的迭代器
        }
        else if(pMP->GetFoundRatio()<0.25f )	// QXC：连续跟踪比例低的地图点
        {
            pMP->SetBadFlag();		// QXC：将该地图点设置为坏点，并且清除观测、擦除关键帧中指向它的指针、从全局地图中将其剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)	// QXC：当前关键帧已经比该地图点首次被观测到时间隔至少两帧（指两帧关键帧），但该地图点被（关键帧）观测次数小于等于2时，将该地图点设置为坏点
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)	// QXC：当前关键帧离地图点首次观测关键帧太远（但现在才添加），舍弃该地图点 （？不太理解这种情形出现的场景，难道是插入关键帧速度快？）
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

// QXC：尝试在新关键帧和周围共视的关键帧中寻找新地图点：
//      将新关键帧和其covisibility graph中前20个关键帧分别进行特征匹配（不包含已经是地图点的），对匹配上的点做三角化，选取质量优良者作为新的地图点。
//      注意，当检测到还有新关键帧未处理时会直接返回。
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())		// QXC：当还有添加的关键帧未处理时函数返回
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)	// QXC：非单目情况，暂时不看
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else			// QXC：单目情况判断基线长度是否达标的方式是：基线长度至少是地图点深度中位值的1%
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;		// QXC：存储下面SearchForTriangulation函数匹配的匹配对，每个匹配对包括匹配特征在两帧特征队列中的ID
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);	// QXC：利用对极几何约束和BoW节点分类对两关键帧中的非地图点特征进行匹配

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
		// QXC：尝试对匹配上的点进行三角化，剔除视差小的、三角化结果深度不正确的、三角化结果像素坐标投影误差过大的以及（特征金字塔）尺度不一致的
		//      成功三角化的点作为新的地图点，维护其相关信息
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;	// QXC：单目情形始终为false

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;	// QXC：单目情形始终为false

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);	// QXC：特征归一化相机系坐标
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

			// QXC：单目情形无论if还是elseif都进不去
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);	// QXC：单目情形cosParallaxStereo仍然是声明时赋予的值

            cv::Mat x3D;
	    
			// QXC：单目情形，下面if的条件简化为：(cosParallaxRays>0 && cosParallaxRays<0.9998)	// QXC：arccos(0.9998)约等于65.66°，因此能进入if的视差为65.66°～90°
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method		// ？QXC：按照代码列出了矩阵计算式，但没看懂计算式的含义
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();				// QXC：此时获得的是一个四维向量，它是三角化点的世界坐标的一个齐次表示

                if(x3D.at<float>(3)==0)		// 剔除无穷远点
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);	// QXC：通过归一化和舍弃齐次项获得三维坐标

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)	// QXC：单目情形始终不成立
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)	// QXC：单目情形始终不成立
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);		// QXC：检查三角化结果在第一帧的深度是否为正
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);		// QXC：检查三角化结果在第二帧的深度是否为正
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)	// QXC：检查三角化结果的像素投影误差（第一帧）
                    continue;
            }
            else	// QXC：非单目情形
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)	// QXC：检查三角化结果的像素投影误差（第二帧）
                    continue;
            }
            else	// QXC：非单目情形
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)		// ？QXC：检查（特征金字塔）尺度一致性，不是太理解
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);		// QXC：能到达这里说明这个点三角化成功了，可以作为地图点了

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
	    
            mlpRecentAddedMapPoints.push_back(pMP);		// QXC：向本地（LocalMapping）添新加地图点

            nnew++;
        }
    }
}

// QXC：在最新添加的关键帧mpCurrentKeyFrame附近的covisibility graph中挑选出一系列关键帧，用mpCurrentKeyFrame中的地图点fuse一遍这些关键帧中的地图点，
//      再用这些关键帧中的地图点fuse一遍mpCurrentKeyFrame的地图点，起到为双方增补地图点的作用。
//      然后更新mpCurrentKeyFrame中所有地图点（包括新增点以及新和这个关键帧建立联系的旧点）的描述子、平均观测矢量及深度，并更新mpCurrentKeyFrame的connection信息。
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    // QXC：在最新关键帧临近的covisibility graph关键帧（单目至多20帧），以及这些关键帧临近的covsiibility graph关键帧（至多5帧）中，选取非坏关键帧加入容器vpTargetKFs
    //      采取了相关手段来规避重复添加（分析mnFuseTargetForKF和mnId）
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    // QXC：将最新关键帧中的地图点和covisibility graph中的地图点进行整合或添加
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);	// QXC：尝试在pKFi中找与vpMapPointMatches中元素对应的特征，若该特征已经是与vpMapPointMatches中对应元素不同的地图点，
												//      则根据特定的策略选出更好的地图点指针，来取代另一个的位置，如果不是则在关键帧pKFi中新增一个地图点。
												//      vpMapPointMatches中被fuse的点有两种状态：
												//      一种是mpCurrentKeyFrame中的地图点主导，这种地图点在之后的fuse中还可能（但可能性很小）继续被修正；
												//      另一种是pKFi中的地图点主导，这时候vpMapPointMatches中相应的元素不能再正确的表示mpCurrentKeyFrame中当前的地图点指针，
												//      但它会在MapPoint::Replace中被设置为坏点，从而不再参与之后的Fuse，
												//      应当认为covisibility graph中的其他关键帧的地图点都是经过了fuse的，所以这些点不再参与fuse也无妨了。
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    // QXC：选出vpTargetKFs中的所有有效地图点，放入容器vpFuseCandidates中，采取了相关手段来规避重复添加。即挑出附近covisibility graph中的所有地图点
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);	// QXC：用covisibility graph中的所有地图点来fuse最新添加的关键帧的地图点


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)		// QXC：这里的更新包括新增地图点以及新和这个关键帧fuse上的旧地图点
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }
    // QXC：vpTargetKFs中的地图点不用再做上述操作，因为它们要么是之前的地图点（已经执行过这些操作），要么来自mpCurrentKeyFrame，在上述代码中已经完成更新

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

// QXC：根据两关键帧的位姿和内参，计算基础矩阵
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

// QXC：在System::TrackMonocular中，当System的mbActivateLocalizationMode标志位发生变化时，会调用该函数；
//      或者，在LoopClosing中，当需要修正回环时（LoopClosing::CorrectLoop），会调用该函数让LocalMapping暂停；
//      或者，在LoopClosing中，当成功进行了globalBA后需要修正全局地图时（LoopClosing::RunGlobalBundleAdjustment），会调用该函数让LocalMapping暂停；
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

// ？QXC：暂时不理解含义
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)	// ？QXC：被要求停止但还未停止？不理解mbNotStop的意义
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

// QXC：返回mbStopped标志位
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;	// QXC：mbStopped为true表示LocalMapping主要功能停止，即不再进行localBA（但是仍会有插入新关键帧的动作）
}

// QXC：返回mbStopRequested标志位。在判断是否进行localBA时，调用该函数查询，若为true说明正在要求LocalMapping线程停止，因此不进行localBA
bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

// QXC：System::TrackMonocular中，当mbDeactivateLocalizationMode置位重新启用局部地图时，会调用本函数，从而解除LocalMapping的stop状态；
//      或者在LoopClosing::CorrectLoop中，当完成了回环闭合后，也会调用本函数，从而解除LocalMapping的stop状态。
void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    
    // QXC：如果线程终结，则直接返回
    if(mbFinished)
        return;
    
    // QXC：解除LocalMapping的stop状态
    mbStopped = false;
    mbStopRequested = false;
    
    // 将mlNewKeyFrames中的所有指针直接删除（彻底释放关键帧指针，这个关键帧对象就不存在了），并且清空最近新添加的关键帧队列
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

// QXC：由Tracking线程调用，查看LocalMapping是否空闲，mbAcceptKeyFrames为true表示LocalMapping空闲，可以立刻添加新关键帧，为false时不能立刻添加（单目情况）
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

// QXC：设置mbAcceptKeyFrames标志位的值，当其值为true说明LocalMapping空闲，其值为false表示LocalMapping忙碌
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

// QXC：在mpCurrentKeyFrame的covisibility graph中挑选冗余关键帧，并将冗余关键帧设置为坏帧。注意，mpCurrentKeyFrame并不会被筛选掉。
//      冗余关键帧指的是：该关键帧中有90%以上的有效地图点能够在其他至少三个关键帧中的相似或更好尺度下被观测到。
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();	// QXC：返回的vector容器中并不包含mpCurrentKeyFrame，因此它不会被筛掉

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)	// QXC：第一帧（初始化基准帧）显然不能剔除
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;	// QXC：记录认为是冗余地图点（在除pKF之外的至少其他三个关键帧中的近似或更好尺度下被观测到）的数目
        int nMPs=0;			// QXC：记录当前关键帧pKF中有效（首先是地图点，然后不能是坏点）的地图点数目
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)	// QXC：对每个地图点，分析其是否满足冗余要求
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)	// QXC：非双目情况
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)	// QXC：地图点在近似或更小的尺度被观测到
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)		// QXC：这个条件表示地图点pMP至少在除了pKF以外的三个其他关键帧中的近似或更好的尺度下被观测到
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();		// QXC：将关键帧设为坏帧
    }
}

// QXC：获得三维矢量v的反对称帧（即李代数中的hat运算符）
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

// QXC：由Tracking::Reset调用，请求对LocalMapping进行reset
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

// QXC：当通过mbResetRequested标志位要求reset时，清空最近添加关键帧/地图点队列
void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

// QXC：在程序的main函数中，当所有图片都读取完毕后，将调用System::Shutdown函数，该函数将调用本函数，示意线程终结（finish）
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// QXC：返回mbFinishRequested，当其为true时表示线程终结（finish）。
//      可以看到，Run函数中调用该函数返回true时，都直接跳出大while循环。
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;	// ？QXC：mbFinishRequested含义？
}

// QXC：设置mbFinished标志位，示意线程已终结。该标志位会在System::Shutdown中查询
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

// QXC：System::Shutdown中调用，查询线程是否终结
bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
