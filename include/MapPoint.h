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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;	// QXC:该地图点被（应当是局部地图中的）关键帧观测的次数

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;		// QXC：表明该地图点当前是哪一帧的跟踪参考地图点
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;		// QXC：存储某个关键帧的ID，用来标识地图点是否已经加入到该帧发起的localBA中（可用来避免重复添加）（详情参见Optimizer::LocalBundleAdjustment函数）
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;		// QXC：存储某个关键帧的ID，用来标识地图点是否已经加入到该帧发起的LoopClosing中（可用来避免重复添加）（详情参见LoopClosing::ComputeSim3函数）
    long unsigned int mnCorrectedByKF;		// QXC：存储某个关键帧的ID，用来标识地图点是否已经在被该帧发起的CorrectLoop修正坐标（可用来避免重复添加）（详情参见LoopClosing::CorrectLoop函数）
    long unsigned int mnCorrectedReference;	// QXC：记录本地图点在被相似变换修正坐标时，观测到其的关键帧ID（详情参见LoopClosing::CorrectLoop函数）
    cv::Mat mPosGBA;				// QXC：记录本地图点经过globalBA优化后的世界坐标
    long unsigned int mnBAGlobalForKF;		// QXC：存储某个关键帧的ID，当LoopClosing发起的globalBA对本地图点进行了优化后，该ID将和发起本次LoopClosing的关键帧ID一致（参见Optimizer::BundleAdjustment）


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;

     // Mean viewing direction
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;		// QXC：地图点的参考关键帧一定是essential graph中spanning tree上的成员

     // Tracking counters
     int mnVisible;	// QXC：地图点被普通帧观测到的次数（不论该地图点此时是否在【应当是局部地图中的】其他关键帧中也被观测到）
     int mnFound;	// QXC：地图点同时在至少一个关键帧（应当是局部关键帧）中有观测时，其被普通帧观测到的次数（可以用来表达连续跟踪的次数）

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
