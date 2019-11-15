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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;	// QXC：标志reset请求的标志位。reset请求由Tracking发起（调用Tracking::Reset），行为是清空mlNewKeyFrames和mlNewKeyFrames这两个容器
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;	// QXC：标志finish请求的标志位。finish请求由System（main函数中）发起，行为是跳出LocalMapping线程大while循环，终结LocalMapping线程
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;	// QXC：存储由Tracking添加的新的关键帧

    KeyFrame* mpCurrentKeyFrame;		// QXC：存储了最近添加的关键帧的指针

    std::list<MapPoint*> mlpRecentAddedMapPoints;	// QXC：暂存最近添加的地图点。其中的元素由LocalMapping自己添加和维护

    std::mutex mMutexNewKFs;

    bool mbAbortBA;	// QXC：判断是否打断LocalBA的标志位，作为参数传递给Optimizer::LocalBundleAdjustment函数，该函数将查询该标志位，当为true时会打断LocalBA（具体由g2o::setForceStopFlag实现）

    bool mbStopped;
    bool mbStopRequested;	// QXC：标志stop请求的标志位。stop请求由Tracking发起（当进入OnlyTracking模式时），行为是停止localBA，但不会阻碍添加新关键帧
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;	// QXC：该标志位由Tracking查询，为true时Tracking可以立刻添加新关键帧，为false时不能立刻添加（单目情况）
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
