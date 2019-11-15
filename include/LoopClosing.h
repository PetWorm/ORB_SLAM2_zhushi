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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
//#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;	// QXC：姑且认为这就是一个普通的map

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;	// QXC：记录最新添加的当前关键帧
    KeyFrame* mpMatchedKF;	// QXC：记录和当前关键帧成功完成相似变换计算的回环成员
    std::vector<ConsistentGroup> mvConsistentGroups;		// QXC：存储了上次的所有一致组pair的容器。一个一致组pair由某个一致组及其一致次数结对组成（一致组由回环成员及其共视图关键帧成员组成）
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;	// QXC：存储满足一致性次数的回环成员关键帧
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;	// QXC：当前关键帧的排序共视图关键帧及其本身
    std::vector<MapPoint*> mvpCurrentMatchedPoints;	// QXC：记录使当前关键帧和回环成员完成匹配（同时完成了相似变化计算）的地图点指针，尺寸和当前关键帧特征点数一致，存储了匹配上的（如果有匹配的话）回环成员中的地图点指针。注意！其中并不存储当前关键帧的其他地图点。注意！在ComputeSim3中对该成员变量进行了赋值，这并不表明当前关键帧中的mvpMapPoints的内容与其一致，mvpMapPoints中的地图点指针仍未和回环成员进行fuse。具体的地图点一致化操作在CorrectLoop中进行。
    std::vector<MapPoint*> mvpLoopMapPoints;		// QXC：存储了回环成员的covisibility graph关键帧中所有非坏地图点
    cv::Mat mScw;		// QXC：世界系（回环成员时刻）到当前关键帧的相似变换（cv::Mat格式）
    g2o::Sim3 mg2oScw;		// QXC：世界系（回环成员时刻）到当前关键帧的相似变换（g2o::Sim3格式）

    long unsigned int mLastLoopKFid;	// QXC：记录最近一次发起并完成回环校正的关键帧的ID（只要完成essential graph即可，不要求完成globalBA）

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;		// QXC：该标志位标识当前是否有一个线程正在进行globalBA
    bool mbFinishedGBA;		// QXC：标识GglobalBA是否已经结束（？仅仅是和mbRunningGBA相反吗？）
    bool mbStopGBA;		// QXC：用于打断globalBA的标志位
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;		// QXC：传入的是“mSensor!=MONOCULAR”，因此单目情形该参数始终为false


    bool mnFullBAIdx;		// QXC：当一次globalBA在另一次还没结束时被发起，则该变量自增1
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
