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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;			// QXC:存储的是关键帧序列中的ID
    const long unsigned int mnFrameId;		// QXC:存储的是该关键帧在所有帧序列中的ID

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;		// QXC：表明该关键帧当前是哪一帧的跟踪参考关键帧（可用于防止重复添加）（详细参见函数Tracking::UpdateLocalKeyFrames）
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;		// QXC：存储某个关键帧的ID，用来标识关键帧是否已经加入到该帧发起的localBA中（详情参见Optimizer::LocalBundleAdjustment函数）
    long unsigned int mnBAFixedForKF;		// QXC：存储某个关键帧的ID，用来标识地图点是否已经加入到该帧发起的localBA的固定帧队列中（可用来避免重复添加）（详情参见Optimizer::LocalBundleAdjustment函数）

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;		// QXC：存储某个关键帧的ID，用来标识关键帧是否已经加入到该帧发起的LoopDetection中（可用来避免重复添加）（详情参见KeyFrameDatabase::DetectLoopCandidates函数）
    int mnLoopWords;				// QXC：存储该关键帧和某个特定关键帧共有的BoW单词类型的数目（详情参见KeyFrameDatabase::DetectLoopCandidates函数）
    float mLoopScore;				// QXC：当mnLoopWords大于一定阈值时，该成员变量将记录本关键帧和特定关键帧的BoW字典分数
    long unsigned int mnRelocQuery;	// QXC：记录了一个帧ID，该帧正在试图由本关键帧对其进行重定位
    int mnRelocWords;			// QXC：与mnRelocQuery配套使用，记录了正试图利用本关键帧进行重定位的帧，其与本关键帧具有相同的单词的数量
    float mRelocScore;			// QXC：与mnRelocQuery配套使用，记录了本关键帧和mnRelocQuery代表的帧的BoW相似度分数

    // Variables used by loop closing
    cv::Mat mTcwGBA;		// QXC：当本关键帧参与了LoopClosing发起的globalBA后，位姿的优化结果将存在这里。（不直接用Tcw存储是因为在LoopClosing::RunGlobalBundleAdjustment中需要利用被globalBA优化的关键帧的Twc和未被优化的子关键帧的Tcw来计算相对位姿，并通过相对位姿和修正过的父帧位姿来修正子帧位姿）
    cv::Mat mTcwBefGBA;		// QXC：存储被globalBA修正之前的相机位姿。（用来接受当前的Tcw，发生在将mTcwGBA赋值给Tcw之前，参见LoopClosing::RunGlobalBundleAdjustment）
    long unsigned int mnBAGlobalForKF;	// QXC：记录了一个关键帧ID，该关键帧发起的LoopClosing触发了globalBA，并对本关键帧位姿进行了优化（参见Optimizer::BundleAdjustment）

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;	// QXC：实际是一个map容器，first指针指向BoW树中的一个节点ID，second指针指向一个vector容器，这个容器内包含了帧内被分类到该BoW节点的特征在该帧特征队列中的ID

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;		// QXC：相机内参矩阵


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;	// QXC：将世界坐标系点坐标转换到相机坐标系的转换矩阵，用于表示相机位姿，包含Rcw（世界系到相机系DCM）和tcw（相机系原点到世界系原点的矢量在相机系的坐标）
    cv::Mat Twc;	// QXC：Tcw的逆
    cv::Mat Ow;		// QXC：当前帧对应相机坐标系的原点在世界系的坐标

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;		// QXC:包含了所有有共视地图点的关键帧和共视地图点个数
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;	// QXC:只包含共视地图点数量超过阈值的关键帧,或共视地图点最多的那个关键帧。按共视地图点数降序排列。与mvOrderedWeights一一对应
    std::vector<int> mvOrderedWeights;				// QXC:只包含共视地图点数量超过阈值的地图点数,或共视地图点最大数量。按共视地图点数降序排列。与mvpOrderedConnectedKeyFrames一一对应
    // QXC：注意！某个关键帧的共视图中，只会存在mnID比其小的关键帧

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;
    // QXC：Spanning Tree由父关键帧和子关键帧这种形式维护

    // Bad flags
    bool mbNotErase;		// QXC：为true时，表示本关键帧暂时不可被设置为坏帧
    bool mbToBeErased;		// QXC：为true时，表示本关键帧应当被设置为坏帧（但目前还未设置）
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
