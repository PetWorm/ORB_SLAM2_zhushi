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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())		// QXC：查看LocalMapping线程有没有向本线程插入新的关键帧
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())		// QXC：以当前关键帧的共视图中满足要求的关键帧作为回环成员，用以回环成员的共视图构成的组和之前的一致组进行一致性匹配，更新一致组。在有一致组的一致性次数达标时逻辑值为true
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())	// QXC：尝试计算当前关键帧和回环成员间的相似变换，如果有满足要求的相似变换和足够的地图点匹配数目，则返回true
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();	// QXC：根据当前关键帧和回环关键帧间的相似变换构造回环边，并对essential graph进行位姿图优化，并开启新线程进行globalBA。
									//      另，如果本函数执行伊始有正在进行的globalBA，则打断之。
               }
            }
        }       

        ResetIfRequested();		// QXC：查看线程是否被要求重置（reset）（由Tracking发起），若要求重置则进行相关变量的重置

        if(CheckFinish())		// QXC：查看主线程是否发来终止（finish）指令，如果有该指令则本线程终结
            break;

        usleep(5000);
    }

    SetFinish();			// QXC：示意线程已终结
}

// QXC：由LocalMapping线程调用，向回环关键帧队列插入新关键帧（除初始化参考帧之外）
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

// QXC：查看回环关键帧队列中是否有元素
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

// ？QXC：根据当前关键帧，检测有没有合适的回环成员。（？细节仍然有疑问）
//      首先从关键帧队列中获得当前关键帧，若当前帧在上次回环不久后就到来，则直接返回。
//      搜寻回环成员（回环成员首先不应属于当前关键帧的covisibility graph，其他需满足的条件参见KeyFrameDatabase::DetectLoopCandidates函数），若没找到回环成员本函数也直接返回。
//      根据回环成员构成的成员组，与之前的一致组进行一致性匹配，试图延续一致性。将延续了之前一致组一致性的成员组保存下来（保存成员组和一致性次数），作为新的一致组。
//      同时，当一致性次数达标时，将成员组对应的回环成员存放到指定容器中。
bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();		// QXC：使得本关键帧暂时不会被设置为坏帧（这个操作在LocalMapping::KeyFrameCulling中进行）
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)	// QXC：回环成员太少之————上次回环刚结束不久
    {
        mpKeyFrameDB->add(mpCurrentKF);		// QXC：将本关键帧添加到BoW数据库中（在BoW相应的单词对应的关键帧指针容器中添加本关键帧）
        mpCurrentKF->SetErase();		// QXC：使得本关键帧可以被设置为坏帧
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)		// QXC：在当前帧（mpCurrentKF）的covisibility graph关键帧BoW中找和当前帧BoW评分的最小的值
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);	// QXC：搜寻mpCurrentKF可能对应的回环成员

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())		// QXC：没找到满足条件的回环成员时
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    // QXC：对每一个当前回环组，试图在之前的一致组（mvConsistentGroups保存）中，找到能继承一致性的一致组。
    //      换个说法是：【对每个mvConsistentGroups中的成员，试图在当前回环成员组中找到和其一致的组】。
    //      如果本函数是首次调用，则每一次外层for循环都会直接进入最后的if(!bConsistentForSomeGroup)；
    //      每个mvConsistentGroups成员只能找一个对应的一致回环成员组（碰到的第一个）（一旦该回环组使得一致性次数满足最小要求，该回环组对应回环成员将被保存至容器...
    //      ...mvpEnoughConsistentCandidates中，而一旦该回环组不能使一致性次数满足最小要求，则对于该一致组，本次DetectLoop也将无法再找到使其一致性达标的回环组）；
    //      每个回环成员组有一次机会（碰上的第一个）成为满足最小一致性次数要求的组，此时对应的回环成员被存储到容器mvpEnoughConsistentCandidates中。
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();	// QXC：获得回环成员的共视图
        spCandidateGroup.insert(pCandidateKF);						// QXC：将回环成员也加入进去形成回环成员组（group）

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;		// QXC：该标志位将标识本回环成员组是否继承了之前某个一致组的一致性
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
			// QXC：遍历回环成员组，看其中是否有关键帧同时属于前一个一致组（？）
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))		// QXC：注意，这里是不关注关键帧是不是坏帧的
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
				// QXC：对于mvConsistentGroups中的每个成员，只找一个一致的回环成员组
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                
                // QXC：当前回环成员对应的成员组所代表的一致组，其一致次数达到了阈值要求
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)	// QXC：对于第iG个nCurrentConsistency，如果它第一次加一就小于阈值，则之后不会再有机会进入这个if内
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)	// QXC：首次调用本函数时，肯定会进入这里；或对于没有对应上之前一致组的回环组，将把该回环组和一致性次数组成的pair作为新的一致组的一个元素
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;	// QXC：更新一致组（mvConsistentGroups）。如果本函数首次调用，则这里是给一致组赋初值
							//      不是第一次调用本函数时，更新的一致组中的每一个成员，其first键是一个当前回环成员组（所有新一致组成员包括所有回环组），
							//      second键则是一致性次数（继承了一致性则是原一致性加1，否则是0）

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();	// QXC：当前关键帧的回环成员组中，没有任何一组使一致性次数达标，则当前关键帧可以舍弃（可以被设置为坏帧）
        return false;
    }
    else	// QXC：这时当前关键帧比较有用，暂时不能被舍弃
    {
        return true;
    }

    // QXC：不应当来到这里
    mpCurrentKF->SetErase();
    return false;
}

// QXC：针对当前关键帧，尝试从备选回环成员中，选出和其计算得到的相似变换最符合要求者，在计算相似变换的同时，建立当前关键帧中特征点/地图点与选定回环成员共及其视图可见地图点的匹配，
//      如果有满足要求的相似变换和足够的地图点匹配数目，则返回true
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();	// QXC：获得”具有足够一致性的回环成员“的个数

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;		// QXC：当某个回环成员是坏帧或者其与当前关键帧匹配上的地图点数量过少时，该容器中对应的标志位为true
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    // QXC：对所有的回环成员进行遍历，对当前帧和回环成员帧中的地图点进行匹配，建立可能是同一地图点的匹配关系
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();	// QXC：所有回环成员现在不可被擦除

        if(pKF->isBad())	// QXC：前面的SetNotErase只是防止被删除（设置为坏帧），但在那之前可能已经被设置为坏帧了
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);	// QXC：利用BoW对特征的分类，将当前关键帧和回环成员中可能是相同点的地图点找出来

        if(nmatches<20)		// QXC：匹配数不够多时舍弃该回环成员
        {
            vbDiscarded[i] = true;
            continue;
        }
        else			// QXC：匹配数够多时针对该回环成员构造Sim3Solver对象
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);	// QXC：第四个参数单目情况始终为false
            pSolver->SetRansacParameters(0.99,20,300);		// QXC：至少20个内点，本Sim3Solver对象最多进行300次迭代
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;		// QXC：能来到这里说明没有continue，也就是说这个回环成员被接受了
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)	// QXC：始终进行回环成员和当前关键帧间的相似变换计算，直到找到合适的相似变换，或所有回环成员都经过了足够多次的迭代而全部被舍弃完
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])		// QXC：筛除坏帧和匹配点不够多的帧（回环成员）
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm = pSolver->iterate(5,bNoMore,vbInliers,nInliers);	// QXC：迭代求取内点数最多时的相似变换，或者当内点数目始终达不到阈值要求时返回空矩阵

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)		// QXC：匹配点数小于最小内点数，或迭代总次数（每调用一次iterate，总迭代次数就将至少增加5）超出最大阈值（前面设置为300），会造成bNoMore为true
            {
                vbDiscarded[i]=true;
                nCandidates--;		// QXC：bNoMore为true表明经过了足够多次数的迭代，本回环成员和当前关键帧之间还是无法找到合适的相似变换，因此舍弃本回环成员
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())	// QXC：Scm非空表示获得了一个内点数足够多的相似变换
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];		// QXC：得到关键帧1中所有内点对应的关键帧2中的地图点指针
                }

                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);	// QXC：利用相似变换，尝试在两关键帧暂未匹配的地图点间建立匹配关系

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);	// QXC：根据内点在两关键帧中的相机系坐标，优化相似变换，根据优化结果筛选内点

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)	// QXC：当内点数足够多时即认为找到了合适的相似变换，记录此时的回环成员、相似变换及地图点匹配关系
                {
                    bMatch = true;
                    mpMatchedKF = pKF;		// QXC：记录完成相似变换计算的回环成员
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    mg2oScw = gScm*gSmw;			// QXC：记录相似变换
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;	// QXC：记录地图点匹配关系
                    break;
                }
            }
        }
    }

    if(!bMatch)		// QXC：bMatch为false说明没有找到合适的回环成员，进而没有完成相似变换的计算，前面的while循环是由于所有回环成员都被舍弃而退出的
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();	// QXC：所有回环成员都可以被擦除
        mpCurrentKF->SetErase();				// QXC：当前关键帧也可以被擦除
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);		// QXC：在容器尾部加上mpMatchedKF
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)	// QXC：不重复的存储回环成员的共视图关键帧能观测到的所有地图点
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);	// QXC：尝试用回环成员共视图可见的地图点填充mvpCurrentMatchedPoints中的NULL位置

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)	// QXC：最终总的回环地图点数目大于40时才接受该回环成员
    {
        for(int i=0; i<nInitialCandidates; i++)			// QXC：除了完成回环匹配的回环成员外，其他备选回环成员现在都可以被擦除了
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else			// QXC：最终回环地图点数不足时，当前关键帧和所有回环成员都可以被擦除
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

// QXC：基于回环成员计算得到的到当前关键帧的相似变换，校正当前关键帧及其共视图能观测到的所有地图点位置，并和回环帧及其共视图中的地图点进行fuse。
//      然后据此得到回环边（当前关键帧及其共视图关键帧中的某一帧A，以及回环关键帧及其共视图关键帧中的某一帧B，它们通过共视地图点建立的边）。
//      利用回环边、spanning tree、以前回环边、共视点超过100的共视边等，进行essential graph的位姿图优化，根据关键帧的优化结果修正地图点位置。
//      在当前关键帧和回环关键帧中分别添加对方为回环成员，这将成为以后的”以前回环边“。
//      发起globalBA对全局地图的所有关键帧和地图点进行优化（发起了一个新线程，对时间不做约束，没完全执行完时就有可能因又调用了一次本函数而被打断）。
//      若globalBA能够执行完，则根据其优化结果尽可能多的修正关键帧位姿和地图点位置。
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();	// QXC：当LocalMapping被要求stop之后，会在一个while循环中一直等待，直到调用LocalMapping::Release才会使其恢复

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())		// QXC：如果有globalBA正在执行，则打断之
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;		// QXC：改变量将在RunGlobalBundleAdjustment函数中用于判断globalBA是否时因被打断而结束

        if(mpThreadGBA)		// QXC：释放globalBA线程指针
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation	// QXC：获得当前关键帧的共视图关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

		// QXC：对当前关键帧及其共视图中的每个关键帧，获得它们的修正相似变换（回环处世界系到关键帧相机系）和未修正相似变换（当前世界系到关键帧相机系）
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

			// QXC：获得修正后的相似变换，指得是回环时刻的世界系到pKFi相机系的相似变换
            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;		// QXC：获得修正过的世界系（回环成员时刻）到pKFi相机系的相似变换
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;	// QXC：键值为当前关键帧及其附近共视图关键帧的指针，内容为回环世界系到该关键帧相机系的修正过的相似变换
            }

            // QXC：获得实时世界系到pKFi相机系的相似变换
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;		// QXC：键值为当前关键帧及其附近共视图关键帧的指针，内容为当前世界系到该关键帧相机系的相似变换（实际即当前估计位姿）
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // QXC：将当前关键帧及其共视图中关键帧观测到的所有地图点的坐标，都修正到回环成员附近时刻的世界系下，并通过将相似变换中的尺度归一化，将当前关键帧及其共视图中的关键帧位姿进行类似修正。
        //      注意，此时当前帧及其共视图中关键帧观测到的地图点，其指针还是未和回环成员进行fuse的。ComputeSim3中在进行地图点匹配时，是用了关键帧外的另一个容器来存储地图点匹配关系。
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
			// QXC：利用相似变换修正世界系的地图点坐标，将当前帧共视图关键帧观测到的地图点世界坐标修正到回环时刻附近世界系下
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));	// QXC：得到的是当前附近地图点在回环时刻世界系的坐标

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);		// QXC：直接用这个修正过的世界坐标来给地图点的世界坐标赋值
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)		// QXC：用回环成员（及其共视图）中观测到的地图点，取代对应的当前关键帧中相应地图点
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);	// QXC：针对当前帧及其共视图中的每一关键帧，尝试将回环共视图观测到的所有地图点和该关键帧的特征点进行匹配，如果匹配上且该点已经是非坏地图点，
									//      则Replace之，若其不是地图点，则添加新地图点。


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;	// QXC：最终其键值将为当前关键帧及其共视图关键帧（指针），内容set容器中只保留键关键帧与回环关键帧及其共视图关键帧间的联系
    // QXC：筛选出当前关键帧及其共视图中关键帧，与回环成员关键帧及其共视图关键帧之间的联系。（即筛选出回环边）
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();	// QXC：当前关键帧共视图关键帧的共视图关键帧（二层共视图）（更新共视连接之前的）

        // Update connections. Detect new links.
        pKFi->UpdateConnections();	// QXC：由于前面SearchAndFuse函数的功能，这将建立当前帧共视图中的关键帧和回环成员及其共视图关键帧的联系
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();	// QXC：建立pKFi（属于当前关键帧的共视图）和其有共视地图点的所有关键帧（此时包括了共视图和回环）的map映射
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);	// QXC：LoopConnections中，将当前关键帧及其共视图中各关键帧，与之前和它们有连接（共视）的关键帧解除联系
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);		// QXC：LoopConnections中，对于当前关键帧及其共视图中各关键帧，将所有这些关键帧之间的联系解除
        }
    }

    // Optimize graph
    // QXC：以位姿图的方式优化essential graph（包括当前回环边、spanning tree、以前回环边、共视点超过100的共视边），并根据优化结果将地图点的世界坐标进行修正
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();	// QXC：更新全局地图发生较大变化的次数

    // Add loop edge	// QXC：在mpMatchedKF和mpCurrentKF中分别添加对方为自己的回环边端点，在以后的OptimizeEssentialGraph中会用到
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

// QXC：针对当前帧及其共视图中的每一关键帧，尝试将回环共视图观测到的所有地图点和该关键帧的特征点进行匹配，如果匹配上且该点已经是非坏地图点，则Replace之，若其不是地图点，则添加新地图点
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);	// QXC：尝试用回环成员共视图观测到的地图点取代pKF中的地图点，或为其增加新地图点

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];	// QXC：vpReplacePoints中存储了要被Replace的地图点坐标
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}

// QXC：由Tracking::Reset调用，请求对LoopClosing进行reset
void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

// QXC：当通过mbResetRequested标志位要求reset时，清空最近添加关键帧队列，重置上次回环关键帧ID
void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

// QXC：由LoopClosing::CorrectLoop函数发起的globalBA线程的入口函数，将对全局地图进行globalBA（包括所有关键帧位姿和地图点坐标）。
//      若本函数（线程）执行完之前没有新的globalBA发起，且globalBA不是被打断而退出的，
//      则利用globalBA优化结果尽可能的对全局地图中（未在此次globalBA中优化的）的关键帧位姿和特征点位置进行修正；
//      若本函数（线程）执行完之前没有新的globalBA发起，但本次globalBA是被打断退出的，则只是简单的将标识globalBA是否在运行和是否结束的标志位置为相应逻辑值；
//      若有新的globalBA在本次globalBA期间发起而将本次globalBA打断，则直接返回。
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);	// QXC：进行globalBA，优化全局地图中的所有关键帧位姿和地图点位置。

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)		// QXC：当判断idx与mnFullBAIdx不符，说明有新的globalBA要发生，前面的GlobalBundleAdjustemnt要么是被打断退出，
            return;			//      要么是刚结束又要有新的globalBA发生而失去了意义，对于这样的globalBA不予采纳。

        if(!mbStopGBA)		// QXC：说明globalBA不是被打断而退出的
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())		// QXC：LocalMapping未终结的情况下，若其未停止时则始终等待其停止
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());	// ？QXC：【推测】lpKFtoCheck中应当是不互为父子的关键帧

	    // QXC：将mvpKeyFrameOrigins能覆盖到的所有spanning tree成员关键帧的位姿都进行修正
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
		// QXC：对pKF的所有未被globalBA优化的子关键帧，利用pKF和子关键帧的（未修正）位姿求得相对位姿，再利用pKF修正后的位姿和上述相对位姿求得子帧修正后的位姿
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);	// QXC：把这个子关键帧添加到list容器尾部，后面也以其为pKF（父关键帧）来修正（未被globalBA修正过的）其子关键帧位姿。
							//      这样可能会增加list中的重复元素，但由于Spanning Tree的单向不循环、一子只有一父的特点，不会没有尽头的添加下去。
							//      如果lpKFtoCheck中是不互为父子的关键帧（推测认为应当是这样的），则不会有重复现象。
                }

                pKF->mTcwBefGBA = pKF->GetPose();	// QXC：存储未被globalBA修正的相机位姿（注意如果list中有重复元素，本句和下一句会出现问题，因此推断list不应有重复元素）
                pKF->SetPose(pKF->mTcwGBA);		// QXC：将修正后的相机位姿（mTcwGBA）赋值给正经百八应当存储关键帧位姿的成员变量（Tcw）
                lpKFtoCheck.pop_front();		// QXC：pKF完成了在globalBA校正位姿中作为父关键帧的使命，可将其从list容器中删除了
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

	    // QXC：对于未被globalBA直接修正的地图点，若它的参考关键帧位姿被修正了，则依据其参考关键帧位姿和地图点相机系坐标修正该地图点世界系位置
            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)	// QXC：该地图点坐标已经在globalBA中修正过的情况
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else					// QXC：该地图点坐标未在globalBA中修正过的情况
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)	// QXC：如果其参考帧并没有被globalBA优化，则不修正该地图点坐标
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();		// QXC：局部地图线程可以继续进行

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

// QXC：在程序的main函数中，当所有图片都读取完毕后，将调用System::Shutdown函数，该函数将调用本函数，示意线程终结（finish）
void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// QXC：返回mbFinishRequested，当其为true时表示线程终结（finish）。
//      可以看到，Run函数中调用该函数返回true时，都直接跳出大while循环。
bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

// QXC：设置mbFinished标志位，示意线程已终结。该标志位会在System::Shutdown中查询
void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

// QXC：System::Shutdown中调用，查询线程是否终结
bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
