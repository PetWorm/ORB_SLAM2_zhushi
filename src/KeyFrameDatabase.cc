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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

// QXC：在LoopClosing::DetectLoop中调用，在mvInvertedFile中，向目标关键帧pKF含有的单词对应的关键帧指针容器中，添加目标关键帧
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

// QXC：在KeyFrame::SetBadFlag中调用，删除mvInvertedFile中各单词下关于目标关键帧pKF的信息
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

// QXC：基于某关键帧pKF搜寻回环成员，回环成员满足条件：
//      （1）不位于pKF的共视图关键帧中；
//      （2）和pKF的共有单词数大于阈值；
//      （3）设某个满足（1）和（2）的关键帧为pKFi，则回环成员是某个pKFi，或位于某个pKFi的共视关键帧中；
//      （4）pKFi及其共视图关键帧和pKF的累计BoW分数较高；
//      （5）回环成员是pKFi及其共视图关键帧中，BoW评分最高者。
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();		// QXC：获得pKF未排序的covisibility graph中关键帧指针组成的set
    list<KeyFrame*> lKFsSharingWords;		// QXC：存储不位于pKF的covisibility graph中，且和pKF有相同BoW单词的关键帧

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

	// QXC：找出在关键帧pKF的covisibility graph中，且和pKF有相同BoW单词的关键帧
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];		// QXC：得到包含ID为vit->first的BoW单词的所有关键帧

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId)	// QXC：该if判断防止重复添加
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi))	// QXC：判断关键帧pKFi是否在pKF的covisibility graph中，不在时为true！！！
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);	// QXC：将不在pKF的covisibility graph中的pKFi放入容器中
                    }
                }
                pKFi->mnLoopWords++;	// QXC：mnLoopWords记录pKFi和pKF共有的BoW单词类型数目
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // QXC：存储lKFsSharingWords里（与pKF）相同单词数量大于最小阈值（minCommonWords）的关键帧及它和pKF的BoW字典分数组成的pair
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)		// QXC：得到最大的共有BoW单词数
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;		// QXC：基于最大共有单词数获得最小共有单词数

    int nscores=0;	// QXC：记录lKFsSharingWords里相同单词数量大于最小阈值的帧数

    // Compute similarity score. Retain the matches whose score is higher than minScore
    // QXC：找出lKFsSharingWords中，和pKF共同单词数大于阈值，且BoW分数也大于阈值的关键帧，将关键帧和BoW分数结对（pair）存储
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)		// QXC：挑选出和pKF的共有单词数大于最小阈值的
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)		// QXC：分数必须大于传入的minScore参数
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    // QXC：对于每一个不属于pKF共视图且共有单词数大于阈值的关键帧pKFi，在其共视图关键帧中，寻找不存在于pKF共视图中，且和pKF共有单词数大于阈值的关键帧，
    //      这些关键帧的mLoopScore成员累加得到accScore（代表pKFi或pBestKF）。
    //      同时，在pKFi和其满足上述要求的共视关键帧中，选出mLoopScore成员值最大者作为pBestKF。将accScore和pBestKF结对（pair）存储到容器中。
    //      对每一个pKFi，记录最高的accScore为bestAccScore。
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);	// QXC：lScoreAndMatch中关键帧的covisibility graph关键帧

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)	// QXC：pKF2还要求不在pKF的covisibility graph中，且和pKF共有单词数要大于阈值
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));	// QXC：上述遍历方式会导致lAccScoreAndMatch中多个pair的second键值重复
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))		// QXC：避免重复添加
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    // QXC：返回的容器中的成员，是pKF的共有单词数大于阈值的（不属于其共视图的）关键帧A中，其共视图关键帧（还需满足和pKF共有单词数大于阈值）和pKF分数的累计值较高时，
    //      其中（包括A及A的大于阈值的共视图关键帧）分数最高的关键帧。
    return vpLoopCandidates;
}

// QXC：获得可用于重定位F的关键帧：
//      根据本对象存储的所有关键帧，首先计算和帧F的相同单词数，并筛选出单词数较多的关键帧；
//      再根据这些关键帧周围covisi关键帧的累计相似度分数（筛选累计分数较高的”covisi关键帧群“），筛选出可用于重定位的关键帧（这些最终的关键帧在周围有限的covisi关键帧中相似度分数最高）
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

	// QXC：对帧F中的每个单词，找到包含该单词的关键帧，在某关键帧包含多个F中的单词时，关键帧的计数器成员mnRelocWords会记录次数
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

	    // QXC：对于每一个包含vit->first所指单词的关键帧，将其成员mnRelocQuery赋帧F的ID；将其压入lKFsSharingWords；其成员mnRelocWords加1表示与帧F相同的单词又多发现一个
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)		// QXC：进入该if表示关键帧pKFi被首次发现和帧F有相同的单词，因此将其成员mnRelocWords置0，并将F的ID赋给成员mnRelocQuery
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);	// QXC：能进入这个if说明pKFi首次被发现和帧F有相同的单词，因此会在这里将关键帧push_back
                }
                pKFi->mnRelocWords++;	// QXC：走到该代码表示关键帧pKFi与帧F有相同的单词，因此给关键帧成员mnRelocWords加1
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)		// QXC：获得最大相同单词数
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;		// QXC：用最大相同单词数的80%来确定阈值

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)		// QXC：对每个和帧F具有足够多相同单词的关键帧，求取BoW相似度分数
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)		// QXC：只考虑相同单词数超过阈值的
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())		// QXC：这里为空几乎不可能
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;	// QXC：存储与帧F有较多相同单词的关键帧，及该关键帧周围的关键帧累计相似度分数
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // QXC：对于lScoreAndMatch中的每个关键帧A，在其附近的（至多）10个covisibilty关键帧中，找出和帧F有相同单词的，并且在包括A的所有这些关键帧中，找出相同单词数最多的那个
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)	// QXC：筛选出当前关键帧pKFi附近也和帧F有相同单词的关键帧
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));	// QXC：（1）处：注意，采用的搜索方式可能会导致lAccScoreAndMatch中不同pair的second键值却相同
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;		// QXC：存储已经在vpRelocCandidates队列中的元素，用于避免重复项【参见（1）处注释】
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))		// QXC：根据（1）处的注释，lAccScoreAndMatch不同元素的second键值可能相同，所以这里要做判断是否已经存过
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
