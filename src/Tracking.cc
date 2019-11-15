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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];		// QXC：EuRoc.yaml中没有这一项

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;				// QXC：每一秒最少要插入一个关键帧

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);		// 单目的时候每帧图可提取特征点翻倍

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// QXC：单目情况对每一帧输入图像和时标进行处理，将图像转化为Frame格式，并调用Track函数进行具体处理
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;				// QXC疑问：既然这里使用了赋值。为何还要传入“常值引用”类型的参数	// 解答：因为这里是传给成员的，光用形参不能实现其他函数里的调用

    // QXC：将图像转换为灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)		// QXC：4通道是增加了一个alpha通道,表示透明度
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    // QXC：再else就表明直接是灰度图了，这个时候mbRGB标志位都用不上了，所以再mbRGB声明处的注释中有“ignored if grayscale”这一句

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)	// QXC:mState的初值为NO_IMAGES_YET
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);	// QXC：单目貌似没有mbf和mThDepth	// QXC：初始化的时候使用mpIniORBextractor使特征点翻倍，增强初始化的能力
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);	// QXC：初始化完成后的其他帧使用mpORBextractorLeft，降低提取特征的数目从而提高运算速度

    Track();

    return mCurrentFrame.mTcw.clone();
}

// QXC：Tracking线程中针对每一帧图像的具体处理函数，包括进行初始化、跟踪和重定位等
void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)		// QXC：进行初始化
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();		// QXC：如果初始化成功，mState会被设置为OK

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else				// QXC：初始化完成后针对一般帧的操作
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)	// QXC：正常状态的track流程
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();	// QXC：检查上一帧中的地图点有没有被replace的，如果有则修正它

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)	// QXC：没有速度模型或刚重定位不久时（刚初始化不久）都应当“跟踪参考关键帧”
                {
                    bOK = TrackReferenceKeyFrame();	// QXC：将当前帧中的特征和参考帧的地图点进行匹配，利用匹配得到的地图点对当前帧位姿进行PnP优化，并根据优化过程剔除地图点外点，返回值为跟踪地图点是否比10多的逻辑运算结果
                }
                else
                {
                    bOK = TrackWithMotionModel();	// QXC：利用速度模型对当前帧的位姿进行预测，从而加速匹配过程，并对当前帧位姿进行优化。但如果匹配点数太少时还是要依赖采用比较暴力的匹配方法的TrackReferenceKeyFrame
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();	// QXC：如果上述基于速度模型的匹配失败了（匹配点数过少），则进行比较暴力的匹配
                }
            }
            else		// QXC：这个else表示mState==LOST
            {
                bOK = Relocalization();			// QXC：尝试找到一个最相关的关键帧（优化结果的内点地图点超过50）来重定位当前帧的位姿
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();		// QXC：（1）处
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;		// QXC：如果重定位成功了，就不管前面（1）处跟踪的结果了
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();	// QXC：更新Tracking中维护的局部关键帧和局部地图点，并尝试给当前帧补充地图点，再次对当前帧位姿进行优化，根据优化过程中决定的地图点内点数决定返回值
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
			// QXC：将在别的关键帧中无观测的地图点剔除
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints	// QXC：暂时不清楚mlpTemporalPoints的用途	// QXC：单目不使用该成员
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())	// QXC：判断是否需要添加关键帧（最宽松条件）
                CreateNewKeyFrame();		// QXC：需要添加关键帧时利用当前帧创建一个关键帧

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();	// QXC：调用这个函数最终将只调用Tracking::Reset
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())	// QXC：mTcw非空时
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();		// QXC：刚初始化完成回来时这里的mCurrentFrame表达的帧和mCurrentFrame.mpReferenceKF表达的关键帧其实是同一帧，因此这个式子的结果为eye(4,4)
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else				// QXC：mTcw空时
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()	// QXC：只有当两帧的特征点数都大于100时才有可能（还要求匹配点对数也大于100）初始化成功
{

    if(!mpInitializer)		// QXC：只要二级if（特征点数超过100）进去过一次，就不会再进入该if，而是进入下面的else(除非重新设定初始化基准帧)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)		// QXC：只有特征点数大于100的帧才有可能成为初始化基准帧
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);		// QXC:设定初始化基准帧

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)		// QXC：特征点数小于等于100时，下一帧重新设定初始化基准帧
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);	// QXC：进行特征匹配，具有一定的筛选功能

        // Check if there are enough correspondences
        if(nmatches<100)					// QXC：匹配点个数也必须大于100
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // QXC：if中的Initialize函数进行了初始化工作，运用了RANSAC分别恢复最好的单映阵和基础矩阵，并选择最好的一种模型来恢复运动，还对质量好的特征进行了三角化，它们的世界坐标存在mvIniP3D中
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))	// QXC:vbTriangulated中,成功三角化且视差较大的点标志位为true,其余为false
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)	// QXC:mvIniMatches的下标i对应第一帧中的特征点ID,其对应元素对应于第二帧中与其匹配的特征点ID(如果不是-1的话)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])	// QXC:剔除成功匹配,但三角化失败或视差较小的点
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));	// QXC:成功初始化后,将初始帧(即初始化的基准帧)的位姿设置为eye矩阵,也就是说世界系就是初始帧的相机系
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);				// QXC:成功初始化后,初始化的后一帧的位姿即按照初始化的结果赋值,注意!!!其中平移量为单位向量

            CreateInitialMapMonocular();			// QXC:在该函数体中可以看到,成功初始化的两帧都被设置为关键帧
        }
        // QXC上述if如果结果为false,则保持初始化基准帧不变,等待下一帧来了再尝试初始化,若下一帧到来后触发了特征点不足或匹配点对不足的条件,则重新设置初始化基准帧
    }
}

// QXC：将初始化基准帧和当前帧作为头两个关键帧，并利用单映阵或基础矩阵恢复它们的相对位姿关系，认为基准帧的位姿为单位阵（即世界坐标系为第一帧的相机系），
//      对地图点进行三角化，并将它们的深度中值设置为1，并依此初始化尺度
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // QXC:计算最初两个关键帧的BoW,细节没仔细研究	// QXC：计算关键帧的BoW用于后面跟踪关键帧时的匹配
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)	// QXC:成为地图点的特征点要求是匹配成功且三角化成功且视差不小的点,mvIniMatches在调用本函数之前已经经过了处理,使得其中不为-1的元素就是满足要求的点
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);		// QXC:mvIniP3D中存储的是初始化基准帧相机系下的特征点位置,可以认为初始化基准帧相机系就是世界坐标系

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);	// QXC:创建地图点时只将其参考帧设置为除了初始化基准帧之外的观察到它的第一个关键帧

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();		// QXC:用到其他描述符的距离为中位值的描述子作为该地图点的描述子
        pMP->UpdateNormalAndDepth();			// QXC:更新特征点平均观测矢量(?意义),以及相对于参考帧的深度/距离上下限	// ？QXC：放在深度归一化之前不会有问题吗？？

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();	// QXC:建立与该关键帧有共视地图点的其他关键帧的联系(保存指针,记录共视地图点数量之类的操作)，并且在共视点很多的其他关键帧中添加联系（如果没有超过阈值的则选共视点最多的关键帧）
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);	// 优化全局Map中所有关键帧（目前只有两帧）的位姿和地图点的位置

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);	// QXC：获得初始帧中所有地图点深度的中位值
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)	// QXC：第二个条件判断pKFcur中被其他帧观测到次数大于“1”的地图点个数是否小于100（在这里指的就是pKFcur中地图点被初始化基准帧观测到的个数是否小于100）
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;	// QXC：重新尺度化了初始化中第二帧的位移量，不再是单位矢量
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);		// QXC：用同样的尺度因子重新尺度化地图点坐标
        }
    }

    // QXC：向LocalMapping中加入两个新关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    
    // QXC：更新“上一关键帧信息”
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    // QXC：更新Tracking维护的局部地图中的关键帧和地图点
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    
    // QXC：设置Tracking中的参考关键帧
    mpReferenceKF = pKFcur;
    
    // QXC：修正当前帧的位姿，设定当前帧的参考关键帧（为其本身构造的关键帧）
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mCurrentFrame.mpReferenceKF = pKFcur;

    // QXC：更新“上一帧”
    mLastFrame = Frame(mCurrentFrame);

    // QXC：更新全局地图中的参考地图点
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    // ？QXC：更新全局地图中的mvpKeyFrameOrigins（？该成员的完整用途仍未知）
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

// QXC：检查上一帧的地图点有没有没replace的，如果有，则在本地成员中修正它
//      LocalMapping::SearchInNeighbors()函数会fuse地图点，此时可能发生地图点Replacement
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;	// QXC：mLastFrame是通过构造函数构造的，其mvpMapPoints成员是复制得到的
            }
        }
    }
}

// QXC：将当前帧中的特征和参考关键帧的地图点进行匹配，利用匹配得到的地图点对当前帧位姿进行PnP优化，并根据优化过程剔除地图点外点，返回值为跟踪地图点是否比10多的逻辑运算结果
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // QXC：利用BoW对特征的分类加速匹配过程，将mCurrentFrame中的特征点和mpReferenceKF中的地图点进行匹配
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;	// QXC：返回值将影响Tracking::mState的值，可见只要有一帧和参考关键帧的共视地图点（注意不仅仅是特征点）小于15就认为跟踪LOST

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // QXC：根据上文，这里是进行“a PnP solver”
    // QXC：前面的if(nmatches<15)判断条件使得这里调用该函数时，其内部的nInitialCorrespondences至少是15
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers		// QXC：上面的PoseOptimization函数对mCurrentFrame.mvbOutlier进行了赋值，因此可用于剔除外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])	// QXC：首先得是地图点
        {
            if(mCurrentFrame.mvbOutlier[i])	// QXC：然后判断是不是外点
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;	// QXC：当把地图点指针mvpMapPoints[i]赋空之后，mvbOutlier[i]也就没有意义，因此将其赋为初始值false
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;	// QXC：这个和文字有点不符，mCurrentFrame明明是第一次看不到该地图点的帧
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)	// QXC：跟踪成功还要求该地图点在别的帧中还能观测到
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;	// QXC：跟踪到的地图点个数大于等于10时才算跟踪成功
}

// QXC：对单目来说，本函数通过参考帧位姿和相对位姿相乘的方式设置了mLastFrame的位姿
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;
    
    // QXC：后面是双目的代码

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

// QXC：利用速度模型对当前帧位姿进行估计，并利用估计结果进行当前帧特征和上一帧地图点的匹配，如果匹配数足够多，则利用匹配结果对当前帧位姿进行优化，并利用优化过程剔除地图点外点
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();		// QXC：对于单目情形，只是简单的利用参考帧位姿和相对位姿更新了上一帧位姿mLastFrame.mTcw

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);		// QXC：利用常速度模型估计当前帧位姿		// QXC：这个地方可以利用IMU改进

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);	// QXC：利用当前帧位姿速度模型预测结果，进行地图点的匹配

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);	// QXC：th控制粗搜索范围
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)		// QXC：TrackingOnly模式下才会进入这里
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

// QXC：更新Tracking中维护的局部关键帧和局部地图点，并尝试给当前帧补充地图点，再次对当前帧位姿进行优化，根据优化过程中决定的地图点内点数决定返回值
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();		// QXC：更新mvpLocalKeyFrames和mvpLocalMapPoints

    SearchLocalPoints();	// QXC：尝试从mvpLocalMapPoints中为当前帧添加可观测到的地图点

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);		// QXC：再进行一次PnP
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();		// QXC：增加地图点“被找到次数”
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)		// QXC：其他关键帧对该地图点也有观测时
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)	// ？QXC：为何单目情况不将指针赋为NULL：如果该帧可能成为关键帧，则BA会决定这些当前是外点的地图点到底是不是外点，所以暂时还保留它们，在判断添加关键帧之后会将这些地图点的指针赋空
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)		// QXC：该if中的第一个条件表示当前帧位于刚刚重定位结束后的下一个关键帧插入之前
        return false;		// QXC：刚进行完重定位不久时（或刚进行完重定位时，此时mnLastRelocFrameId=mCurrentFrame.mnId），本函数要求的内点个数更多

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


// QXC：根据一系列条件判断是否插入关键帧，TrackingOnly模式下不插入；局部地图被锁定时不插入；全局地图关键帧数目超过最大限制时，需在最大容忍间隔时插入关键帧；
//      上述条件都不满足时则判断是否到达最大容忍间隔或localmap是否空闲，且当前帧跟踪的地图点是否在指定范围内：如果是且localmap空闲则返回true，如果不空闲则通知其准备打断BA。
// QXC：单目情况只有在localmap空闲且当前帧跟踪的地图点数目在指定范围内时，该函数才会返回true
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)		// QXC：TrackingOnly模式不插入关键帧
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())		// QXC：局部地图被锁定时不插入关键帧
        return false;

    const int nKFs = mpMap->KeyFramesInMap();	// QXC：得到全局地图中的关键帧数目

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)	// QXC：全局地图关键帧数目超过最大限制时，得等到最大容忍间隔才插入关键帧
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);		// QXC：返回参考关键帧观测到的所有地图点在所有关键帧中观测数量大于nMinObs的点数量

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();		// QXC：由localmap线程决定

    // Check how many "close" points are being tracked and how many could be potentially created.		// QXC：双目情况考虑，单目情况不考虑
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);					// QXC：单目情况该标志位始终为0

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);	// QXC：由于mMinFrames为0，因此只要localmap空闲就可以插入关键帧，增强跟踪能力
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;	// QXC：单目情况该标志位始终为0
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);	// QXC：？避免让冗余地图点很多的帧成为关键帧？

    // QXC：单目情况下c1c始终为0，于是该判断的前一部分是判断是否到达最大容忍的关键帧插入间隔，或者localmap是不是空闲；第二部分判断当前帧跟踪的地图点数是否位于设定的区间内
    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;	// QXC：满足了上述条件后，还需要localmap空闲才可以插入关键帧
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)	// QXC：双目情况，单目不考虑，直接转入下面的else返回false
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

// QXC：利用当前帧创建关键帧，更改参考关键帧（Tracker和当前帧都更改），并向localmap增加关键帧
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))	// QXC：单目情况能进入这里说明localmap的mbStopped是false，于是SetNotStop函数不会返回false
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)	// QXC：双目情况考虑，单目情况无视
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);		// QXC：向LocalMapping添加新关键帧，并发出打断BA标志

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

// QXC：挑出mvpLocalMapPoints中还不包含在当前帧地图点队列中，但经计算可能被当前帧观测到的地图点，尝试用当前帧中的特征和其进行匹配，如果有满足要求的匹配，则给当前帧地图点队列加上这个地图点
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;	// ？QXC：不解，为何将此标志位复位？		// QXC：下面的SearchByProjection函数中不会处理该标志位为false的点
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)		// QXC：这类点要么是外点，要么已经被当前帧观测到
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

// QXC：更新Tracking维护的局部地图，包括更新mvpLocalKeyFrames和mvpLocalMapPoints
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();	// QXC：更新Tracking的局部关键帧队列mvpLocalKeyFrames，更新Tracking的参考关键帧mpReferenceKF以及当前帧的参考关键帧mCurrentFrame.mpReferenceKF
    UpdateLocalPoints();	// QXC：更新Tracking的局部地图点队列mvpLocalMapPoints，以局部关键帧mvpLocalKeyFrames观测到的地图点为基准进行更新（剔除坏点，不重复）
}

// QXC：更新mvpLocalMapPoints，其中元素为mvpLocalKeyFrames的所有可视地图点（不重复，非坏点）
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

// QXC：更新mvpLocalKeyFrames，其包括所有和当前帧mCurrentFrame有共视地图点的关键帧，（如果数量不超过80时还将包括）这些关键帧的最佳共视关键帧和一个子帧，以及某个关键帧的父帧
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;		// QXC：second键存储first键指示的KF和当前帧的共视地图点数目
    for(int i=0; i<mCurrentFrame.N; i++)	// QXC：对于当前帧的所有特征点（会筛选出地图点）统计keyframeCounter，因此后面分析的都是关键帧和当前帧共视的地图点
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)		// QXC：限制mvpLocalKeyFrames的规模
            break;

        KeyFrame* pKF = *itKF;

		// QXC：对于和当前帧有共视地图点的某一关键帧，向mvpLocalKeyFrames中添加一个该关键帧的较佳共视关键帧（称之为较佳是因为是在前10最佳共视KF中选出来的）（避免重复添加）
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)	// QXC：mnTrackReferenceForFrame如果等于mCurrentFrame.mnId说明该关键帧作为和mCurrentFrame有直接共视点的关键帧，之前已经放入mvpLocalKeyFrames中了
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;	// QXC：对于和当前帧有共视特征点的每个关键帧，间接的加入mvpLocalKeyFrames中的最多只有一帧
                }
            }
        }

        // QXC：对于和当前帧有共视地图点的某一关键帧，向mvpLocalKeyFrames中添加一个该关键帧的子关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();	// QXC：根据UpdateConnections函数，子系的概念是指：当某关键帧A位于关键帧B的顺序共视帧队列首位，则B是A的子系
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // QXC：对于和当前帧有共视地图点的某一关键帧，向mvpLocalKeyFrames中添加一个该关键帧的父关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;		// QXC：这个break的存在，使得一旦添加了某关键帧的父帧，则给mvpLocalKeyFrames添加成员的整个for循环就退出了	// ？QXC：感觉这个break不应当存在
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;					// QXC：与当前帧共视地图点最多的关键帧设置为Tracking线程的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;		// QXC：与当前帧共视地图点最多的关键帧设置为当前帧的参考关键帧
    }
}

// QXC：利用BoW，从mpKeyFrameDB中找出可能与当前帧产生关联的关键帧，并试图利用这些关键帧（实际上只选中其中一帧）来求解（重定位）当前帧的位姿。只有在优化的内点数大于50时才算重定位成功
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);	// QXC：由BoW获得可用于重定位当前帧的关键帧（们）

    if(vpCandidateKFs.empty())		// QXC：根据DetectRelocalizationCandidates代码，vpCandidateKFs为空可能的原因有：当前帧和BoW数据库没有相同元素
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;	// QXC：存储每个关键帧的地图点
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    // QXC：进行当前帧特征点和所有可能的关键帧中地图点的匹配
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);	// QXC：利用BoW从pKF中匹配出当前帧可能观测到的地图点
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);	// QXC：构造对象的同时会传入地图点和对应特征的相关信息
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);				// QXC：设置RANSAC的相关参数
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // QXC：对于每一个匹配到足够地图点的关键帧，用那些地图点对当前帧位姿进行PnP求解，并根据内点数和阈值的关系决定是否进一步优化/选下一帧优化/确定当前帧位姿结果
    //      bMatch只有在某次调用Optimizer::PoseOptimization进行图优化返回的内点数大于等于50时才会为true，只有这种情形下才会接受重定位结果
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])		// QXC：剔除bad关键帧以及匹配点数不足的关键帧
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);	// QXC：利用和关键帧nKFs中匹配上的地图点PnP求解当前帧位姿，并标识出地图点内点	// QXC：也就是说每一次for循环，其实只是在用一个关键帧中的地图点来做初始PnP

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)		// QXC：bNoMore为true时，iterate的返回值Tcw仍有可能不为空，这是其结果是一个并不突出的当前帧位姿
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();	// QXC：np和当前帧特征点数相同

		// QXC：为当前帧的内点地图点赋指针
                for(int j=0; j<np; j++)
                {
		    // QXC：利用vbInliers（该变量由pSolver->iterate赋值）来筛选当前帧的地图点指针，会使得重定位实际上是选中了一个关键帧中的地图点来对当前帧位姿进行计算
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);		// QXC：sFound中存放了当前帧在某关键帧中暂时匹配到的地图点
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
		    // QXC：放粗匹配标准（特征的描述子距离阈值为100），看看关键帧中是否还有其他地图点可能被当前帧观测到
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);	// QXC：（1）处

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);					// QXC：将存储（1）处给当前帧找到的地图点
                            nadditional = matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);	// QXC：当前帧位姿优化过了，所以这里有意义

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    // QXC：bMatch只有在某次调用Optimizer::PoseOptimization进行图优化返回的内点数大于等于50时才会为true，只有这种情形下才会接受重定位结果
    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
