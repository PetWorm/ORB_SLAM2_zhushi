/**
* This file is a test script based on ORB-SLAM2.
* Editted by Xiaochen Qiu (Beihang University)
*
* Original description of ORB-SLAM2 is shown below:
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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Frame.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Initializer.h"

using namespace std;
using namespace ORB_SLAM2;

void LoadImages(const string &strImagePath, const string &strPathTimes,
				vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
	if(argc != 5)
	{
		cerr << endl << "Usage: ./qxctest_euroc path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file" << endl;
		return 1;
	}

	cout << endl;

	// Construct vocabulary object
	ORBVocabulary* pVocabulary = new ORBVocabulary();
// 	bool bVocLoad = pVocabulary->loadFromTextFile(string(argv[1]));
// 	if(!bVocLoad)
// 	{
// 		cerr << "Wrong path to vocabulary. " << endl;
// 		cerr << "Falied to open at: " << argv[1] << endl;
// 		exit(-1);
// 	}
	cout << "Vocabulary loaded!" << endl << endl;

	// Read setting file;
	cv::FileStorage fsSettings(string(argv[2]), cv::FileStorage::READ);

	// Read color order (RGB or BGR)
	int bRGB = fsSettings["Camera.RGB"];
	if(bRGB)
		cout << "- color order: RGB (ignored if grayscale)" << endl << endl;
	else
		cout << "- color order: BGR (ignored if grayscale)" << endl << endl;

	// Read parameters for ORB feature extractor
	int nFeatures = fsSettings["ORBextractor.nFeatures"];
	float fScaleFactor = fsSettings["ORBextractor.scaleFactor"];
	int nLevels = fsSettings["ORBextractor.nLevels"];
	int fIniThFAST = fsSettings["ORBextractor.iniThFAST"];
	int fMinThFAST = fsSettings["ORBextractor.minThFAST"];
// 	cout << nFeatures << " " << fScaleFactor << " " << nLevels << " " << fIniThFAST << " " << fMinThFAST << endl;

	// Construct ORB feature extractor objects
	ORBextractor* pIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
// 	ORBextractor* pORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

	// Read camera internal matrix
	float fx = fsSettings["Camera.fx"];
	float fy = fsSettings["Camera.fy"];
	float cx = fsSettings["Camera.cx"];
	float cy = fsSettings["Camera.cy"];
	cv::Mat K = cv::Mat::eye(3,3,CV_32F);
	K.at<float>(0,0) = fx;
	K.at<float>(1,1) = fy;
	K.at<float>(0,2) = cx;
	K.at<float>(1,2) = cy;
// 	cout << K << endl;

	// Read camera distort coefficients
	cv::Mat DistCoef(4,1,CV_32F);
	DistCoef.at<float>(0) = fsSettings["Camera.k1"];
	DistCoef.at<float>(1) = fsSettings["Camera.k2"];
	DistCoef.at<float>(2) = fsSettings["Camera.p1"];
	DistCoef.at<float>(3) = fsSettings["Camera.p2"];
	const float k3 = fsSettings["Camera.k3"];
	if(k3!=0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}

	// Retrieve paths to images
	vector<string> vstrImageFilenames;
	vector<double> vTimestamps;
	LoadImages(string(argv[3]), string(argv[4]), vstrImageFilenames, vTimestamps);

	int nImages = vstrImageFilenames.size();

	if(nImages<=0)
	{
		cerr << "ERROR: Failed to load images" << endl;
		return 1;
	}

	cout << "Ready to Process !" << endl << endl;
	
	// The image pair to be analyzed
	size_t nImgPairID;
	cout << "Please input a postive integer to decide which pair to be analyzed:" << endl;
	cin >> nImgPairID;
	cout << "Pair No." << nImgPairID << " will be analyzed" << endl << endl;
	
	// Simulate feature matching in initialization process of ORB-SLAM2
	cv::Mat im;
	cv::Mat imGray;
	Frame* pFrame1 = static_cast<Frame*>(NULL);
	Frame* pFrame2 = static_cast<Frame*>(NULL);
// 	ORBmatcher matcher(0.9,false);
	ORBmatcher matcher(0.9,true);
	vector<cv::Point2f> vbPrevMatched;
	vector<int> vIniMatches;
	cv::Mat imGray1, imGray2;
	vector<pair<Frame*,Frame*>> matchedImagesPairs;
	vector<vector<int>> vvIniMatches;
	vector<int> vnmatches;
	vector<vector<int>> vvIniMatches_t;
	vector<int> vnmatches_t;
	vector<Initializer*> vpInitializers;
	vector<cv::Mat> vRcw;
	vector<cv::Mat> vtcw;
	vector<vector<cv::Point3f>> vvIniP3D;
	for(int ni=0; ni<nImages; ni++)
	{
		// Read image from file
		im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
		double tframe = vTimestamps[ni];
		if(im.empty())
		{
			cout << "No image! ni = " << ni << endl;
			cout << vstrImageFilenames[ni] << endl;
			break;
		}

		// Transform RGB/RGBA into gray-scale
		if(im.channels()==3)
		{
			if(bRGB)
				cvtColor(im,imGray,CV_RGB2GRAY);
			else
				cvtColor(im,imGray,CV_BGR2GRAY);
		}
		else if(im.channels()==4)		// QXC：4通道是增加了一个alpha通道,表示透明度
		{
			if(bRGB)
				cvtColor(im,imGray,CV_RGBA2GRAY);
			else
				cvtColor(im,imGray,CV_BGRA2GRAY);
		}
		else
			im.copyTo(imGray);

		if(imGray.empty())
		{
			cout << "imGray is empty! ni = " << ni << endl;
			break;
		}

		// Construct frame objects
		Frame frame = Frame(imGray,tframe,pIniORBextractor,pVocabulary,K,DistCoef,0,0);

		// Try to find images pairs which fulfill the condition of initialization
		if(!pFrame1)	// 首帧还未建立
		{
			if(frame.mvKeys.size()>100)		// 只保留特征点数超过100的为首帧
			{
				if(matchedImagesPairs.size()==nImgPairID-1)		// 分析第nImgPairID对图像
					imGray.copyTo(imGray1);
				
				pFrame1 = new Frame(frame);
				fill(vIniMatches.begin(),vIniMatches.end(),-1);
			}
		}
		else			// 首帧已经建立
		{
			if(frame.mvKeys.size()<=100)	// 次帧特征点不超过100时重新找首帧
			{
				delete pFrame1;
				pFrame1 = static_cast<Frame*>(NULL);
				fill(vIniMatches.begin(),vIniMatches.end(),-1);
			}
			else							// 次帧特征点超过100，尝试进行匹配
			{
				vbPrevMatched.resize(pFrame1->mvKeysUn.size());
				for(size_t i=0; i<pFrame1->mvKeysUn.size(); i++)
					vbPrevMatched[i] = pFrame1->mvKeysUn[i].pt;

				pFrame2 = new Frame(frame);
				int nmatches = matcher.SearchForInitialization(*pFrame1,*pFrame2,vbPrevMatched,vIniMatches,100);	// 匹配

				if(nmatches<100)		// 匹配数小于100时重新找首帧
				{
					delete pFrame1;
					pFrame1 = static_cast<Frame*>(NULL);
					delete pFrame2;
					pFrame2 = static_cast<Frame*>(NULL);
					fill(vIniMatches.begin(),vIniMatches.end(),-1);
				}
				else					// 匹配数大于等于100时尝试进行初始化
				{
					Initializer* pInitializer = new Initializer(*pFrame1,1.0,200);
					vpInitializers.push_back(pInitializer);
					
					cv::Mat Rcw; // Current Camera Rotation
					cv::Mat tcw; // Current Camera Translation
					vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
					vector<cv::Point3f> vIniP3D;
					if(pInitializer->Initialize(*pFrame2,vIniMatches,Rcw,tcw,vIniP3D,vbTriangulated))	// 初始化成功时，存储该对
					{
						if(matchedImagesPairs.size()==nImgPairID-1)		// 分析第nImgPairID对图像
						imGray.copyTo(imGray2);
						
						/* 成功三角化且视差不小的点的相关信息将被保存，这些点首先是匹配上的点 */
						
						matchedImagesPairs.push_back(make_pair(pFrame1,pFrame2));	// 存储匹配上的两帧
						
						vvIniMatches.push_back(vIniMatches);						// 存储成功匹配点的匹配关系
						vnmatches.push_back(nmatches);								// 存储成功匹配的点的个数
						
						for(size_t i=0; i<vIniMatches.size(); i++)		// 剔除成功匹配，但三角化失败或视差较小的点
						{
							if(vIniMatches[i]>=0 && !vbTriangulated[i])
							{
								vIniMatches[i]=-1;
								nmatches--;
							}
						}
						vvIniMatches_t.push_back(vIniMatches);						// 存储成功三角化且视差不小的匹配关系
						vnmatches_t.push_back(nmatches);							// 存储成功三角化且视差不小的点的个数
						
						vRcw.push_back(Rcw);										// 存储旋转量
						vtcw.push_back(tcw);										// 存储单位平移量
						vvIniP3D.push_back(vIniP3D);								// 存储首帧中的成功三角化且视差不小的点的坐标
						
						//delete pFrame1;		// 这里使用delete会有问题
						pFrame1 = static_cast<Frame*>(NULL);
						//delete pFrame2;
						pFrame2 = static_cast<Frame*>(NULL);
						fill(vIniMatches.begin(),vIniMatches.end(),-1);
						
						cout << "Found No." << matchedImagesPairs.size() << " triangulated pair. With Frame2 of ni = " << ni << endl;
						
						if(matchedImagesPairs.size()==nImgPairID)		// 当到达要分析的匹配对时跳出
							break;
					}
					else																				// 初始化失败时，重新找首帧
					{
						vpInitializers.pop_back();
						delete pInitializer;
					}
				}
			}
		}

		if(ni==nImages-1)
			if(!pFrame2)
				if(pFrame1)
					delete pFrame1;
	}
	
	
	// Visulization of specified matching pair
	if(imGray1.empty())
		cout << "Pair No." << nImgPairID << " do not exist!" << endl;
	else
	{
		cout << endl;
		
		Frame* pF1 = matchedImagesPairs[nImgPairID-1].first;		// 获得匹配的两帧
		Frame* pF2 = matchedImagesPairs[nImgPairID-1].second;
		vector<cv::KeyPoint> vKp1 = pF1->mvKeys;					// 特征点坐标（未修正）
		vector<cv::KeyPoint> vKp2 = pF2->mvKeys;
		vector<int> iniMatches = vvIniMatches[nImgPairID-1];		// 匹配对关系
		vector<int> iniMatches_t = vvIniMatches_t[nImgPairID-1];	// 三角化成功且视差不小的匹配对关系
		cv::Mat im1, im2;
		cvtColor(imGray1,im1,CV_GRAY2BGR);
		cvtColor(imGray2,im2,CV_GRAY2BGR);
		int nmatches = vnmatches[nImgPairID-1];
		int nmatches_t = vnmatches_t[nImgPairID-1];
		for(size_t i=0; i<iniMatches.size(); i++)
		{
			if(iniMatches[i]>0)
			{
				// 将匹配上的特征点标记出来
				cv::Point2f pt1,pt2;
				pt1.x = vKp1[i].pt.x-5;
				pt1.y = vKp1[i].pt.y-5;
				pt2.x = vKp1[i].pt.x+5;
				pt2.y = vKp1[i].pt.y+5;
				cv::rectangle(im1,pt1,pt2,cv::Scalar(0,255,0));
				pt1.x = vKp2[iniMatches[i]].pt.x-5;
				pt1.y = vKp2[iniMatches[i]].pt.y-5;
				pt2.x = vKp2[iniMatches[i]].pt.x+5;
				pt2.y = vKp2[iniMatches[i]].pt.y+5;
				cv::rectangle(im2,pt1,pt2,cv::Scalar(0,255,0));
				
				// 对其中三角化成功且视差不小的点做特殊标记
				if(iniMatches_t[i]>0)
				{
					cv::circle(im1,vKp1[i].pt,3,cv::Scalar(0,255,255),-1);
					cv::circle(im2,vKp2[iniMatches_t[i]].pt,3,cv::Scalar(0,255,255),-1);
				}
			}
		}
		
		cout << "Frame1 has " << vKp1.size() << " key points" << endl;
		cout << "Frame2 has " << vKp2.size() << " key points" << endl;
		cout << "Frame1 and Frame2 have " << nmatches_t << " triangulated points out of " << nmatches << " matched points" << endl;
		
		cv::namedWindow("Frame 1");
		cv::imshow("Frame 1",im1);
		cv::waitKey(0);
		cv::namedWindow("Frame 2");
		cv::imshow("Frame 2",im2);
		cv::waitKey(0);
		cout << "imshow ok" << endl;
	}
	
	// Delete Frame pointers
	for(size_t i=0; i<matchedImagesPairs.size(); i++)
	{
		Frame* pFirst = matchedImagesPairs[i].first;
		Frame* pSecond = matchedImagesPairs[i].second;
		if(pFirst)
			delete pFirst;
		if(pSecond)
			delete pSecond;
	}
	cout << "delete ok" << endl;
	
	cv::waitKey(0);

	return 0;
}


void LoadImages(const string &strImagePath, const string &strPathTimes,
				vector<string> &vstrImages, vector<double> &vTimeStamps)
{
	ifstream fTimes;
	fTimes.open(strPathTimes.c_str());
	vTimeStamps.reserve(5000);
	vstrImages.reserve(5000);
	while(!fTimes.eof())
	{
		string s;
		getline(fTimes,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
			double t;
			ss >> t;
			vTimeStamps.push_back(t/1e9);

		}
	}
}
