/**
* This file is for transforming euroc ground truth into TUM Data Format.
* Editted by Xiaochen Qiu (Beihang University)
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <iomanip>

#include<opencv2/core/core.hpp>

using namespace std;

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		cerr << endl << "Usage: ./EuRoCPoses_2_TUMFormat path_to_data_csv path_to_result" << endl;
		return 1;
	}
	
	// group truth文件
	ifstream if_poses(argv[1]);
	if(!if_poses)
	{
		cerr << endl << "Can't find data.csv in this location: " << argv[1] << endl;
		return 1;
	}
	// 转换结果文件
	ofstream of_result(argv[2]);
	of_result << fixed;
	
	// 读取数据
	string poseLine;
	getline(if_poses,poseLine,'\n');	// 舍弃第一行
	while(!if_poses.eof())
	{
		// 转换本行表达的位姿
		getline(if_poses,poseLine,'\n');
		stringstream posetream(poseLine);
		string strpose;
		vector<string> vposeline;
		while(getline(posetream,strpose,','))		// 分解每行中的元素（以逗号为分隔符）
			vposeline.push_back(strpose);
		
		if(17==vposeline.size())	// 当元素个数为17时才进行转换
		{
			// 时间（ns转s）
			long int ltime = atol(vposeline[0].c_str());
			double dtime = ltime*1e-9;
			// 存储第i个相机原点位置在参考坐标系下的坐标
			cv::Mat t = ( cv::Mat_<float>(3,1) << atof(vposeline[1].c_str()), atof(vposeline[2].c_str()), atof(vposeline[3].c_str()) );
			// 存储第i个相机系到参考坐标系的旋转矩阵（四元数形式）
			cv::Mat q = ( cv::Mat_<float>(4,1) << atof(vposeline[5].c_str()), atof(vposeline[6].c_str()), atof(vposeline[7].c_str()), atof(vposeline[4].c_str()) );
			
			// 写入文件
			of_result << setprecision(6) << dtime << " " 
						<< setprecision(7) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " 
						<< q.at<float>(0) << " " << q.at<float>(1) << " " << q.at<float>(2) << " " << q.at<float>(3) << endl;
		}
	}
	
	// 关闭文件
	if_poses.close();
	of_result.close();
	cout << "Transformation completed!" << endl;
	
	return 0;
}