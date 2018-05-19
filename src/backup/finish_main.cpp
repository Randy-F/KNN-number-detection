// knnrecognizenum.cpp:使用knn识别手写数字  
//  
//#include "stdafx.h"  
#include<iostream>  
#include<opencv2/ml/ml.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <string>

using namespace std;  
using namespace cv;  
using namespace cv::ml;  
  
int main()  
{  

    Mat data, labels, testData; //data和labels分别存放  
    Mat tepImage,img; 

    string imgPath;

	// 读取0-9的样本图片
    for (int i = 0; i <= 9; i++)  
    {  
		imgPath = "../src/NumberTrain/jpg/"+ to_string(i) +".jpg";		//训练集图片位置
        tepImage = imread(imgPath, IMREAD_GRAYSCALE );					//灰度图形式读取，后面才能转换 CV_32F格式，无参数读取，转换有问题
        data.push_back(tepImage.reshape(0, 1));         				//将图像转成一维数组插入到data矩阵中
      	labels.push_back(i);             								//将图像对应的标注插入到labels矩阵中
    }  
	data.convertTo(data, CV_32F);										//转换 CV_32F格式，此方法的库要求必须用CV_32F格式

	// 测试集
    for (int i = 0; i <= 9; i++)  
    {  
		imgPath = "../src/NumberTrain/"+ to_string(i) +".jpg";			// 测试集图片位置
		//imgPath = "../src/samples/sample"+ to_string(i) +".bmp";			// 测试集图片位置
		tepImage = imread(imgPath, IMREAD_GRAYSCALE ); 
        resize(tepImage, img, Size(500,400));
        // imshow("22", img);
		//imshow("22", tepImage);
    	cv::waitKey ( 0 ); 
		testData.push_back(img.reshape(0, 1));						//将图像转成一维数组插入到data矩阵中 

    }  

	//  labels.push_back(0);   
    testData.convertTo(testData, CV_32F); 


	// 建立训练集 
    Mat trainData, trainLabel;  
    trainData = data(Range(0, 10), Range::all());  
    trainLabel = labels(Range(0, 10), Range::all());  
   

	//使用KNN算法  
    int k = 1;	//每个数字只有一个样本，所以k=1最合适，相当于就是与那个样本误差最小就选谁

    Ptr<TrainData>   tData = TrainData::create(trainData,ROW_SAMPLE, trainLabel); //ROW_SAMPLE表示一行一个样本  
    Ptr<KNearest> model = KNearest::create();  
    model->setDefaultK(k); model->setIsClassifier(true);  
    model->train(tData);  
     
    double train_hr=0, test_hr=0;  
    Mat response;  

	//   训练集上效果的验证
    for (int i = 0; i <= 9; i++)  
    {  
       Mat sample = data.row(i);  
       float r = model->predict(sample);  
       cout << "i= " << i << "result: " << r << endl << endl;

    }

     
    for (int i = 0; i <= 9; i++)  
    {  
		Mat sample = testData.row(i);  
		float r = model->predict(sample);  
		cout << "i= " << i << "result: " << r << endl << endl;

    }

	/*  for (int i = 0; i < 1; i++)  
    {  
        Mat sample = testData.row(i);  

		cout << ( sample.type() == CV_32F ) << endl << endl; 

        float r = model->predict(sample);  
        r = abs(r - labels.at<int>(i));  
        if (r <= FLT_EPSILON)// FLT_EPSILON表示最小的float浮点数，小于它，就是等于0  
            r = 1.f;  
        else  
            r = 0.f;  
        if (i < trainNum)  
            train_hr=train_hr+r;  
       else  
            test_hr=test_hr + r;  
    }  
    //cout << train_hr << "   " << test_hr << endl;  

    cout << "KNN模型在训练集上的准确率为" << train_hr / 9 * 100 << "%，在测试集上的准确率为" 
        // << test_hr / (data.rows-trainNum)*100<<"%"<<endl;  
         << test_hr / 1*100<<"%"<<endl;  */

    cv::waitKey ( 0 );              
    return 0;  
}  
