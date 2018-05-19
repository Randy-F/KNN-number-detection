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



    for (int i = 0; i <= 9; i++)  
    {  
        imgPath = "../src/NumberTrain/png/"+ to_string(i) +".png";
        tepImage = imread(imgPath, IMREAD_GRAYSCALE );  
	    cout << '(' << tepImage.rows << ',' << tepImage.cols << ')' << endl << endl;
        data.push_back(tepImage.reshape(0, 1));         //将图像转成一维数组插入到data矩阵中  
    	labels.push_back(i);             //将图像对应的标注插入到labels矩阵中 
     	}  


    data.convertTo(data, CV_32S);  

	//cout << '(' << img.rows << ',' << img.cols << ')' << endl << endl;

    testData.push_back(img.reshape(0, 1));         //将图像转成一维数组插入到data矩阵中  
    labels.push_back(0);   

  
    Mat trainData, trainLabel;  
    trainData = data(Range(0, 9), Range::all());  
    trainLabel = labels(Range(0, 9), Range::all());  

    //使用KNN算法  
    int k = 5;  



    Ptr<TrainData>   tData = TrainData::create(trainData,ROW_SAMPLE, trainLabel); //ROW_SAMPLE表示一行一个样本  
    Ptr<KNearest> model = KNearest::create();  
    model->setDefaultK(k); model->setIsClassifier(true);  
    model->train(tData);  
     
    double train_hr=0, test_hr=0;  
    Mat response;  
    for (int i = 0; i <= 9; i++)  
    {  
       Mat sample = data.row(i);  
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
    //    if (i < trainNum)  
    //        train_hr=train_hr+r;  
    //    else  
            test_hr=test_hr + r;  
      //  cout << "i= " <<  << ',' << img.cols << ')' << endl << endl;
    }  
    //cout << train_hr << "   " << test_hr << endl;  

    cout << "KNN模型在训练集上的准确率为" << train_hr / 9 * 100 << "%，在测试集上的准确率为" 
        // << test_hr / (data.rows-trainNum)*100<<"%"<<endl;  
         << test_hr / 1*100<<"%"<<endl;  */
    cv::waitKey ( 0 );              
    return 0;  
}  
