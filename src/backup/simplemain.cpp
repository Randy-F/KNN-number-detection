// knnrecognizenum.cpp:使用knn识别手写数字  
//  
//#include "stdafx.h"  

#include "KNN_number_detection.h"

  
int main()  
{   
   	string imgPath = "../src/NumberTrain/6.jpg";

	Mat img = imread(imgPath, IMREAD_GRAYSCALE ); 

    numDetection numDetector;

	numDetector.trainModel();
	numDetector.detectNum(img);

    cv::waitKey ( 0 );              
    return 0;  
}  
