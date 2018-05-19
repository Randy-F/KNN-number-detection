#if !defined(_KNN_NUMBER_DETECTION_ )
#define _KNN_NUMBER_DETECTION_ 
 
//#include "stdafx.h"  
#include <iostream>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <string>

using namespace std;  
using namespace cv;  
using namespace cv::ml;  
  
class numDetection {
    public:
    Ptr<KNearest> model = KNearest::create();  
 
	void trainModel();
	void detectNum(Mat &_img);
};

#endif



