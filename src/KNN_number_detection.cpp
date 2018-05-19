#include "KNN_number_detection.h"


void numDetection::trainModel()
{
  	Mat data, labels; //data和labels分别存放  
    Mat tepImage; 


	// 读取0-9的样本图片
    for (int i = 0; i <= 9; i++)  
    {  
		string imgPath = "../src/NumberTrain/jpg/"+ to_string(i) +".jpg";		//训练集图片位置
        tepImage = imread(imgPath, IMREAD_GRAYSCALE );					//灰度图形式读取，后面才能转换 CV_32F格式，无参数读取，转换有问题
        data.push_back(tepImage.reshape(0, 1));         				//将图像转成一维数组插入到data矩阵中
      	labels.push_back(i);             								//将图像对应的标注插入到labels矩阵中
    }  
	data.convertTo(data, CV_32F);										//转换 CV_32F格式，此方法的库要求必须用CV_32F格式

    Mat trainData, trainLabel;  
    trainData = data(Range(0, 10), Range::all());  
    trainLabel = labels(Range(0, 10), Range::all());  

	//使用KNN算法  
    int k = 1;	//每个数字只有一个样本，所以k=1最合适，相当于就是与那个样本误差最小就选谁

    Ptr<TrainData>   tData = TrainData::create(trainData,ROW_SAMPLE, trainLabel); //ROW_SAMPLE表示一行一个样本  
    model->setDefaultK(k); model->setIsClassifier(true);  
    model->train(tData);       

}

void numDetection::detectNum(Mat &_img)
{
	Mat img;
	Mat testData;

    resize(_img, img, Size(500,400));
	testData = img.reshape(0, 1);						//将图像转成一维数组插入到data矩阵中   
    testData.convertTo(testData, CV_32F); 
   
	float r = model->predict(testData);  
	cout << "result: " << r << endl << endl;
               

}

