// knnrecognizenum.cpp:使用knn识别手写数字  
//  

#include<iostream>  
#include<opencv2/ml/ml.hpp>  
#include<opencv2/highgui/highgui.hpp>  
using namespace std;  
using namespace cv;  
using namespace cv::ml;  
  
int main()  
{  
    Mat img = imread("digits.png", 0);  
    int boot = 20;  
    int m = img.rows / boot;   int n = img.cols / boot;  
    Mat data, labels; //data和labels分别存放  
  
    //截取数据的时候要按列截取  
    for (int i = 0; i < n; i++)  
    {  
        int  colNum = i * boot;  
        for (int j = 0; j < m; j++)  
        {  
            int rowNum = j * boot;  
            Mat tmp;  
            img(Range(rowNum, rowNum + boot), Range(colNum, colNum + boot)).copyTo(tmp);  
            data.push_back(tmp.reshape(0, 1));         //将图像转成一维数组插入到data矩阵中  
            labels.push_back((int)j / 5);             //将图像对应的标注插入到labels矩阵中  
        }  
    }  
    data.convertTo(data, CV_32F);  
	cout << data.type() << endl;
    int sampleNum = data.rows;  
    int trainNum = 3000;  
  
    Mat trainData, trainLabel;  
    trainData = data(Range(0, trainNum), Range::all());  
    trainLabel = labels(Range(0, trainNum), Range::all());  
  
    //使用KNN算法  
    int k = 5;  
    Ptr<TrainData>   tData = TrainData::create(trainData,ROW_SAMPLE, trainLabel); //ROW_SAMPLE表示一行一个样本  
    Ptr<KNearest> model = KNearest::create();  
    model->setDefaultK(k); model->setIsClassifier(true);  
    model->train(tData);  
  
    //预测分类  
    /*  Mat sample = data.row(500); 
    float res = model->predict(sample); 
    cout << "预测结果是："<< res << endl;*/ //预测一个的代码  
      
    double train_hr=0, test_hr=0;  
    Mat response;  
    for (int i = 0; i < sampleNum; i++)  
    {  
        Mat sample = data.row(i);  
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
    cout << "KNN模型在训练集上的准确率为" << train_hr / trainNum * 100 << "%，在测试集上的准确率为" << test_hr / (data.rows-trainNum)*100<<"%"<<endl;  
    system("pause");  
    return 0;  
}  
