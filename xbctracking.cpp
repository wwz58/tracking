// tracking.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include "opencv2/video/tracking.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "XBTracking.h"  
#include <time.h>

#include <iostream>  
#include <ctype.h>  

using namespace cv;  
using namespace std;  



int main( int argc, const char** argv )  
{  
	// help();  

	VideoCapture cap; //定义一个摄像头捕捉的类对象  labOriginal
	//cap.open("D:\\graduate\\video\\allMy\\good-boy.mp4");
	cap.open("D:\\graduate\\video\\IMG_9832.mp4");

	//cap.open("F:\\xbProject\\25.Eric\\视频\\VID_20180208_165656.mp4");
	if( !cap.isOpened() )  
	{  
		//help();  
		cout << "***Could not initialize capturing...***\n";  
		cout << "Current parameter's value: \n";  
		//parser.printMessage();  
		return -1;  
	}  

	namedWindow( "scaled", 0 );  
	//namedWindow( "CamShift Demo", 0 ); 
	//namedWindow( "project", 0 ); 


	Mat img,frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;  
	bool paused = false;  
	XBTracking xbt; 
	xbt.setShowNewImg(true);
	//xbt.setScale(0.5);
	//xbt.setResponseThrd(0.2);
	int tag=-1,pretag=0;
	int N=2;
	time_t t,t1;
	struct tm  *lt1;
	time (&t);//获取Unix时间戳。

	for(;;)  
	{  
		if( !paused )//没有暂停  
		{  
			cap >> frame;//从摄像头抓取一帧图像并输出到frame中  
			if( frame.empty() )  
				break;  
			tag++;
			if (tag%N!=0)
				continue;

			cout << tag <<":";
			frame.copyTo(img);  
			Mat image;
			if( !paused )//没有按暂停键  
			{
				RotatedRect trackBox;
				vector<Rect>  hogx;
				Rect kcfx;

				int ret = xbt.go(&img,image,kcfx, hogx);
				//cout << ret << endl;
				if (ret)
					rectangle(image, kcfx.tl(), kcfx.br(), cv::Scalar(255,0,0), 2);

				//ellipse( image, trackBox, Scalar(0,0,255), 1, CV_AA );//跟踪的时候以椭圆为代表目标  
				if (1)for(int i = 0; i < hogx.size(); i++ )
				{
					Rect &r = hogx[i];
					// the HOG detector returns slightly larger rectangles than the real objects.
					rectangle(image, r.tl(), r.br(), cv::Scalar(0,0,255), 2);
					putText(image, "hoggetit", r.br(), 1, 2.0, Scalar(0, 0, 255),2);
				}

			} 
			imshow( "scaled", image );  

			//imshow( "CamShift Demo", image );  
			//imshow( "Histogram", histimg );  


			time(&t1);
			if (t1>t){
				t = t1;
				lt1 = localtime (&t);//转为时间结构。
				printf ("\n-%02d-----------------%d/%d/%d %d:%d:%d--------------\n",(tag-pretag)/N,lt1->tm_year+1900, lt1->tm_mon, lt1->tm_mday, lt1->tm_hour, lt1->tm_min, lt1->tm_sec);//输出结果
				pretag=tag;
			}
		}
		//cout << endl;
		char c = (char)waitKey(10);  
		if( c == 27 )              //退出键  
			break;
		else if (c==32)
			paused=!paused;
	}  
	return 0;  
}