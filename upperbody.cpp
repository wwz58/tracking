#include "XBfun.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/tracking.hpp"  
#include "detectPeople.h"
#include "XBTracking.h"

using namespace std;
using namespace cv;


int vmin1 = 10, vmax1 = 256, smin1 = 30;  

/** Global variables */

string upper_body_cascade_name[] ={ 
	"haarcascade_frontalface_alt2.xml", //侧脸，歪脑袋都不行
	"haarcascade_frontalface_alt_tree.xml" , //检测不出来
	"haarcascade_frontalface_alt.xml",
	"haarcascade_frontalface_default.xml",
	"haarcascade_fullbody.xml",
	"haarcascade_upperbody.xml",
	"haarcascade_eye.xml",
	"haarcascade_eye_tree_eyeglasses.xml",
	"haarcascade_lefteye_2splits.xml",
	"haarcascade_licence_plate_rus_16stages.xml",
	"haarcascade_lowerbody.xml",
	"haarcascade_profileface.xml",
	"haarcascade_righteye_2splits.xml",
	"haarcascade_russian_plate_number.xml",
	"haarcascade_smile.xml",

	"haarcascade_frontalcatface.xml",
	"haarcascade_frontalcatface_extended.xml",
	""
};


string window_name = "Capture - Body detection";
RNG rng(12345);

/** @function detectAndDisplay */
struct detectPart{
	CascadeClassifier some_body_part_cascade;
	detectPart(int kk)
	{
		char ss[200];
		strcpy(ss,"conf/");
		strcat(ss,upper_body_cascade_name[kk].c_str());
		if (!some_body_part_cascade.load(ss))
		{
			printf("--(!)Error loading\n"); 
			return;
		}
	}

	void detect(Mat &frame, std::vector<Rect> &bodies)
	{
		;
		Mat frame_gray;

		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		//-- Detect faces
		some_body_part_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t i = 0; i < bodies.size(); i++)
		{
			rectangle(frame, bodies[i], Scalar(255, 0, 255));
		}
		//-- Show what you got
		imshow(window_name, frame);
	}
};



int trackObject1 = 0; //代表跟踪目标数目  
/** @function main */
int main2(int argc, const char** argv)
{


	VideoCapture capture(0);
	//VideoCapture capture1(1);
	char * aargv="F:\\xbProject\\25.Eric\\视频\\good-boy\\good-boy.mp4";
	std::string video = aargv;//argv[1];
	//VideoCapture capture(video);
	//VideoCapture capture("F:/xbProject/25.Eric/视频/寝室三人.mp4");

	
	printf("type c to exit.\n");

	namedWindow( window_name, 0 ); 

	//-- 2. Read the video stream

	if (!capture.isOpened()) return -1;

	XBTracking xbt;
	Mat frame;

	while (true)
	{
		capture >> frame;
		if ( frame.cols<10) break;
		Rect roi;
		Mat img;
		xbt.go(&frame,img, roi);
		rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
		imshow(window_name, frame );  
		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}
	return 0;
}

