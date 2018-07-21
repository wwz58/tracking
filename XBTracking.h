#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <vector>
#include"kcftracker.hpp"
#define XBD_
//#undef XBD_

using namespace cv;
using namespace std;

/** @function cascade detect */
struct detectPart{
	CascadeClassifier some_body_part_cascade;
	detectPart();
	void detect(Mat &frame, std::vector<Rect> &bodies);	
};

struct detectPeople{
	HOGDescriptor hog;
	detectPeople();
	//detectPeople(int);
	void detect(float DetectPeople_offset,Mat & img, std:: vector<Rect> & found_filtered);
	detectPart cascadedp;
	bool reDetect(const Mat &x,  Rect );
	float alpha;//0-0.45
	float beta;//0-0.45
	int use_cascade;
};


class XBTracking{
public:
	XBTracking();
	bool setScale(float x);
	bool setResponseThrd(float x);

	//detect. if return 0, no obj, else get obj..
	int go(Mat * frame,//input
		Mat & img,//output
		Rect & x //output
		);
	int go(Mat * frame,//input
		Mat & img,//output
		Rect & x, //output
		vector <Rect> & hogx //output
		);
    int detectGo( Rect &r);
	void setShowNewImg(bool);
	void change(Rect & b,Rect & x,float scale);
	int isCenter( vector <Rect> & bs,Mat &img);
	void change(Rect2d & b,Rect & x,float scale);
	int m_stdHeight;
	float DetectPeople_offset;//[0~0.5)
	int getit; //2,detected,but not go; 1, detected, go; 0, not detected; -1, go back.
	double scale;//0.4;
	int countk;
	float ResponseThrd; //0.12
	//Ptr<Tracker> tracker ;
	KCFTracker tracker_hog;
	//Ptr<TrackerKCF> tracker_hog;
	//Ptr<TrackerCSRT> tracker_hog;

	//std::vector<Rect> bodies;
	int first;
	int w_init;
	bool m_ShowNewImg;
	Rect pre_rect;
	//detectPeople dp;
	detectPart dp;

	MatND std_Hhist;
	MatND std_Ghist;
};




//OpenCV里貌似没有判断rect1是否在rect2里面的功能，所以自己写一个吧  
bool isInside(Rect &rect1, Rect &rect2);
Rect bigger(Rect &b, int maxwidth, int maxheight);
