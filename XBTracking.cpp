#include "XBTracking.h"
/*string upper_body_cascade_name[] ={ 
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
	"haarcascade_mcs_upperbody.xml"
};
*/
/** @function detectAndDisplay */
detectPart::detectPart()
{
	//char ss[200];
	//strcpy(ss,"conf/");
	//strcat(ss,upper_body_cascade_name[kk].c_str());
	if (!some_body_part_cascade.load("D:\\opencv320\\opencv_contrib-3.2.0\\modules\\face\\data\\cascades\\haarcascade_mcs_upperbody.xml"))
	{
		printf("--(!)Error loading\n"); 
		return;
	}
}

void detectPart::detect(Mat &frame, std::vector<Rect> &bodies)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect 
	//some_body_part_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	some_body_part_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0, Size(30, 30));
}

XBTracking::XBTracking():dp(){
	//tracker = Tracker::create("KCF");
	ResponseThrd=0.17;
	m_stdHeight = 145;
	DetectPeople_offset =0.2;

	getit=0;
	countk=0;
	first=1;
	tracker_hog = KCFTracker(true, false, true, true);
	//tracker_hog = TrackerKCF::create();
	//tracker_hog = TrackerCSRT::create();

	m_ShowNewImg = false;
}
void XBTracking::setShowNewImg(bool x)
{
	m_ShowNewImg=x;
}
bool XBTracking::setScale(float x){
	if (0>=x || x>=1 ) return -1;
	scale = x;
	return 0;
}

bool XBTracking::setResponseThrd(float x){
	if (0>=x || x>=0.5 ) return -1;
	ResponseThrd = x;
	return 0;

}
int XBTracking::detectGo( Rect &r)
{
	int ret;
	float scale=1;
	
	if (r.width > w_init * 3 ||r.width < w_init /5)
	{
		ret =  0;
	}
	else if (r.width > w_init * scale)
	{
		ret =  -1;
	}
	else if (r.width < w_init /scale)
	{
		ret = 1;
	}
	else{
		ret = 2;
	}
	/*if (1){
		int precenter_x = pre_rect.x+pre_rect.width/2;
		int precenter_y = pre_rect.y+pre_rect.height/2;
		int current_x = r.x + r.width/2;
		int current_y = r.y + r.height/2;
		if (abs(current_x-precenter_x)>w_init*2/3)
		{
			ret = 0;
		}

	}*/
	return ret;
}
double XBcompareHist(MatND &hist1, MatND &hist2,int n=1)
{
	//int n=1;
	//直方图hist1和hist2的比较
	//n=1用的是Correlation，c1越大表示相似度越高
	//n=2用的是Chi-square，c1越小表示相似度越高
	//n=3用的是Intersection，c1越大表示相似度越高
	//n=4用的是Bhattacharyya距离，c1越小表示相似度越高
	//cout << hist1.type()<<":"<<hist2.type()<<" ";
	//cout << hist1.depth()<<":"<<hist2.depth()<<" ";

	double c1 = compareHist(hist1, hist2, n);
	if (n==1 || n==3) c1=-c1;
	return c1;
}
void getHist(Mat & image, Rect &b, MatND & Hhist, MatND & Ghist)
{
	Rect whole_img(0, 0, image.cols, image.rows);
	b = b & whole_img;
	Mat img(image,b);
	int hsize = 16; 
	int MaxHeight=1;
	{//转Hue空间

		Mat hsv,  hue;
		cvtColor(img, hsv, CV_BGR2HSV);//将rgb摄像头帧转化成hsv空间的
		hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度  
		int ch[] = {0, 0};  
		mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组  
		//calcHist()函数:
		//第1个参数为输入矩阵序列，
		//第2个参数表示输入的矩阵数目，
		//第3个参数表示将被计算直方图维数通道的列表，
		//第4个参数表示可选的掩码函数  
		//第5个参数表示输出直方图，第6个参数表示直方图的维数，
		//第7个参数为每一维直方图数组的大小，第8个参数为每一维直方图bin的边界  
		float hranges[] = {0,180};	//hranges在后面的计算直方图函数中要用到  
		const float* phranges = hranges;
		calcHist(&hue, 1, 0, cv::Mat(), Hhist, 1, &hsize, &phranges);//将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小  
		normalize(Hhist, Hhist, 0, MaxHeight, CV_MINMAX);//将hist矩阵进行数组范围归一化，都归一化到0~255  
	}
#ifdef XBD_
{ 
		//show hist img
		Mat histimg = Mat::zeros(200, 320, CV_8UC3);
		int binW = histimg.cols / hsize;  //histing是一个200*300的矩阵，hsize应该是每一个bin的宽度，也就是histing矩阵能分出几个bin出来  
		Mat buf(1, hsize, CV_8UC3);//定义一个缓冲单bin矩阵  
		for( int i = 0; i < hsize; i++ )//saturate_case函数为从一个初始类型准确变换到另一个初始类型  
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);//Vec3b为3个char值的向量  
		cvtColor(buf, buf, CV_HSV2BGR);//将hsv又转换成bgr  

		for( int i = 0; i < hsize; i++ )  
		{  
			int val = saturate_cast<int>(Hhist.at<float>(i)*histimg.rows/MaxHeight);//at函数为返回一个指定数组元素的参考值  
			rectangle( histimg, Point(i*binW,histimg.rows),    //在一幅输入图像上画一个简单抽的矩形，指定左上角和右下角，并定义颜色，大小，线型等  
				Point((i+1)*binW,histimg.rows - val),  
				Scalar(buf.at<Vec3b>(i)), -1, 8 );  
		}  
		imshow( "hist", histimg ); 

	}
#endif

	{
		//转grey空间
		//int hsize = 16;  
		float hranges[] = {0,255};	//hranges在后面的计算直方图函数中要用到  
		const float* phranges = hranges;
		Mat hue;
		cvtColor(img, hue, CV_BGR2GRAY);
		calcHist(&hue, 1, 0, cv::Mat(), Ghist, 1, &hsize, &phranges);//将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小  
		normalize(Ghist, Ghist, 0, MaxHeight, CV_MINMAX);//将hist矩阵进行数组范围归一化，都归一化到0~255  
	}

#ifdef XBD_
 { 		//show hist img
		Mat histimg = Mat::zeros(200, 256, CV_8UC3);
		int binW = histimg.cols / hsize;  //histing是一个200*300的矩阵，hsize应该是每一个bin的宽度，也就是histing矩阵能分出几个bin出来  
		Mat buf(1, hsize, CV_8UC3);//定义一个缓冲单bin矩阵  
		for( int i = 0; i < hsize; i++ )//saturate_case函数为从一个初始类型准确变换到另一个初始类型  
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);//Vec3b为3个char值的向量  
		cvtColor(buf, buf, CV_HSV2BGR);//将hsv又转换成bgr  

		for( int i = 0; i < hsize; i++ )  
		{  
			int val = saturate_cast<int>(Ghist.at<float>(i)*histimg.rows/MaxHeight);//at函数为返回一个指定数组元素的参考值  
			rectangle( histimg, Point(i*binW,histimg.rows),    //在一幅输入图像上画一个简单抽的矩形，指定左上角和右下角，并定义颜色，大小，线型等  
				Point((i+1)*binW,histimg.rows - val),  
				Scalar(buf.at<Vec3b>(i)), -1, 8 );  
		}  
		imshow( "gray", histimg ); 

	}
#endif

}
int cal_dist(int x1, int y1, int x2, int y2){
	return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
}
int XBTracking::isCenter( vector <Rect> & bs,Mat &img)
{
	int h=img.rows, w=img.cols;
	vector<int> candidates;
	for (int i=0;i<bs.size();i++){
		Rect &b = bs[i];
		if(b.width>10&&b.height>10&&b.y<h/2)
			candidates.push_back(i);
	}

#ifdef XBD_
	cout << "CAND:"<< candidates.size()<<" ";
#endif

	if (candidates.size()==0) return 0;
	if (std_Hhist.type()!=0){

		MatND Hhist, Ghist;
		Rect &r = bs[candidates[0]];
		//Point c = r.tl() + r.br();
		Rect tmp((0.25 * r.x), (r.br().y), 0.5 * r.width, 30);

		getHist(img, tmp, Hhist, Ghist);
		double min_dist1 = XBcompareHist(std_Hhist, Hhist);
		double min_dist1Gray = XBcompareHist(std_Ghist, Ghist);
#ifdef XBD_
		cout << "[" << fabs(min_dist1) << fabs(min_dist1Gray) << "] ";
#endif
		if (candidates.size() == 1) {
			if (fabs(min_dist1) > 30 && fabs(min_dist1Gray) > 30)
				return 0;
			else
				return 1;
		}
		else if (candidates.size() == 2) {
			MatND Hhist2, Ghist2;
			Rect &rr = bs[candidates[1]];
			//c = rr.tl() + rr.br();
			tmp = Rect((0.25 * rr.x), (rr.br().y), 0.5 * rr.width, 30);

			getHist(img, tmp, Hhist2, Ghist2);
			double min_dist2 = XBcompareHist(std_Hhist, Hhist2);
			double min_dist2Gray = XBcompareHist(std_Ghist, Ghist2);

			if (( min_dist1 >= min_dist2 && min_dist1Gray >= min_dist2Gray) && (fabs(min_dist2) <= 30 || fabs(min_dist2Gray) <= 30))
				return 2;
			else if (fabs(min_dist1) <= 30)
				return 1;
			else
				return 0;
		}
		else
			return 0;
	}
	else{
		int min_dist=cal_dist(
			bs[candidates[0]].x+bs[candidates[0]].width/2, 
			bs[candidates[0]].y+bs[candidates[0]].height/2,
			w/2, 
			h/2);
		int min_i = 0;
		for (int i=1;i<candidates.size();i++){
			int dist = cal_dist(
				bs[candidates[i]].x+bs[candidates[i]].width/2, 
				bs[candidates[i]].y+bs[candidates[i]].height/2,
				w/2, 
				h/2);
			if (min_dist>dist){
				min_dist=dist, min_i = i;
			}
		}
		return min_i+1;
	}
}

int XBTracking::go(Mat * srcImage,//input
	Mat & img,//output
	Rect & x //output
	)
{
	vector <Rect> hogx;		 //output
	return go(srcImage,img, x, hogx);
}



int XBTracking::go(Mat * srcImage,//input
	Mat & img,//output
	Rect & x,//output
	vector <Rect> & bodies		 //output
	)
{
	x.height=x.width=x.x=x.y=0;
	Mat frame;
	cvtColor( * srcImage, frame,COLOR_RGBA2RGB);
	Rect b;
	scale = 1.0 * m_stdHeight / frame.rows;
	Size ResImgSiz = Size(frame.cols*(scale), frame.rows*(scale));
	img = Mat(ResImgSiz, frame.type());
	//Mat img1 = Mat(ResImgSiz, frame1.type());

	resize(frame, img, ResImgSiz, CV_INTER_CUBIC);
	
	dp.detect( img, bodies);
	//dp.detect(DetectPeople_offset,img,bodies);
#ifdef XBD_
	cout <<"bodies:"<<bodies.size()<<" ";
#endif
	int k=0;
	//本次HOG检测到（hoggetit==1 ），则进行KCF初始化，不进行KCF跟踪，直接返回非0（-1，1，2）
	//本次HOG未检测到（hoggetit==0 ），上次都未检测到(getit==0)，则不进行KCF，直接返回0
	//本次HOG未检测到（hoggetit==0 ），上次已经检测到(getit!=0)，则进行KCF，可能返回（-1 0 1 2）
	int hoggetit = 0;
	if (  bodies.size()>=1 && (k=isCenter(bodies,img))) // && faces.size()==1 )
	{
		b=bodies[k-1];
		//tracker->init(img,b);
		if (first) {
			hoggetit = 2;
			first=0;
			w_init = b.width;
			pre_rect = b;
			//获取图像b区域的直方图
			Mat hist;
			getHist(img,b,std_Hhist,std_Ghist);
		}
		else{
			hoggetit = detectGo(b);
#ifdef XBD_
	cout <<"detectGo:"<<hoggetit<< " ";
#endif
		}
		if (hoggetit /*&& dp.reDetect(img,b)*/)
		{
			getit=hoggetit;
			tracker_hog.init(b, img);
			//tracker_hog->init( img,b);
			//Rect bb = b;

		}
		else{
			hoggetit = 0;			
		}
		//change(b,x,scale);
		//return ;
	}
#ifdef XBD_
	cout <<(k?"":"noBody ");
#endif
	if ( !hoggetit && getit)  //以前检测到，本次HOG没检测到，则使用kcf
	{

		float peak = tracker_hog.update(img,b);
		//Rect2d tmp = Rect2d(b);
		//float peak = tracker_hog->update(img,tmp);
		//b = Rect(tmp);

		//getit = detectGo(b);
#ifdef XBD_
		//cout << "CR:"<<peak << " ";
		cout <<"detectGo:"<<!!getit<<" ";
#endif
		if (peak<ResponseThrd)
		//if (!isfound)
			getit=0;
#ifdef XBD_
		cout <<"Thrd:"<<!!getit<<" ";
#endif
	}

	if (getit)
		change(b,x,scale);
	for (int i=0;i<bodies.size();i++){
		Rect &b = bodies[i];
		change(b,b,scale);
	}

	char s[][5]={"--- " ,//!!hoggetit=0, !!getit=0
		"kcf ", //!!hoggetit=0, !!getit=1
		"hog "  //!!hoggetit=1, !!getit=1
	};
#ifdef XBD_
	cout << s[!!hoggetit+!!getit] <<" ("<<x.x<<","<<x.y<<" "<<x.width<<","<<x.height<<")"<<endl;
#endif
	/*if (!getit) first=1;
	else pre_rect = b;*/
	if (getit)
		pre_rect = b;

	return getit;
}
void XBTracking::change(Rect & b,Rect & x,float scale)
{
	//return;
	float scale1 = m_ShowNewImg?1:scale;
	x.x = b.x / scale1;
	x.y = b.y / scale1;
	x.width = b.width / scale1;
	x.height = b.height / scale1;
}
void XBTracking::change(Rect2d & b,Rect & x,float scale)
{
	//return;
	x.x = b.x / scale;
	x.y = b.y / scale;
	x.width = b.width / scale;
	x.height = b.height / scale;

}

/*detectPeople::detectPeople(int x):hog(cv::Size(48,96),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8),9,1,-1,cv::HOGDescriptor::L2Hys,0.2,true,cv::HOGDescriptor::DEFAULT_NLEVELS)
{
	use_cascade = 0;
	alpha=0.2;//0-0.45
	beta=0.1;//0-0.45
	static vector <float> x1 = HOGDescriptor::getDaimlerPeopleDetector();
	hog.setSVMDetector(x1);//得到检测器
}*/

detectPeople::detectPeople()
{
	use_cascade = 0;
	alpha=0.2;//0-0.45
	beta=0.2;//0-0.45
	static vector <float> x = HOGDescriptor::getDefaultPeopleDetector();
	hog.setSVMDetector(x);//得到检测器
}


void detectPeople::detect(float offset, Mat & img2, vector<Rect> & found_filtered)
{
	int h = img2.rows;
	int w = img2.cols;

	int OFFx=offset*w;
	Rect r(OFFx,0,w*(1-2*offset),h);
	Mat img1 (img2,r);
	found_filtered.clear();
	vector<Rect> found;
	int i,j;
	hog.detectMultiScale(img1, found, 0, Size(8,8), Size(32,32), 1.05, 2);
	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == found.size() )
			found_filtered.push_back(r);
	}
	//printf("tdetection time = %gms, found_filtered size:%d\n", t*1000./cv::getTickFrequency(),found_filtered.size());
	for( i = 0; i < found_filtered.size(); i++ )
	{
		Rect &r = found_filtered[i];

		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.

		r.x += cvRound(r.width*alpha)+OFFx;
		r.width = cvRound(r.width*(1-2*alpha));
		r.y += cvRound(r.height*beta);
		r.height = cvRound(r.height*(1-2*beta));
		//rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 5);
	}

	//imshow(window_name, img);
}

bool detectPeople::reDetect(const Mat & img1,Rect r)
{
	if (!use_cascade) return true;
	r.width = cvRound(r.width/(1-2*alpha));
	r.height = cvRound(r.height/(1-2*beta));
	r.x -= cvRound(r.width*alpha);
	r.y -= cvRound(r.height*beta);
	if (r.width+r.x>img1.cols) r.width = img1.cols - r.x;
	if (r.height+r.y>img1.rows) r.height = img1.rows - r.y;

	Mat ni(img1,r);
	vector<Rect> x;
	cascadedp.detect(ni,x);
	return x.size();
}


bool isInside(Rect &rect1, Rect &rect2)
{
	return rect2.y<rect1.y+rect1.height/2
		&& rect2.x< rect1.x
		&& rect1.width< rect2.width
		&& rect1.height< rect2.height/3;
	//return (rect1 == (rect1&rect2));
}


Rect bigger(Rect &b, int maxwidth, int maxheight)
{
	Rect x=b;
	float alpha=2;
	x.width = b.width * alpha;
	x.height = b.height * alpha;
	x.x = b.x - b.width*(alpha/2-0.5);
	x.y = b.y - b.height*(alpha/2-0.5);
	if (x.x<0) x.x=0;
	if (x.y<0) x.y=0;
	if (x.x+x.width>=maxwidth) x.width = maxwidth-x.x;
	if (x.y+x.height>=maxheight) x.height = maxheight-x.y;

	return x;
}