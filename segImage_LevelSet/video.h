#ifndef OPENCV
#define OPENCV
#include <opencv2\opencv.hpp> 
using namespace cv;
#endif // !OPENCV

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

class Video
{
public:
	Mat frame[2];
	Mat gray[2];
	int frame_length;

	Video(const char *);
};