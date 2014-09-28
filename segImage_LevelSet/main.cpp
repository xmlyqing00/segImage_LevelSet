#ifndef VIDEO
#define VIDEO
#include "video.h"
#endif // !VIDEO

#ifndef OPENCV
#define OPENCV
#include <opencv2\opencv.hpp> 
using namespace cv;
#endif // !OPENCV

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace cv;

#define sqr(_x) ((_x) * (_x))

const double pi = 3.141592653586369;
const double small_number = 1e-10;
const double eps = 1e-8;
const int route[8][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 }, { 1, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 } };

struct mouseArg
{
	short pushdown_flag;
	Point selected;
	mouseArg()
	{
		pushdown_flag = 0;
		selected.x = 0;
		selected.y = 0;
	};
};

void print_double_mat(Mat mat)
{
	for (int y = 0; y < mat.rows; y++)
	{
		printf("[");
		for (int x = 0; x < mat.cols; x++)
		{
			printf("%5.1lf", mat.ptr<double>(y)[x]);
			if (x != mat.cols - 1)
				printf("");
			else
				printf("]\n");
		}
	}
	printf("\n\n");
}
bool isOutside(int x, int y, int rows, int cols)
{
	if (x < 0 || y < 0 || x >= cols || y >= rows) return true;
	return false;
}
void NeumannBoundCond(Mat &f)
{
	int rows = f.rows;
	int cols = f.cols;
	
	f.row(2).copyTo(f.row(0));
	f.row(rows - 3).copyTo(f.row(rows - 1));
	f.col(2).copyTo(f.col(0));
	f.col(cols - 3).copyTo(f.col(cols - 1));

	f.ptr<double>(0)[0] = f.ptr<double>(2)[2];
	f.ptr<double>(0)[cols-1] = f.ptr<double>(2)[cols-3];
	f.ptr<double>(rows-1)[0] = f.ptr<double>(rows-3)[2];
	f.ptr<double>(rows-1)[cols-1] = f.ptr<double>(rows-3)[cols-3];
}
void div(const Mat &nx, const Mat &ny, Mat &output)
{
	Mat nxx, nyy;
	Sobel(nx, nxx, CV_64FC1, 1, 0, 1, 0.5);
	Sobel(ny, nyy, CV_64FC1, 0, 1, 1, 0.5);
	add(nxx, nyy, output);
}
void distReg_p2(const Mat &phi, const Mat &phi_x, const Mat &phi_y, const Mat &s, Mat &output)
{
	int rows = phi.rows;
	int cols = phi.cols;
	double ss, ps;
	Mat dps(rows, cols, CV_64FC1);
	
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			ss = s.ptr<double>(y)[x];
			ps = ((ss >= 0) && (ss <= 1)) * sin(2 * pi * ss) / (2 * pi) + (ss > 1) * (ss - 1);
			dps.ptr<double>(y)[x] = ((fabs(ps) < eps) ? 1 : ps) / ((fabs(ss) < eps) ? 1 : ss);
		}
	}

	//print_double_mat(dps);

	Mat temp_mat1, temp_mat2;
	temp_mat1 = dps.mul(phi_x) - phi_x;
	temp_mat2 = dps.mul(phi_y) - phi_y;
	div(temp_mat1, temp_mat2, output);
	Laplacian(phi, temp_mat2, CV_64FC1, 1);
	output = output + temp_mat2;
}
void Dirac(const Mat &phi, const double epsilon, Mat &output)
{
	for (int y = 0; y < phi.rows; y++)
	{
		for (int x = 0; x < phi.cols; x++)
		{
			double phii = phi.ptr<double>(y)[x];
			if (fabs(phii) <= epsilon)
			{
				output.ptr<double>(y)[x] = 0.5 / epsilon * (1 + cos(pi * phii / epsilon));
			}
			else output.ptr<double>(y)[x] = 0;
		} 
	}
}
void drlse_edge(Mat &phi, const Mat &g, const Mat &vx, const Mat &vy, 
				const double lambda, const double mu, const double alpha, const double epsilon, const double timestep, const int iter)
{
	int rows = phi.rows;
	int cols = phi.cols;

	for (int k = 1; k <= iter; k++)
	{
		NeumannBoundCond(phi);

		Mat phi_x, phi_y;
		Sobel(phi, phi_x, CV_64FC1, 1, 0, 1, 0.5);
		Sobel(phi, phi_y, CV_64FC1, 0, 1, 1, 0.5);

		Mat s;
		sqrt(phi_x.mul(phi_x) + phi_y.mul(phi_y), s);

		Mat temp_mat = Mat::ones(rows, cols, CV_64FC1);
		add(s, temp_mat.mul(small_number), s, noArray());
		Mat Nx, Ny;
		divide(phi_x, s, Nx);
		divide(phi_y, s, Ny);
		add(s, temp_mat.mul(-small_number), s, noArray());
		//print_double_mat(Nx);
		//print_double_mat(Ny);

		Mat curvature;
		div(Nx, Ny, curvature);

		Mat dist_reg_term;
		distReg_p2(phi, phi_x, phi_y, s, dist_reg_term);
		//print_double_mat(dist_reg_term);

		Mat dirac_phi(rows, cols, CV_64FC1);
		Dirac(phi, epsilon, dirac_phi);
		//print_double_mat(dirac_phi);

		Mat area_term;
		area_term = dirac_phi.mul(g);

		Mat edge_term;
		edge_term = dirac_phi.mul(vx.mul(Nx) + vy.mul(Ny)) + area_term.mul(curvature);

		phi = phi + ((Mat)(dist_reg_term.mul(mu) + edge_term.mul(lambda) + area_term.mul(alpha))).mul(timestep);
		//print_double_mat(phi);
	}
}
void debugLevelSet(const Mat &img, const Mat &phi)
{
	int rows = img.rows;
	int cols = img.cols;
	Vec3b intensity;
	intensity.val[0] = 0;
	intensity.val[1] = 0;
	intensity.val[2] = 255;
	Mat output = img.clone();
	Mat mask = Mat::zeros(rows, cols, CV_32SC1);
	Point next;

	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{ 
			if (phi.ptr<double>(y)[x] > 0) continue;

			for (int k = 0; k < 4; k++)
			{
				next.x = x + route[k][0];
				next.y = y + route[k][1];

				if (isOutside(next.x, next.y, rows, cols)) continue;
				if (phi.ptr<double>(next.y)[next.x] > 0)
				{
					output.ptr<Vec3b>(y)[x] = intensity;
					mask.ptr<int>(y)[x] = 1;
					break;
				}
			}
		}
	}

	//cout << mask << endl;
	print_double_mat(phi);
	namedWindow("test", WINDOW_AUTOSIZE);
	imshow("test", output);
	cv::waitKey(0);
}
void LevelSet(const Mat &img, const Mat &gray, Mat &phi, int direct = 1)
{
	int iter_inner, iter_outer;
	double  timestep, mu, lambda, alfa, epsilon;
	timestep = 5;
	mu = 0.2 / timestep;
	iter_inner = 5;
	iter_outer = 40;
	lambda = 5;
	alfa = 1.5 * direct;
	epsilon =  1.5;
	
	int rows = phi.rows;
	int cols = phi.cols;
	Mat Ix, Iy;
	Sobel(gray, Ix, CV_16SC1, 1, 0, 1, 0.5);
	Sobel(gray, Iy, CV_16SC1, 0, 1, 1, 0.5);

	Mat f = Ix.mul(Ix) + Iy.mul(Iy); // f = Ix.^2 + Iy.^2
	Mat g;
	add(f, Mat::ones(rows, cols, CV_8UC1), g, noArray(), CV_64FC1);
	divide(1, g, g); // g = 1 ./ (1+f);

	Mat vx, vy;
	Sobel(g, vx, CV_64FC1, 1, 0, 1, 0.5);
	Sobel(g, vy, CV_64FC1, 0, 1, 1, 0.5);

	for (int n = 1; n <= iter_outer; n++)
	{
		drlse_edge(phi, g, vx, vy, lambda, mu, alfa, epsilon, timestep, iter_inner);
		debugLevelSet(img, phi);
	}

	alfa = 0;
	int iter_refine = 5;
	for (int n = 1; n <= iter_refine; n++)
	{
		drlse_edge(phi, g, vx, vy, lambda, mu, alfa, epsilon, timestep, iter_inner);
		debugLevelSet(img, phi);
	}

}

void segFrame(Video &video)
{
	int rows = video.frame[0].rows;
	int cols = video.frame[0].cols;

	double c0 = 2.0;
	Mat initialLSF = Mat::ones(rows, cols, CV_64FC1);
	initialLSF = initialLSF.mul(c0);
	Rect roi(4, 4, 75, 55);
	Mat LSF_roi(initialLSF, roi);
	LSF_roi = LSF_roi.mul(-1);
	Mat phi = initialLSF.clone();
	
	LevelSet(video.frame[0], video.gray[0], phi);
}
void onMouse(int event, int x, int y, int flags, void* param)
{
	mouseArg *mouse_arg = (mouseArg*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		if (mouse_arg->pushdown_flag == 0) mouse_arg->pushdown_flag = 1;
	}
	if (event == EVENT_LBUTTONUP)
	{
		if (mouse_arg->pushdown_flag == 1)
		{
			mouse_arg->selected = Point(x, y);
			mouse_arg->pushdown_flag = 2;
		}
	}
}
void floodFill(const Mat &img, Mat &seed_points, const Scalar diff, int &sum_points)
{

}
int main(int argc, char** argv)
{
	Video video("test4");
	
	mouseArg mouse_arg;
	char* window_name = "video";
	namedWindow(window_name);
	imshow(window_name, video.frame[0]);
	setMouseCallback(window_name, onMouse, &mouse_arg);
	waitKey(0);

	if (mouse_arg.pushdown_flag == 2) cout << mouse_arg.selected << endl;

	int rows = video.frame[0].rows;
	int cols = video.frame[0].cols;
	int sum_points;
	Mat seed_points = Mat::zeros(rows, cols, CV_8UC1);
	seed_points.ptr<uchar>(mouse_arg.selected.y)[mouse_arg.selected.x] = 1;
	sum_points = 1;
	//cout << seed_points << endl;
	floodFill(video.frame[0], seed_points, sum_points);

	//segFrame(video);

 	return 0;
}