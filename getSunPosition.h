#ifndef GETSUNPOSITION_H
#define GETSUNPOSITION_H
#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv.hpp>
#include <vector>

#define SCALE 100
#define LINE_POINT_NUM 18

static const float PI = 3.1415926;
static const int FRAME_SUM =  150;
typedef CvPoint3D64f XgyVector3D;
class getSunPosition
{
private:
	 cv::Mat gray_prev; // previous frame of light tracking
     cv::Mat gray;   // current frame of light tracking
    cv::Mat rotateMat;
   cv::Mat tranVector;
     double focalLength;
    
     cv::Point3f camPos;
    typedef struct Line{
	//直线上的两点；
	cv::Point2f x1;
	cv::Point2f x2;

	//设定直线
	void setLine(cv::Point2f p1,cv::Point2f p2);
	cv::Point2f getDirection();	//获取直线的方向

    }XgyLine;
   typedef struct Plane //平面数据结构ax+by+cz+d=0;
   {
	double a;
	double b;
	double c;
	double d;

	void setPlane(double a,double b,double c,double d);
   }XgyPlane;
public:
	 double sunAzimuth;//方位角
    double sunZenith;
     double sunAltitude;//高度角，太阳方向和地面的夹角
	 double width;
  double height;
private:
	void Calibrate(cv::Mat frame, cv::Point2f data_point[11], double &focal_length, cv::Mat &R, cv::Mat &T);
	void find_line_point(cv::Point2f data[11], cv::Point2f line_point[4][LINE_POINT_NUM]);
	//
    cv::Point2f  point_on_line(cv::Point2f pt, cv::Point2f line_start, cv::Point2f line_end);
	/**求焦距 返回焦距**/
    float get_focal_length(cv::Point2f v1, cv::Point2f v2, const int& width, const int& height);
	double get_focal_length2(cv::Point2f v1, cv::Point2f v2);
	// line is composed of four float: (x1, y1; x2, y2)
    cv::Point2f	get_vanishing_point(const float* line1, const float* line2);
	// light tracking
    void tracking(cv::Mat & frame, cv::Point2f point_data[11]);
	void tracking(cv::Mat & frame , cv::Point2f point_data[11] , int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM]);
	// get mid points when out of range
    bool half_line(int out_point_num, cv::Point2f pre_point[11], cv::Point2f now_point[11]);
	int getCamPara(int W, int H, const double* pointVec, double& phy, double &theta, double &xxx, double &yyy);
	void get_vp_for_R(cv::Mat frame, const float *pointVec, cv::Point2f &v1, cv::Point2f &v2, double &f);
    void get_rotation_matrix(const cv::Point2f v1, const cv::Point2f v2, const double f, const cv::Point3f Zc, cv::Mat &R);
    void get_rotation_mat(const cv::Point2f v1, const cv::Point2f v2, const double f, cv::Mat &R);
	bool get_translation_vector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, double T[]);
    bool adjust_rotation_matrix(cv::Mat src_R, cv::Mat &dst_R);
    void para_to_decare(const double r, const double phi, const double theta, double &x, double &y, double &z);
    void switch_to_cam_coordinate(const cv::Mat _Pw, const cv::Mat _R, const cv::Mat _t, cv::Mat &_Pc, cv::Mat &_M);
	bool reAllocation(const cv::Mat frame, cv::Point2f data[11] ,int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM]);//点出界时重新划分数据点
    void extend(cv::Point2f point_data[11] ) ; //延长数据线
    /**中心对称 将src 以origin 为中心作中心对称点**/
    void central_symmetry( cv::Point2f src, cv::Point2f origin, cv::Point2f& dst );
    /**从旋转矩阵求x-axis 和 y-axis 方便画坐标轴**/
    void xy_axis(cv::Mat R, cv::Point2f& x_axis, cv::Point2f& y_axis);
    void setRT(cv::Mat R,cv::Mat T);
    bool img2world(cv::Point2f iP,XgyPlane pl,cv::Point3f &wP);
    double getVectorLength3D(XgyVector3D v);
    double normalizeVector3D(XgyVector3D &v);
    XgyVector3D setVector3D(cv::Point3f p1,cv::Point3f p2);
    void getcamPose(cv::Point3f & campose);
    double dotProduct3D(XgyVector3D v1,XgyVector3D v2);
    void scalarMul(double num,XgyVector3D src,XgyVector3D &tar);
    bool setSunAngleEx(std::vector<XgyLine> lineArray);
    void setFlocallength(double f);
    void printAngle();
    void setWandH(double w,double h);
    void setVecA(std::vector<XgyLine>& lineArray,std::vector<cv::Point2f> data);
    void getAngle(cv::Mat R,cv::Mat T,double f,std::vector<cv::Point2f> data,double&_sunAzimuth,double& _sunAltitude,double w,double h );//data为三个点，第一个点为阴影点，第二个点为脚点，第三个点为头点,f为focallength
    bool world2img(cv::Mat rotateMat,cv::Mat tranVector,double focalLength,cv::Point3d wP, cv::Point2d &iP);
    double getXYlength(std::vector<cv::Point2d> data,double scale);
    double getFocalLength(cv::Point2f vx, cv::Point2f vy);
    void getIntrinsicParameter(double focal_length, cv::Point2f principal_point, cv::Mat & K);
    void getRotateMat(cv::Point2f vx, cv::Point2f vy, double focal_length, cv::Mat &R);
    void adjustRotateMat(cv::Mat srcR, cv::Mat &dstR);
    void getSunPos(const int H, const int W, const cv::Point2f data_point[11],cv::Point2d &sunPos, double &theta, double &phi);
    /**求theta, phi*/
    void get_theta_and_phi(const cv::Mat w, const cv::Point2d sunPos, const cv::Point2d v_prime, \
                       const cv::Point2d vx, const cv::Point2d vy, const cv::Point2d vz,  double &theta, double &phi);
    bool getTranslationVector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, cv::Mat &T);cv::Point2f getVanishPoint(cv::Point2f v0, cv::Point2f v1, cv::Point2f v2, cv::Point2f v3);
    cv::Point2f getLineIntersection(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
    bool adjust_vanishing_point(const cv::Point2f  _v1, const cv::Point2f _v2, const cv::Point2f origin, cv::Point2f &v1, cv::Point2f &v2);
    void adjustInteraction(cv::Point2f src_point[11], cv::Point2f dst_point[11]);
	bool getPlanefromLine(XgyLine l, XgyPlane &pl);
	void fitline_reAllocation(cv::Point2f data[11] , cv::Point2f line_point[4][LINE_POINT_NUM]);
	float distance(cv::Point2f pt1, cv::Point2f pt2);
    /**  求两直线的的焦点 p0, p1代表第一条直线。 p2, p3代表第二条直线  返回交点*/
    cv::Point2f get_line_intersection(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
	int draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, std::vector<cv::Point2d> imagePoints ,bool change_flag);
    int draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, float x, float y);
	int draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, std::vector<cv::Point2d> imagePoints);
    int draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame,bool change_flag ,float x , float y);
	void draw_line_2(cv::Mat & temp , cv::Point2f data[11] ,int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM]);
public:
	// point_data should have 11 elements
    int sunPositionProcess(int init_num, std::string fileName, cv::Point2f point_data[]);
    

};


#endif