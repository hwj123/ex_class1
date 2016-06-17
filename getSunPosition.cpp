#include"getSunPosition.h"
void getSunPosition::Calibrate(cv::Mat frame, cv::Point2f data_point[11], double &focal_length, cv::Mat &R, cv::Mat &T)
{
    assert(R.rows == 3 && R.cols == 3);
    assert(T.rows == 3 && T.cols == 1);
    cv::Point2f point[11];
    cv::Point2f principal_point;
    principal_point.x = (double)frame.cols/2.0;
    principal_point.y = (double)frame.rows/2.0;

    for(int i=0; i<11; i++)
    {
        point[i] = data_point[i]-principal_point;
    }
    cv::Point2f origin = point[10];

    cv::Point2f v1 = getVanishPoint(point[0], point[1], point[2], point[3]);
    cv::Point2f v2 = getVanishPoint(point[4], point[5], point[6], point[7]);


    focal_length = getFocalLength(v1, v2);
    getRotateMat(v1, v2, focal_length, R);
    getTranslationVector(origin,v1, R, focal_length, T);
}
float getSunPosition::distance(cv::Point2f pt1, cv::Point2f pt2)
{
    return sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
}

cv::Point2f  getSunPosition::point_on_line(cv::Point2f pt, cv::Point2f line_start, cv::Point2f line_end)
{
    cv::Point2f v_ret;

    cv::Mat A(2, 2, CV_32F);
    A.at<float>(0, 0) = line_end.x - line_start.x;
    A.at<float>(0, 1) = line_end.y - line_start.y;
    A.at<float>(1, 0) = line_end.y - line_start.y;
    A.at<float>(1, 1) = line_start.x - line_end.x;
    cv::Mat B(2, 1, CV_32F);
    B.at<float>(0, 0) = (line_end.x - line_start.x)*pt.x + (line_end.y - line_start.y)*pt.y;
    B.at<float>(1, 0) = line_start.x*(line_end.y - line_start.y) - line_start.y*(line_end.x - line_start.x);
    cv::Mat ret_mat = A.inv()*B;

    v_ret.x = ret_mat.at<float>(0, 0);
    v_ret.y = ret_mat.at<float>(1, 0);
    return v_ret;
}

float getSunPosition::get_focal_length(cv::Point2f v1, cv::Point2f v2, const int& width, const int& height)
{
    cv::Point2f oi; // main point
    cv::Point2f vi; // projection of oi on vanishing line

    oi.x = (float)(width / 2);
    oi.y = (float)(height / 2);

    vi = point_on_line(oi, v1, v2);

    float oi_vi = distance(oi, vi);
    float v1_vi = distance(v1, vi);
    float v2_vi = distance(v2, vi);

    float oc_vi = sqrt(v1_vi * v2_vi);
    float f = sqrt(oc_vi * oc_vi - oi_vi*oi_vi);
    return f;
}

double getSunPosition::get_focal_length2(cv::Point2f v1, cv::Point2f v2)
{
    return sqrt( -1*(v1.x * v2.x + v1.y * v2.y) );
//      return sqrt( 1*(v1.x * v2.x + v1.y * v2.y) );
}

cv::Point2f getSunPosition::get_line_intersection(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
{
    cv::Point2f vp;
    float x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = p0.x;   y0 = p0.y;
    x1 = p1.x;   y1 = p1.y;
    x2 = p2.x;   y2 = p2.y;
    x3 = p3.x;   y3 = p3.y;
    vp.y = ( (y0-y1)*(y3-y2)*x0 + (y3-y2)*(x1-x0)*y0 + (y1-y0)*(y3-y2)*x2 + (x2-x3)*(y1-y0)*y2 ) / ( (x1-x0)*(y3-y2) + (y0-y1)*(x3-x2) );
    vp.x = x2 + (x3-x2)*(vp.y-y2) / (y3-y2);
    return vp;
}

cv::Point2f	getSunPosition::get_vanishing_point(const float* line1, const float* line2)
{
    cv::Point2f vp;
    float x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = line1[0];  y0 = line1[1];
    x1 = line1[2];  y1 = line1[3];
    x2 = line2[0];  y2 = line2[1];
    x3 = line2[2];  y3 = line2[3];
    vp.y = ( (y0-y1)*(y3-y2)*x0 + (y3-y2)*(x1-x0)*y0 + (y1-y0)*(y3-y2)*x2 + (x2-x3)*(y1-y0)*y2 ) / ( (x1-x0)*(y3-y2) + (y0-y1)*(x3-x2) );

    vp.x = x2 + (x3-x2)*(vp.y-y2) / (y3-y2);
    return vp;
}



void getSunPosition::tracking(cv::Mat & frame , cv::Point2f point_data[11] , int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM]) //光流追踪
{
//    static cv::Mat gray_prev;	// 先前图片
    cv::Mat flow;
    cvtColor(frame, gray, CV_BGR2GRAY);
    if (gray_prev.empty())
    {
        gray.copyTo(gray_prev);
        return;
    }

    //判断出界与否然后再判断是否跟踪
    cv::calcOpticalFlowFarneback(gray_prev,gray,flow,0.5,3,15,3,5,1.2,0);
    for(int i=0;i<11;i++)
    {
        if(out_point_mark[i])
        {
            cv::Point2f fxy = flow.at<cv::Point2f>(point_data[i].y, point_data[i].x);
            point_data[i].x += fxy.x;
            point_data[i].y += fxy.y;
        }

    }

    for(int i = 0 ; i < 4 ;i++)
    {
        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
           if(out_line_mark[i][j])
           {
               cv::Point2f fxy = flow.at<cv::Point2f>(line_point[i][j].y, line_point[i][j].x);
               line_point[i][j].x += fxy.x;
               line_point[i][j].y += fxy.y;           }
        }
    }
    swap(gray_prev, gray);
}

bool getSunPosition::half_line(int out_point_num, cv::Point2f pre_point[11], cv::Point2f now_point[11])
{
    int line_num;
    bool if_frame_break = false;
    line_num = out_point_num / 2;
    line_num *= 2;

    pre_point[out_point_num].x = now_point[out_point_num].x = (pre_point[line_num].x + pre_point[line_num + 1].x) / 2;
    pre_point[out_point_num].y = now_point[out_point_num].y = (pre_point[line_num].y + pre_point[line_num + 1].y) / 2;
    if ((pre_point[line_num].x - pre_point[line_num + 1].x)*(pre_point[line_num].x - pre_point[line_num + 1].x) + (pre_point[line_num].y - pre_point[line_num + 1].y) * (pre_point[line_num].y - pre_point[line_num + 1].y) < 100)
        if_frame_break = true;
    return if_frame_break;
}
void getSunPosition::find_line_point(cv::Point2f data[11], cv::Point2f line_point[4][LINE_POINT_NUM])//找到线中的点
{
    for(int i = 0 ; i <4 ; i++)
    {
        cv::Point2f step;
        step.x = (data[2*i+1].x - data[2*i].x)/(LINE_POINT_NUM + 1 );//一条直线划分为9个小段
        step.y = (data[2*i+1].y - data[2*i].y)/(LINE_POINT_NUM + 1);

        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
            line_point[i][j].x = data[2*i].x+(j+1)*step.x;
            line_point[i][j].y = data[2*i].y+(j+1)*step.y;
        }
    }
}


void getSunPosition::extend(cv::Point2f point_data[11] )//延长数据线
{
    for(int line_num = 0 ; line_num < 4 ; line_num++)
    {
        if((point_data[2*line_num].x - point_data[2*line_num + 1].x)*(point_data[2*line_num].x - point_data[2*line_num + 1].x) + (point_data[2*line_num].y - point_data[2*line_num + 1].y) * (point_data[2*line_num].y - point_data[2*line_num + 1].y) < 22500)
        {
            float length = sqrt((point_data[2*line_num].x - point_data[2*line_num + 1].x)*(point_data[2*line_num].x - point_data[2*line_num + 1].x) + (point_data[2*line_num].y - point_data[2*line_num + 1].y) * (point_data[2*line_num].y - point_data[2*line_num + 1].y));
            point_data[2*line_num+1].x = ((150 - length)/(2*length))*(point_data[2*line_num+1].x - point_data[2*line_num].x) + point_data[2*line_num+1].x;
            point_data[2*line_num+1].y = ((150 - length)/(2*length))*(point_data[2*line_num+1].y - point_data[2*line_num].y) + point_data[2*line_num+1].y;
            point_data[2*line_num].x = ((150 - length)/(2*length))*(point_data[2*line_num].x - point_data[2*line_num+1].x) + point_data[2*line_num].x;
            point_data[2*line_num].y = ((150 - length)/(2*length))*(point_data[2*line_num].y - point_data[2*line_num+1].y) + point_data[2*line_num].y;

        }
    }

}

bool getSunPosition::reAllocation(const cv::Mat frame, cv::Point2f data[11] ,int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM])//点出界时重新划分数据点
{
    for(int line_num = 0 ; line_num < 4 ; line_num++)
    {
        cv::Point2f *first_point = NULL , *last_point = NULL;
        int on_line_num = 0;//线上的点数
        if(out_point_mark[line_num*2])
        {
            on_line_num++;
            first_point = &data[line_num*2];
        }
        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
            if(out_line_mark[line_num][j])
            {
                if(on_line_num)//头点已记录时
                {
                    last_point = &line_point[line_num][j];
                }
                else
                {
                    first_point = &data[line_num*2];
                }
                on_line_num++;

            }
        }
        if(out_point_mark[line_num*2+1])
        {
            if(on_line_num)//头点已记录时
            {
                last_point = &data[line_num*2+1];
            }
            else
            {
                first_point = &data[line_num*2+1];
            }
            on_line_num++;
        }
        if(on_line_num < 15) return true;//线上点过少时结束跟踪
        if(last_point == NULL) return true;//线上只有一个点时结束跟踪
        if((data[2*line_num].x - data[2*line_num + 1].x)*(data[2*line_num].x - data[2*line_num + 1].x) + \
                            (data[2*line_num].y - data[2*line_num + 1].y) * (data[2*line_num].y - data[2*line_num + 1].y) < ( frame.cols / 20) )//线过短时不再循环
        {
            return true;
        }


        data[line_num*2] = *first_point;//记录新的端点
        data[line_num*2+1] = *last_point;
    }
    find_line_point(data,line_point);//由新的端点划分所有新点
    for(int i = 0 ; i < 11 ; i++)//重新初始化
    {
       out_point_mark[i]=1;
    }
    for(int i = 0 ; i <4 ; i ++ )
    {
        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
            out_line_mark[i][j]=1;
        }
    }
    return false;

}

void getSunPosition::fitline_reAllocation(cv::Point2f data[11] , cv::Point2f line_point[4][LINE_POINT_NUM])//拟合直线，重分配点
{
    std::vector<cv::Point2f> line[4];//直线拟合
    std::vector<float> new_line[4];
    cv::Point2f new_line_point[11];
    for(int i = 0 ; i <4 ;i++)
    {
        line[i].push_back(data[ i*2]);
        for(int j =0 ;j <LINE_POINT_NUM ;j++ )
        {
            line[i].push_back(line_point[i][j]);
        }
        line[i].push_back(data[i*2+1]);
        cv::fitLine(line[i],new_line[i],CV_DIST_L1,0,0.01,0.01);
        new_line_point[i*2].x =((new_line[i][1] * new_line[i][1] * new_line[i][2]) + (new_line[i][0] * new_line[i][0] * data[i*2].x) + (new_line[i][0] * new_line[i][1] * (data[i*2].y - new_line[i][3])))  ;
        new_line_point[i*2+1].x =((new_line[i][1] * new_line[i][1] * new_line[i][2]) + (new_line[i][0] * new_line[i][0] * data[i*2+1].x) + (new_line[i][0] * new_line[i][1] * (data[i*2+1].y - new_line[i][3])));
        new_line_point[i*2].y=(new_line_point[i*2].x-new_line[i][2])*new_line[i][1]/new_line[i][0]+new_line[i][3];
        new_line_point[i*2+1].y=(new_line_point[i*2+1].x-new_line[i][2])*new_line[i][1]/new_line[i][0]+new_line[i][3];
    }
    for(int i =0 ; i< 8 ; i++)
    {
        data[i]=new_line_point[i];
    }
    find_line_point(data,line_point);

}

/*get parameter for rotation matrix*/
void getSunPosition::get_vp_for_R(cv::Mat frame, const float *pointVec, cv::Point2f &v1, cv::Point2f &v2, double &f)
{
    float* lineTemp1 = new float[4];
    float* lineTemp2 = new float[4];
    lineTemp1[0] = pointVec[0];
    lineTemp1[1] = pointVec[1];
    lineTemp1[2] = pointVec[2];
    lineTemp1[3] = pointVec[3];

    lineTemp2[0] = pointVec[4];
    lineTemp2[1] = pointVec[5];
    lineTemp2[2] = pointVec[6];
    lineTemp2[3] = pointVec[7];
    v1 = get_vanishing_point(lineTemp1, lineTemp2);
    v1.x = v1.x - 0.5*frame.cols;
    v1.y = v1.y - 0.5*frame.rows;
    // vanishing point y
    lineTemp1[0] = pointVec[8];
    lineTemp1[1] = pointVec[9];
    lineTemp1[2] = pointVec[10];
    lineTemp1[3] = pointVec[11];

    lineTemp2[0] = pointVec[12];
    lineTemp2[1] = pointVec[13];
    lineTemp2[2] = pointVec[14];
    lineTemp2[3] = pointVec[15];
    v2 = get_vanishing_point(lineTemp1, lineTemp2);
    v2.x = v2.x - 0.5*frame.cols;
    v2.y = v2.y - 0.5*frame.rows;
    f = get_focal_length2(v1, v2);

}

void getSunPosition::draw_line_2(cv::Mat & temp , cv::Point2f data[11] ,int out_point_mark[11] , cv::Point2f line_point[4][LINE_POINT_NUM] , int out_line_mark[4][LINE_POINT_NUM])
{
    //画线
    cv::line(temp,data[0],data[1],cv::Scalar(255,0,0),2,8);//第一对平行线
    cv::line(temp,data[2],data[3],cv::Scalar(255,0,0),2,8);

    cv::line(temp,data[4],data[5],cv::Scalar(0,255,0),2,8);//第二对
    cv::line(temp,data[6],data[7],cv::Scalar(0,255,0),2,8);

    //cv::line(temp,data[8],data[9],cv::Scalar(0,0,255),2,8);//人头顶到阴影点的直线

    //画点
    for(int point_num = 0 ; point_num < 11 ; point_num++)
    {
        if(out_point_mark[point_num])
        {
            if(point_num <= 3)	cv::circle(temp,data[point_num],3,cv::Scalar(255,0,0),-1);
            else if(point_num > 3 && point_num <= 7 ) cv::circle(temp,data[point_num],3,cv::Scalar(0,255,0),-1);
//            else if(point_num ==8 ) cv::circle(temp,data[point_num],3,cv::Scalar(0,0,255),-1);
            else if(point_num == 10) cv::circle(temp,data[point_num],3,cv::Scalar(0,0,255),-1);
        }

    }

//    for(int line_num = 0 ; line_num < 4 ; line_num++ )
//    {
//        if(line_num < 2)
//        {
//            for(int i =0 ; i <LINE_POINT_NUM ;i++)
//            {
//                if(out_line_mark[line_num][i])
//                {
//                    cv::circle(temp,line_point[line_num][i],3,cv::Scalar(255,0,0),-1);
//                }
//
//            }
//        }
//        else
//        {
//            for(int i =0 ; i <LINE_POINT_NUM ;i++)
//            {
//                if(out_line_mark[line_num][i])
//                {
//                    cv::circle(temp,line_point[line_num][i],3,cv::Scalar(0,255,0),-1);
//                }
//            }
//        }
//
//    }

    //画箭头
    //const float PI = 3.14159;
//    cv::Point2i arrow;
//    float len = 15; //箭头长度
//    float arrow_angle = 45;
//    float angle = atan2((double)(data[8].y - data[9].y), (double)(data[8].x - data[9].x));  //计算箭头指向角度
//
//    arrow.x = data[9].x + len * cos(angle + PI *  arrow_angle/ 180);   //箭头夹角15度
//    arrow.y = data[9].y + len * sin(angle + PI * arrow_angle/ 180);
//    cv::line(temp, data[9], arrow, cv::Scalar(0,0,255), 2, 8);
//    arrow.x = data[9].x + len * cos(angle - PI * arrow_angle/ 180);
//    arrow.y = data[9].y + len * sin(angle - PI * arrow_angle/ 180);
//    cv::line(temp, data[9], arrow, cv::Scalar(0,0,255), 2, 8);
}

int getSunPosition::draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, float x, float y)
{


	float line[4][4];
	cv::Point2f vp[2];
	cv::Point2f coordinate_sys_point[3];
	cv::Point2f sun;
	float coordinate_sys_lenth = frame.cols / 8;

	for (int line_num = 0; line_num < 4; line_num++)
	{
		line[line_num][0] = point_data[line_num * 2].x;
		line[line_num][1] = point_data[line_num * 2].y;
		line[line_num][2] = point_data[line_num * 2 + 1].x;
		line[line_num][3] = point_data[line_num * 2 + 1].y;
	}

	vp[0] = get_vanishing_point(line[0], line[1]);
	vp[1] = get_vanishing_point(line[2], line[3]);


	coordinate_sys_point[0] = point_data[10];
	coordinate_sys_point[1].x = point_data[10].x + (vp[0].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));
	coordinate_sys_point[1].y = point_data[10].y + (vp[0].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));
	coordinate_sys_point[2].x = point_data[10].x + (vp[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
	coordinate_sys_point[2].y = point_data[10].y + (vp[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
//	if ((point_data[9].x - point_data[8].x)*(x - point_data[10].x) + (point_data[9].y - point_data[8].y)*(y - point_data[10].y) < 0)
//	{
//		sun.x = point_data[10].x + (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
//		sun.y = point_data[10].y + (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
//	}
//	else
//	{
//		sun.x = point_data[10].x - (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
//		sun.y = point_data[10].y - (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
//	}
    if(y < point_data[10].y)
    {
        sun.x = point_data[10].x + (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
		sun.y = point_data[10].y + (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
    }
    else
	{
		sun.x = point_data[10].x - (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
		sun.y = point_data[10].y - (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
	}

	//右手坐标系判断
	cv::Mat lvector[3];//坐标轴方向向量
	lvector[0] = cv::Mat_<double>(2, 1);
	lvector[1] = cv::Mat_<double>(2, 1);
	lvector[2] = cv::Mat_<double>(2, 1);
	lvector[0].at<double>(0, 0) = coordinate_sys_point[1].x - point_data[10].x;
	lvector[0].at<double>(1, 0) = coordinate_sys_point[1].y - point_data[10].y;
	lvector[1].at<double>(0, 0) = coordinate_sys_point[2].x - point_data[10].x;
	lvector[1].at<double>(1, 0) = coordinate_sys_point[2].y - point_data[10].y;
	//std::cout << "lvector12" << " " << coordinate_sys_point[1].x - point_data[10].x << " " << lvector[0].at<double>(0, 0) << " " << coordinate_sys_point[2].x - point_data[10].x<<" "<<lvector[1].at<double>(0, 0) << std::endl;
	double angle_cos = (lvector[0].at<double>(0, 0) * lvector[1].at<double>(0, 0) + lvector[0].at<double>(1, 0) * lvector[1].at<double>(1, 0)) / (sqrt(lvector[0].at<double>(0, 0)*lvector[0].at<double>(0, 0) + lvector[0].at<double>(1, 0)*lvector[0].at<double>(1, 0))*sqrt(lvector[1].at<double>(0, 0)*lvector[1].at<double>(0, 0) + lvector[1].at<double>(1, 0)*lvector[1].at<double>(1, 0)));//夹角cos值
	double xy_angle = acos(angle_cos);
	double angle_sin = sin(xy_angle);
	cv::Mat R = cv::Mat_<double>(2, 2);//角度旋转矩阵
	R.at<double>(0, 0) = angle_cos;
	R.at<double>(0, 1) = -angle_sin;
	R.at<double>(1, 0) = angle_sin;
	R.at<double>(1, 1) = angle_cos;

	cv::gemm(R, lvector[0], 1, 0, 0, lvector[2], 0);
	double dx = lvector[2].at<double>(0, 0) - lvector[1].at<double>(0, 0), dy = lvector[2].at<double>(1, 0) - lvector[1].at<double>(1, 0);

	if ((dx > 1) || (dx < -1) || (dy > 1) || (dy < -1))
	{
		cv::Point2f temp;
		temp = coordinate_sys_point[1];
		coordinate_sys_point[1] = coordinate_sys_point[2];
		coordinate_sys_point[2] = temp;
		//std::cout << "translate was happend " << std::endl;
	}

    coordinate_sys_point[1].x = cvRound(coordinate_sys_point[1].x);
    coordinate_sys_point[1].y = cvRound(coordinate_sys_point[1].y);
    coordinate_sys_point[2].x = cvRound(coordinate_sys_point[2].x);
    coordinate_sys_point[2].y = cvRound(coordinate_sys_point[2].y);
    sun.x = cvRound(sun.x);
    sun.y = cvRound(sun.y);

    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[1], cv::Scalar(0, 0, 255), 2, 8);
    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[2], cv::Scalar(0, 255, 0), 2, 8);
    cv::line(frame, coordinate_sys_point[0], sun, cv::Scalar(0, 255, 255), 2, 8);

    //画箭头
   // float PI = 3.14159;
    cv::Point2i arrow;
    float len = 15; //箭头长度
    float arrow_angle = 30;



    float angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[1].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[1].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[1].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[1].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    arrow.x = coordinate_sys_point[1].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[1].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion;
    posion.x = coordinate_sys_point[1].x;
    posion.y = coordinate_sys_point[1].y - 10;
    cv::putText(frame, "y", posion, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[2].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[2].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[2].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[2].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    arrow.x = coordinate_sys_point[2].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[2].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion1;
    posion1.x = coordinate_sys_point[2].x;
    posion1.y = coordinate_sys_point[2].y - 10;
    cv::putText(frame, "x", posion1, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - sun.y), (double)(coordinate_sys_point[0].x - sun.x));  //计算箭头指向角度
    arrow.x = sun.x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = sun.y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    arrow.x = sun.x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = sun.y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion2;
    posion2.x = sun.x;
    posion2.y = sun.y - 10;
    cv::putText(frame, "sun", posion2, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255, 255), 1, 8);

    return 0;
}
int getSunPosition::draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, std::vector<cv::Point2d> imagePoints)
{
	float coordinate_sys_lenth = frame.cols / 8;//坐标轴长度

	cv::Point2f coordinate_sys_point[3];
	cv::Point2f sun;
	coordinate_sys_point[0] = point_data[10];//脚点

	//坐标轴长度规划
	coordinate_sys_point[1].x = point_data[10].x + (imagePoints[0].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[0].x)*(point_data[10].x - imagePoints[0].x) + (point_data[10].y - imagePoints[0].y)*(point_data[10].y - imagePoints[0].y));
	coordinate_sys_point[1].y = point_data[10].y + (imagePoints[0].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[0].x)*(point_data[10].x - imagePoints[0].x) + (point_data[10].y - imagePoints[0].y)*(point_data[10].y - imagePoints[0].y));
	coordinate_sys_point[2].x = point_data[10].x + (imagePoints[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[1].x)*(point_data[10].x - imagePoints[1].x) + (point_data[10].y - imagePoints[1].y)*(point_data[10].y - imagePoints[1].y));
	coordinate_sys_point[2].y = point_data[10].y + (imagePoints[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[1].x)*(point_data[10].x - imagePoints[1].x) + (point_data[10].y - imagePoints[1].y)*(point_data[10].y - imagePoints[1].y));
	sun.x = point_data[10].x + (imagePoints[2].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[2].x)*(point_data[10].x - imagePoints[2].x) + (point_data[10].y - imagePoints[2].y)*(point_data[10].y - imagePoints[2].y));
	sun.y = point_data[10].y + (imagePoints[2].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[2].x)*(point_data[10].x - imagePoints[2].x) + (point_data[10].y - imagePoints[2].y)*(point_data[10].y - imagePoints[2].y));

	cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[1], cv::Scalar(0, 0, 255), 2, 8);
	cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[2], cv::Scalar(0, 255, 0), 2, 8);
	cv::line(frame, coordinate_sys_point[0], sun, cv::Scalar(0, 255, 255), 2, 8);



    //画箭头
	// float PI = 3.14159;
	cv::Point2i arrow;
	float len = 15; //箭头长度
	float arrow_angle = 30;



	float angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[1].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[1].x));  //计算箭头指向角度
	arrow.x = coordinate_sys_point[1].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
	arrow.y = coordinate_sys_point[1].y + len * sin(angle + PI * arrow_angle / 180);
	cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
	arrow.x = coordinate_sys_point[1].x + len * cos(angle - PI * arrow_angle / 180);
	arrow.y = coordinate_sys_point[1].y + len * sin(angle - PI * arrow_angle / 180);
	cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
	//黄雯嘉在此修改，标明x轴
	cv::Point2f posion;
	posion.x = coordinate_sys_point[1].x;
	posion.y = coordinate_sys_point[1].y - 10;
	cv::putText(frame, "x", posion, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255, 255), 1, 8);

	angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[2].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[2].x));  //计算箭头指向角度
	arrow.x = coordinate_sys_point[2].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
	arrow.y = coordinate_sys_point[2].y + len * sin(angle + PI * arrow_angle / 180);
	cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
	arrow.x = coordinate_sys_point[2].x + len * cos(angle - PI * arrow_angle / 180);
	arrow.y = coordinate_sys_point[2].y + len * sin(angle - PI * arrow_angle / 180);
	cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
	//黄雯嘉在此修改，标明x轴
	cv::Point2f posion1;
	posion1.x = coordinate_sys_point[2].x;
	posion1.y = coordinate_sys_point[2].y - 10;
	cv::putText(frame, "y", posion1, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0, 255), 1, 8);

	angle = atan2((double)(coordinate_sys_point[0].y - sun.y), (double)(coordinate_sys_point[0].x - sun.x));  //计算箭头指向角度
	arrow.x = sun.x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
	arrow.y = sun.y + len * sin(angle + PI * arrow_angle / 180);
	cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
	arrow.x = sun.x + len * cos(angle - PI * arrow_angle / 180);
	arrow.y = sun.y + len * sin(angle - PI * arrow_angle / 180);
	cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
	//黄雯嘉在此修改，标明x轴
	cv::Point2f posion2;
	posion2.x = sun.x;
	posion2.y = sun.y - 10;
	cv::putText(frame, "sun", posion2, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255, 255), 1, 8);

	return 0;
}

int getSunPosition::draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame, std::vector<cv::Point2d> imagePoints ,bool change_flag)
{


	float line[4][4];
	cv::Point2f vp[2];
	cv::Point2f coordinate_sys_point[3];
	cv::Point2f sun;
	float coordinate_sys_lenth = frame.cols / 8;

	for (int line_num = 0; line_num < 4; line_num++)
	{
		line[line_num][0] = point_data[line_num * 2].x;
		line[line_num][1] = point_data[line_num * 2].y;
		line[line_num][2] = point_data[line_num * 2 + 1].x;
		line[line_num][3] = point_data[line_num * 2 + 1].y;
	}

	vp[0] = get_vanishing_point(line[0], line[1]);
	vp[1] = get_vanishing_point(line[2], line[3]);


	coordinate_sys_point[0] = point_data[10];
	coordinate_sys_point[1].x = point_data[10].x + (vp[0].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));
	coordinate_sys_point[1].y = point_data[10].y + (vp[0].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));

	sun.x = point_data[10].x + (imagePoints[2].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[2].x)*(point_data[10].x - imagePoints[2].x) + (point_data[10].y - imagePoints[2].y)*(point_data[10].y - imagePoints[2].y));
	sun.y = point_data[10].y + (imagePoints[2].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - imagePoints[2].x)*(point_data[10].x - imagePoints[2].x) + (point_data[10].y - imagePoints[2].y)*(point_data[10].y - imagePoints[2].y));

	if (change_flag)
	{
		coordinate_sys_point[2].x = point_data[10].x - (vp[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
		coordinate_sys_point[2].y = point_data[10].y - (vp[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
	}
	else
	{
		coordinate_sys_point[2].x = point_data[10].x + (vp[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
		coordinate_sys_point[2].y = point_data[10].y + (vp[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
	}



    coordinate_sys_point[1].x = cvRound(coordinate_sys_point[1].x);
    coordinate_sys_point[1].y = cvRound(coordinate_sys_point[1].y);
    coordinate_sys_point[2].x = cvRound(coordinate_sys_point[2].x);
    coordinate_sys_point[2].y = cvRound(coordinate_sys_point[2].y);
    sun.x = cvRound(sun.x);
    sun.y = cvRound(sun.y);

    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[1], cv::Scalar(0, 0, 255), 2, 8);
    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[2], cv::Scalar(0, 255, 0), 2, 8);
    cv::line(frame, coordinate_sys_point[0], sun, cv::Scalar(0, 255, 255), 2, 8);
//画箭头
   // float PI = 3.14159;
    cv::Point2i arrow;
    float len = 15; //箭头长度
    float arrow_angle = 30;



    float angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[1].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[1].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[1].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[1].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    arrow.x = coordinate_sys_point[1].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[1].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion;
    posion.x = coordinate_sys_point[1].x;
    posion.y = coordinate_sys_point[1].y - 10;
    cv::putText(frame, "x", posion, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[2].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[2].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[2].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[2].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    arrow.x = coordinate_sys_point[2].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[2].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion1;
    posion1.x = coordinate_sys_point[2].x;
    posion1.y = coordinate_sys_point[2].y - 10;
    cv::putText(frame, "y", posion1, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - sun.y), (double)(coordinate_sys_point[0].x - sun.x));  //计算箭头指向角度
    arrow.x = sun.x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = sun.y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    arrow.x = sun.x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = sun.y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion2;
    posion2.x = sun.x;
    posion2.y = sun.y - 10;
    cv::putText(frame, "sun", posion2, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255, 255), 1, 8);

    return 0;
}


int getSunPosition::draw_coordinate_sys(cv::Point2f point_data[11], cv::Mat &frame,bool change_flag ,float x , float y)
{


	float line[4][4];
	cv::Point2f vp[2];
	cv::Point2f coordinate_sys_point[3];
	cv::Point2f sun;
	float coordinate_sys_lenth = frame.cols / 8;

	for (int line_num = 0; line_num < 4; line_num++)
	{
		line[line_num][0] = point_data[line_num * 2].x;
		line[line_num][1] = point_data[line_num * 2].y;
		line[line_num][2] = point_data[line_num * 2 + 1].x;
		line[line_num][3] = point_data[line_num * 2 + 1].y;
	}

	vp[0] = get_vanishing_point(line[0], line[1]);
	vp[1] = get_vanishing_point(line[2], line[3]);


	coordinate_sys_point[0] = point_data[10];
	coordinate_sys_point[1].x = point_data[10].x + (vp[0].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));
	coordinate_sys_point[1].y = point_data[10].y + (vp[0].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[0].x)*(point_data[10].x - vp[0].x) + (point_data[10].y - vp[0].y)*(point_data[10].y - vp[0].y));

	if(y < point_data[10].y)
    {
        sun.x = point_data[10].x + (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
		sun.y = point_data[10].y + (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
    }
    else
	{
		sun.x = point_data[10].x - (x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
		sun.y = point_data[10].y - (y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - x)*(point_data[10].x - x) + (point_data[10].y - y)*(point_data[10].y - y));
	}
	if (change_flag)
	{
		coordinate_sys_point[2].x = point_data[10].x - (vp[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
		coordinate_sys_point[2].y = point_data[10].y - (vp[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
	}
	else
	{
		coordinate_sys_point[2].x = point_data[10].x + (vp[1].x - point_data[10].x)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
		coordinate_sys_point[2].y = point_data[10].y + (vp[1].y - point_data[10].y)*coordinate_sys_lenth / sqrt((point_data[10].x - vp[1].x)*(point_data[10].x - vp[1].x) + (point_data[10].y - vp[1].y)*(point_data[10].y - vp[1].y));
	}



    coordinate_sys_point[1].x = cvRound(coordinate_sys_point[1].x);
    coordinate_sys_point[1].y = cvRound(coordinate_sys_point[1].y);
    coordinate_sys_point[2].x = cvRound(coordinate_sys_point[2].x);
    coordinate_sys_point[2].y = cvRound(coordinate_sys_point[2].y);
    sun.x = cvRound(sun.x);
    sun.y = cvRound(sun.y);

    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[1], cv::Scalar(0, 0, 255), 2, 8);
    cv::line(frame, coordinate_sys_point[0], coordinate_sys_point[2], cv::Scalar(0, 255, 0), 2, 8);
    cv::line(frame, coordinate_sys_point[0], sun, cv::Scalar(0, 255, 255), 2, 8);
//画箭头
   // float PI = 3.14159;
    cv::Point2i arrow;
    float len = 15; //箭头长度
    float arrow_angle = 30;



    float angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[1].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[1].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[1].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[1].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    arrow.x = coordinate_sys_point[1].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[1].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[1], arrow, cv::Scalar(0, 0, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion;
    posion.x = coordinate_sys_point[1].x;
    posion.y = coordinate_sys_point[1].y - 10;
    cv::putText(frame, "x", posion, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - coordinate_sys_point[2].y), (double)(coordinate_sys_point[0].x - coordinate_sys_point[2].x));  //计算箭头指向角度
    arrow.x = coordinate_sys_point[2].x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = coordinate_sys_point[2].y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    arrow.x = coordinate_sys_point[2].x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = coordinate_sys_point[2].y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, coordinate_sys_point[2], arrow, cv::Scalar(0, 255, 0), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion1;
    posion1.x = coordinate_sys_point[2].x;
    posion1.y = coordinate_sys_point[2].y - 10;
    cv::putText(frame, "y", posion1, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0, 255), 1, 8);

    angle = atan2((double)(coordinate_sys_point[0].y - sun.y), (double)(coordinate_sys_point[0].x - sun.x));  //计算箭头指向角度
    arrow.x = sun.x + len * cos(angle + PI *  arrow_angle / 180);   //箭头夹角15度
    arrow.y = sun.y + len * sin(angle + PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    arrow.x = sun.x + len * cos(angle - PI * arrow_angle / 180);
    arrow.y = sun.y + len * sin(angle - PI * arrow_angle / 180);
    cv::line(frame, sun, arrow, cv::Scalar(0, 255, 255), 2, 8);
    //黄雯嘉在此修改，标明x轴
    cv::Point2f posion2;
    posion2.x = sun.x;
    posion2.y = sun.y - 10;
    cv::putText(frame, "sun", posion2, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255, 255), 1, 8);

    return 0;
}



int getSunPosition::sunPositionProcess(int init_num, std::string fileName, cv::Point2f point_data[])
{
     
    cv::VideoCapture capture(fileName); // load video
    std::cout << fileName << std::endl;
    if (!capture.isOpened()) {
        std::cout << "Failed to open video." << std::endl;
        return -1;
    }
    std::ofstream outPut("out.txt");
    if(!outPut.is_open())
    {
        std::cout<<"error occured while creating file out.txt"<<std::endl;
        std::cout<<"program closed..."<<std::endl;
        return -1;
    }

    cv::Mat frame;
    for(int i = 1 ; i <=init_num ; i++)//找到交互初始帧
    {
        capture >> frame;
    }
    cv::Point2f line_point[4][LINE_POINT_NUM];//线上的坐标点
    int out_point_mark[11] , out_line_mark[4][LINE_POINT_NUM];//记录出界的信息点
    for(int i = 0; i<11; i++)//默认所有点没出界
    {
        out_point_mark[i]=1;
    }
    for(int i = 0 ; i < 4 ;i++)
    {
        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
            out_line_mark[i][j]=1;
        }
    }
    extend(point_data);//数据线过短时加以延长
    find_line_point(point_data,line_point);//初始化线上的点

    cv::VideoWriter writer("dst.avi", CV_FOURCC('M', 'J', 'P', 'G'), 8.0, cv::Size(frame.cols, frame.rows));
    bool if_frame_break = false;
    cv::Point2f v1,v2,vi;
    double focal_length;
    cv::Mat R = cv::Mat_<double>(3,3);
    std::vector<cv::Point2d> datatemp;
    datatemp.push_back(point_data[10]);
    datatemp.push_back(point_data[11]);

    double xyLength = getXYlength(datatemp,200);

    for (int frame_num = 0; frame_num < nFrameEnd && !frame.empty(); frame_num++, capture >> frame)
    {
        std::cout << "Frame No."<< frame_num+init_num << std::endl;
        outPut << "#Frame No." <<frame_num+init_num<<std::endl;

        cv::Point2f point[11];
        for(int i=0; i<11; i++)
        {
            point[i] = point_data[i]-cv::Point2f(frame.cols*0.5, frame.rows*0.5);
        }
       // adjustInteraction(point, point);
        v1 = getVanishPoint(point[0], point[1], point[2], point[3]);
        v2 = getVanishPoint(point[4], point[5], point[6], point[7]);
        focal_length = getFocalLength(v1, v2);


        //get_vp_for_R(frame, pointVec, v1, v2, focal_length);

        //adjust_vanishing_point(v1, v2, point[10], v1, v2);

        cv::Point3f Xc, Yc, Zc;

        get_rotation_mat( v1, v2,  focal_length, R);

		//std::cout<<">> no adjust R = "<<R<<std::endl;
        bool flag=adjust_rotation_matrix(R, R);
        //std::cout<<">> adjust R "<<R<<std::endl;

        /**compute translation vector*/
        cv::Point2f origin;
        origin.x = point_data[10].x-0.5*frame.cols;
        origin.y = point_data[10].y-0.5*frame.rows;
        /**bool get_translation_vector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, double T[]);**/
        double T[3] = {0,0,0};
        if(!get_translation_vector(origin, v1, R, focal_length, T) )
        {
            std::cout<<">> error while computing T"<<std::endl;
            continue;
        }
        //std::cout<<"T = "<<std::endl;
       // std::cout<<T[0]<<", "<<T[1]<<", "<<T[2]<<std::endl;



        /**get prarmeter in gluLookAt() function for openGL**/

        cv::Mat t = cv::Mat_<double>(3,1);/**平移矩阵*/
       /* if(std::isnan(T[0]))
            break;*/

        t.at<double>(0,0) = T[0];
        t.at<double>(1,0) = T[1];
        t.at<double>(2,0) = T[2];

//
//        std::cout<<">> R = "<<std::endl;
//        std::cout<< R <<std::endl;
//        std::cout<<">> T = "<<std::endl;
//        std::cout<< t <<std::endl;

        /***********************************  ************************************************************/
        /**                      R, T 均已求出.                                                           *
        *                                                                                                 *
        ***************************************************************************************************/
        double phi, theta;
        double xxxx, yyyy;

        cv::Point2d sunPos;
        getSunPos(frame.rows, frame.cols, point_data, sunPos, theta, phi);
        xxxx = (float)sunPos.x;
        yyyy = (float)sunPos.y;


        std::vector<cv::Point2f> triangle;
        triangle.push_back(point_data[9]);
        triangle.push_back(point_data[10]);
        triangle.push_back(point_data[8]);
		//std::cout<<"R="<<R<<std::endl;
		//std::cout<<"T="<<t<<std::endl;
		//std::cout<<"f="<<focal_length<<std::endl;
        getAngle( R, t,focal_length ,triangle,theta,phi,frame.cols,frame.rows);
        /**将参数坐标系转换成笛卡尔坐标系**/
        double dex, dey, dez;


         xyLength = xyLength;

        //std::cout<<"xyLength:"<<xyLength<<std::endl;

        para_to_decare( xyLength,phi, theta, dex, dey, dez);
        //para_to_decare(10000,0.908, -0.082581, dex, dey, dez);
        std::vector<cv::Point3d> objectPoints;
        std::vector<cv::Point2d> imagePoints(3);
        cv::Point3d Position;

        Position.x = xyLength;
        Position.y = 0;
        Position.z = 0;
        objectPoints.push_back(Position);

        Position.x = 0;
        Position.y = xyLength;
        Position.z = 0;
        objectPoints.push_back(Position);
       /* Position.x = 0;
        Position.y = 0;
        Position.z = 100000;
        objectPoints.push_back(Position);*/
        Position.x = dex;
        Position.y = dey;
        Position.z = dez;
        objectPoints.push_back(Position);

        cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
        cv::Rodrigues(R,rVec);
        cv::Mat K = cv::Mat::zeros(cv::Size(3,3),CV_64F);
        K.at<double>(0,0) = focal_length;
        K.at<double>(1,1) = focal_length;
        K.at<double>(0,2) = frame.cols/2.0;
        K.at<double>(1,2) = frame.rows/2.0;
        K.at<double>(2,2) = 1.0;

        //cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);   // Distortion vector
        cv::Mat distCoeffs = cv::Mat::zeros(cv::Size(5,1),CV_64F);
        cv::projectPoints(objectPoints,rVec,t,K,distCoeffs,imagePoints);
//		world2img(R, t,focal_length,objectPoints[1],imagePoints[1]); //compute the image
//		world2img(R, t,focal_length,objectPoints[0],imagePoints[0]);
//		world2img(R, t,focal_length,objectPoints[2],imagePoints[2]);
		//world2img(R, t,focal_length,objectPoints[3],imagePoints[3]);
      /*  std::cout<<">> x-axis = "<<std::endl;
        std::cout<<imagePoints[0].x<<", "<<imagePoints[0].y<<std::endl;
        std::cout<<">> y-axis = "<<std::endl;
        std::cout<<imagePoints[1].x<<", "<<imagePoints[1].y<<std::endl;
        std::cout<<">> z-axis = "<<std::endl;
        std::cout<<imagePoints[2].x<<", "<<imagePoints[2].y<<std::endl;
        std::cout<<">> sun position = "<<std::endl;
        std::cout<<imagePoints[3].x<<", "<<imagePoints[3].y<<std::endl;*/
//        cv::line(frame,point_data[10],imagePoints[0],cv::Scalar(0,255,0),2,8);
//        cv::line(frame,point_data[10],imagePoints[1],cv::Scalar(255,0,0),2,8);
//        cv::line(frame,point_data[10],imagePoints[2],cv::Scalar(0,255,255),2,8);
     //  cv::line(frame,point_data[10],imagePoints[3],cv::Scalar(0,0,0),2,8);
//        cv::line(frame,point_data[10], cv::Point(592,207), cv::Scalar(34,35,23));
//        cv::line(frame,point_data[10], cv::Point(-2046, -14615), cv::Scalar(255,135,23));
        /**sun direction in world coordinate*/
        cv::Mat Pw = cv::Mat_<double>(4,1);  /**世界坐标系下太阳位置**/
        static cv::Mat P0c = cv::Mat_<double>(4,1);/**初始帧相机坐标*/
        cv::Mat Pc = cv::Mat_<double>(4,1); /**当前帧相机坐标*/
        //cv::Mat t = cv::Mat_<double>(3,1);  /**平移矩阵*/
        static cv::Mat M0 = cv::Mat_<double>(4,4);
        cv::Mat M  = cv::Mat_<double>(4,4);

        Pw.at<double>(0,0) = dex;
        Pw.at<double>(1,0) = dey;
        Pw.at<double>(2,0) = dez;
        Pw.at<double>(3,0) = 1.0;

        t.at<double>(0,0) = T[0];
        t.at<double>(1,0) = T[1];
        t.at<double>(2,0) = T[2];

        if(frame_num == 0) /**第一帧*/
        {
            /**switch to camera coordinate*/
            switch_to_cam_coordinate(Pw, R, t, P0c, M0);
            cv::Mat P = cv::Mat_<double>(3,1);
            P.at<double>(0,0) = P0c.at<double>(0,0);
            P.at<double>(1,0) = P0c.at<double>(1,0);
            P.at<double>(2,0) = P0c.at<double>(2,0);
            outPut<<"# Sun Position in World Coordinate = "<<std::endl;
            outPut<<"("<<Pw.at<double>(0,0)<<", "<<Pw.at<double>(1,0)<<", "<<Pw.at<double>(2,0)<<")"<<std::endl;

            cv::normalize(P,P);
            outPut<<"# Sun position in camera coordinate = "<<std::endl;
            outPut<<"("<<P.at<double>(0,0)<<", "<<P.at<double>(1,0)<<", "<<P.at<double>(2,0)<<")"<<std::endl;
            //outPut<<P<<std::endl;
//            cv::normalize(P,P);
//            outPut  << "#Direction = " <<std::endl;
//            outPut<<P<<std::endl;
        }
        else
        {
            /**switch to camera coordinate*/
            switch_to_cam_coordinate(Pw, R, t, Pc,M);
            cv::Mat P = cv::Mat_<double>(3,1);
            P.at<double>(0,0) = Pc.at<double>(0,0);
            P.at<double>(1,0) = Pc.at<double>(1,0);
            P.at<double>(2,0) = Pc.at<double>(2,0);

//            std::cout<<">> Relation Matrix = "<<std::endl;
//            std::cout<<M*M0.inv()<<std::endl;
//
//            outPut << "#Relation Matrix = " <<std::endl;
//            outPut << M*M0.inv() <<std::endl;

//            std::cout<<">> T = "<<std::endl;
//            std::cout<<t<<std::endl;
            cv::normalize(P,P);
            outPut<<"# Sun Position in camera coordinate = "<<std::endl;
            outPut<<"("<<P.at<double>(0,0)<<", "<<P.at<double>(1,0)<<", "<<P.at<double>(2,0)<<")"<<std::endl;
//            cv::normalize(P, P);
//            std::cout<<P<<std::endl;
        }


        cv::Point2f point_data1[11];
        for (int i = 0; i < 11; i++)
        {
            point_data1[i] = point_data[i];
        }

        int out_point_num = -1;  //记录是否有信息点出界

        tracking(frame,point_data1,out_point_mark,line_point,out_line_mark);//光流跟踪
        fitline_reAllocation(point_data1,line_point);//直线拟合，重分配点

        for(int i=0 ; i < 11 ; i++)//判断是否有点出界
        {
            if(out_point_mark[i])
            {
                if(point_data1[i].x > frame.cols || point_data1[i].x < 0 || point_data1[i].y > frame.rows || point_data1[i].y < 0)
                {
                    out_point_mark[i] = 0 ;
                    out_point_num = i ;
                }
            }

        }
        for(int i = 0 ; i < 4 ;i++)
        {
            for(int j = 0 ; j < LINE_POINT_NUM ;j++)
            {
                if(out_line_mark[i][j])
                {
                    if(line_point[i][j].x > frame.cols ||line_point[i][j].x < 0 || line_point[i][j].y > frame.rows || line_point[i][j].y < 0)
                    {
                        out_line_mark[i][j] = 0 ;
//                           QMessageBox::warning(NULL, "warning", "Line's points is out of the video's boundary", QMessageBox::Yes , QMessageBox::Yes);
//                           if_frame_break = true;
                    }
                }

            }
        }
        if_frame_break = reAllocation(frame, point_data1,out_point_mark,line_point,out_line_mark);


            if (out_point_num >= 8)
            {
                std::cout<<"people's point is out of frame"<<std::endl;
                if_frame_break = true;
            }//人及影子的信息点出界直接判断不能继续运行

            if(if_frame_break == true)
            {
                std::cout<<"some point is out of frame"<<std::endl;
                break ;
            }


        for (int i = 0; i < 11; i++)
        {
            point_data[i] = point_data1[i];
        }

		//draw_coordinate_sys(point_data, frame, xxxx ,yyyy);
		draw_coordinate_sys(point_data,frame,imagePoints ,flag);
	//	draw_coordinate_sys(point_data, frame,flag , xxxx ,yyyy);

        draw_line_2(frame,point_data,out_point_mark,line_point,out_line_mark);

        writer << frame;
        cv::imshow("readvideo",frame);
        //cv::VideoWriter writer("dst.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15.0, cv::Size(frame.cols, frame.rows));
        cv::waitKey(200);
        std::cout<<"process succeeded..."<<std::endl;

    }
    outPut.close();
    gray_prev.release();
    return 0;
}


void getSunPosition::get_rotation_mat(const cv::Point2f v1, const cv::Point2f v2, const double f, cv::Mat &R)
{
   // std::cout<< "in get_rotation_mat :"<< std::endl;
   // std::cout<< "v1 = "<< v1 <<"v2 = "<<v2 <<std::endl;
   // std::cout<<" f = "<<f<<std::endl;
    cv::Mat x_axis = cv::Mat_<double>(3,1);
    cv::Mat y_axis = cv::Mat_<double>(3,1);
    cv::Mat z_axis = cv::Mat_<double>(3,1);
    x_axis.at<double>(0,0) = v1.x;
    x_axis.at<double>(1,0) = v1.y;
    x_axis.at<double>(2,0) = f;
    /**归一化*/
    cv::normalize(x_axis,x_axis);

    y_axis.at<double>(0,0) = v2.x;
    y_axis.at<double>(1,0) = v2.y;
    y_axis.at<double>(2,0) = f;
    cv::normalize(y_axis,y_axis);


    z_axis = x_axis.cross(y_axis);

    R.at<double>(0,0) = x_axis.at<double>(0,0);
    R.at<double>(1,0) = x_axis.at<double>(1,0);
    R.at<double>(2,0) = x_axis.at<double>(2,0);

    R.at<double>(0,1) = y_axis.at<double>(0,0);
    R.at<double>(1,1) = y_axis.at<double>(1,0);
    R.at<double>(2,1) = y_axis.at<double>(2,0);

    R.at<double>(0,2) = z_axis.at<double>(0,0);
    R.at<double>(1,2) = z_axis.at<double>(1,0);
    R.at<double>(2,2) = z_axis.at<double>(2,0);

}

bool getSunPosition::get_translation_vector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, double T[])
{
	if (fabs((double)(origin.x - v1.x))<20 && fabs((double)(origin.y - v1.y))<20)
        return false;

    /**x-axis is a line from origin to v1 in image space*/
    cv::Mat line_x_axis = cv::Mat_<double>(4,1);
    line_x_axis.at<double>(0,0) = origin.x;
    line_x_axis.at<double>(1,0) = origin.y;
    line_x_axis.at<double>(2,0) = v1.x;
    line_x_axis.at<double>(3,0) = v1.y;

    /**direction of x-axis */
    cv::Mat dir_x_axis = cv::Mat_<double>(2,1);
    dir_x_axis.at<double>(0,0) = v1.x - origin.x;
    dir_x_axis.at<double>(1,0) = v1.y - origin.y;
    cv::Mat length = cv::Mat_<double>(1,1);
    length = dir_x_axis.t()*dir_x_axis;
    dir_x_axis.at<double>(0,0) /= sqrt(length.at<double>(0,0));
    dir_x_axis.at<double>(1,0) /= sqrt(length.at<double>(0,0));

    /**p为x轴上一点在图像上的投影坐标,其在世界坐标系内的坐标为（scale*d,0,0）**/
    cv::Point2f p;
    /**p点同原点在图像上的距离**/
    int d=20;
    int scale = d * SCALE;

    p.x = (int)(origin.x + d*dir_x_axis.at<double>(0,0) + 0.5);
    p.y = (int)(origin.y + d*dir_x_axis.at<double>(1,0) + 0.5);

    /**translation vector*/
	if (fabs((double)(p.x - origin.x)) > fabs((double)(p.y - origin.y)))
    {
        T[2] = scale  * (R.at<double>(0,0)*focal_length - R.at<double>(2,0)*p.x)/(double)(p.x - origin.x);
    }
    else
    {
        T[2] = scale  * (R.at<double>(1,0)*focal_length - R.at<double>(2,0)*p.y)/(double)(p.y - origin.y);
    }

    if(T[2] < 0)
    {
        T[2] =  -1 * T[2];
    }
    if(T[2] < focal_length)
    {
        return false;
    }
    T[0] = origin.x * T[2] / focal_length;
    T[1] = origin.y * T[2] / focal_length;
    return true;
}

bool getSunPosition::adjust_rotation_matrix(cv::Mat src_R, cv::Mat &dst_R)
{
    bool flag=false;
    cv::Mat x_axis = cv::Mat_<double>(3,1);
    cv::Mat y_axis = cv::Mat_<double>(3,1);
    cv::Mat z_axis = cv::Mat_<double>(3,1);
    x_axis.at<double>(0,0) = src_R.at<double>(0,0);
    x_axis.at<double>(1,0) = src_R.at<double>(1,0);
    x_axis.at<double>(2,0) = src_R.at<double>(2,0);

    y_axis.at<double>(0,0) = src_R.at<double>(0,1);
    y_axis.at<double>(1,0) = src_R.at<double>(1,1);
    y_axis.at<double>(2,0) = src_R.at<double>(2,1);

    z_axis.at<double>(0,0) = src_R.at<double>(0,2);
    z_axis.at<double>(1,0) = src_R.at<double>(1,2);
    z_axis.at<double>(2,0) = src_R.at<double>(2,2);

    int biggest_index=1;
    double biggest_val = fabs((double)x_axis.at<double>(1,0));
	if (fabs((double)y_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 2;
		biggest_val = fabs((double)y_axis.at<double>(1, 0));
    }
	if (fabs((double)z_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 3;
		biggest_val = fabs((double)z_axis.at<double>(1, 0));
    }
    int symble;
    cv::Mat v1 = cv::Mat_<double>(3,1);
    cv::Mat v2 = cv::Mat_<double>(3,1);
    cv::Mat v3 = cv::Mat_<double>(3,1);
   // std::cout<<">> case: "<<std::endl;
   // std::cout<<biggest_index<<std::endl;
    switch(biggest_index)
    {
    case 1:
        if(x_axis.at<double>(1,0) > 0) /**竖直向下*/
            symble = -1;
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * x_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * x_axis.at<double>(1,0);
        z_axis.at<double>(2,0) = symble * x_axis.at<double>(2,0);
        v1.at<double>(0,0) = y_axis.at<double>(0,0);
        v1.at<double>(1,0) = y_axis.at<double>(1,0);
        v1.at<double>(2,0) = y_axis.at<double>(2,0);

        v2.at<double>(0,0) = z_axis.at<double>(0,0);
        v2.at<double>(1,0) = z_axis.at<double>(1,0);
        v2.at<double>(2,0) = z_axis.at<double>(2,0);
        /**cross pruduct v1 * v2 */
        v3 = v1.cross(v2);
        //v3 = v2.cross(v1);
        x_axis = v3;
        break;
    case 2:
        if(y_axis.at<double>(1,0) > 0)
        {
            symble = -1;
        }
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * y_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(1,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(2,0);
        v1.at<double>(0,0) = z_axis.at<double>(0,0);
        v1.at<double>(1,0) = z_axis.at<double>(1,0);
        v1.at<double>(2,0) = z_axis.at<double>(2,0);

        v2.at<double>(0,0) = x_axis.at<double>(0,0);
        v2.at<double>(1,0) = x_axis.at<double>(1,0);
        v2.at<double>(2,0) = x_axis.at<double>(2,0);

        v3 = v1.cross(v2);
        //v3 = v2.cross(v1);

        y_axis = v3.clone();
        break;
    case 3:
        if(z_axis.at<double>(1,0) > 0)
        {
            //std::cout<<"in case 3"<<std::endl;
//            symble = -1;
            z_axis.at<double>(0,0) = -1 * z_axis.at<double>(0,0);
            z_axis.at<double>(1,0) = -1 * z_axis.at<double>(1,0);
            z_axis.at<double>(2,0) = -1 * z_axis.at<double>(2,0);
            v1.at<double>(0,0) = z_axis.at<double>(0,0);
            v1.at<double>(1,0) = z_axis.at<double>(1,0);
            v1.at<double>(2,0) = z_axis.at<double>(2,0);
            v2.at<double>(0,0) = x_axis.at<double>(0,0);
            v2.at<double>(1,0) = x_axis.at<double>(1,0);
            v2.at<double>(2,0) = x_axis.at<double>(2,0);
            v3 = v1.cross(v2);
            //v3 = v2.cross(v1);
            y_axis = v3.clone();
            flag = true;
        }

        break;
    default:
        std::cout<<">> case default"<<std::endl;
        break;

    }

    dst_R.at<double>(0,0) = x_axis.at<double>(0,0);
    dst_R.at<double>(1,0) = x_axis.at<double>(1,0);
    dst_R.at<double>(2,0) = x_axis.at<double>(2,0);

    dst_R.at<double>(0,1) = y_axis.at<double>(0,0);
    dst_R.at<double>(1,1) = y_axis.at<double>(1,0);
    dst_R.at<double>(2,1) = y_axis.at<double>(2,0);

    dst_R.at<double>(0,2) = z_axis.at<double>(0,0);
    dst_R.at<double>(1,2) = z_axis.at<double>(1,0);
    dst_R.at<double>(2,2) = z_axis.at<double>(2,0);
    return flag;
}

void getSunPosition::para_to_decare(const double r, const double phi, const double theta, double &x, double &y, double &z)
{
    //!!!!!我改的这里，你看看，改了后就不对了，感觉还是哪里没有对，邢老师的求出来的角度和我们求出来的不一样
    x = r*cos(phi)*cos(theta);
    y = r*cos(phi)*sin(theta);
    z = r* sin(phi);
    return ;
}

void getSunPosition::switch_to_cam_coordinate(const cv::Mat _Pw, const cv::Mat _R, const cv::Mat _t, cv::Mat &_Pc, cv::Mat &_M)
{
    assert(_Pw.rows == 4 && _Pw.cols == 1&& _R.rows == 3 && _R.cols == 3 && _t.rows == 3 && _t.cols == 1 && _Pc.rows == 4 && _Pc.cols == 1
            && _M.rows == 4 && _M.cols == 4);
    cv::Mat Pw = cv::Mat_<double>(4,1);
    cv::Mat R  = cv::Mat_<double>(4,4);
    cv::Mat T  = cv::Mat_<double>(4,4);
    cv::Mat Pc = cv::Mat_<double>(4,1);

    for(int i = 0; i<3; i++)
        Pw.at<double>(i,0) = _Pw.at<double>(i,0);
    Pw.at<double>(3,0) = 1;

    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
            R.at<double>(i,j) = _R.at<double>(i,j);
    }
    for(int i=0; i<3; i++)
        R.at<double>(i,3) = 0;
    for(int i=0; i<3; i++)
        R.at<double>(3,i) = 0;
    R.at<double>(3,3) = 1;

    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4;j++)
        {
            if(i == j)
                T.at<double>(i,j) = 1;
            else
                T.at<double>(i,j) = 0;
        }
    }
    for(int i=0; i<3; i++)
    {
        T.at<double>(i,3) = (-1) * _t.at<double>(i,0);
    }
    _M = R*T;
    _Pc = _M*Pw;

}

void getSunPosition::central_symmetry( cv::Point2f src, cv::Point2f origin, cv::Point2f& dst )
{
    dst = 2*origin - src;
}
void getSunPosition::XgyLine::setLine(cv::Point2f p1,cv::Point2f p2)
{
	x1.x=p1.x;
	x1.y=p1.y;
	x2.x=p2.x;
	x2.y=p2.y;
}
cv::Point2f getSunPosition::XgyLine::getDirection()
{
	cv::Point2f v;
	v.x=x2.x-x1.x;
	v.y=x2.y-x1.y;
	double l=sqrt(v.x*v.x+v.y*v.y);
	if(l!=0)
	{
		v.x/=l;
		v.y/=l;
	}
	return v;
}

void getSunPosition::XgyPlane::setPlane(double a, double b, double c, double d)
{
	this->a=a;
	this->b=b;
	this->c=c;
	this->d=d;
}
void getSunPosition::setRT(cv::Mat R,cv::Mat T)
{
	rotateMat=R;
	tranVector=T;
}
void getSunPosition::setWandH(double w,double h)
{
	width = w;
	height = h;
}
bool getSunPosition::img2world(cv::Point2f iP,XgyPlane pl,cv::Point3f &wP)
{

	cv::Mat temp_cP(3,1,CV_64F);
	temp_cP.at<double>(0,0)=(iP.x-0.5*width)/focalLength;
	temp_cP.at<double>(1,0)=(iP.y-0.5*height)/focalLength;
	temp_cP.at<double>(2,0)=1;



	cv::Mat v1(3,1,CV_64F);
	cv::Mat invRot(3,3,CV_64F);
	invRot=rotateMat.t();
	v1=invRot*temp_cP;
	cv::Point3f t1;
	t1.x=v1.at<double>(0,0);
	t1.y=v1.at<double>(1,0);
	t1.z=v1.at<double>(2,0);

	cv::Mat v2(3,1,CV_64F);
	v2=invRot*tranVector;
	cv::Point3f t2;
    t2.x=v2.at<double>(0,0);
	t2.y=v2.at<double>(1,0);
	t2.z=v2.at<double>(2,0);


	double temp=pl.a*t1.x+pl.b*t1.y+pl.c*t1.z;
	double cPz;
	if(temp!=0)
		cPz=(pl.a*t2.x+pl.b*t2.y+pl.c*t2.z-pl.d)/temp;
	else
	{
		return false;
	}

	if(cPz<0)
	{
		return false;
	}

	wP.x=cPz*t1.x-t2.x;
	wP.y=cPz*t1.y-t2.y;
	wP.z=cPz*t1.z-t2.z;



	return true;
}
bool getSunPosition::getPlanefromLine(XgyLine l, XgyPlane &pl)
{
	XgyPlane floor;
	floor.setPlane(0,0,1,0);

	//计算线段两端点的3维位置
	cv::Point3f wP1;
	if(!(img2world(l.x1,floor,wP1)))
	{
		//cout<<"Can not create plane!"<<endl;
		return false;
	}

	cv::Point3f wP2;
	if(!(img2world(l.x2,floor,wP2)))
	{
		//cout<<"Can not create plane!"<<endl;
		return false;
	}

	XgyVector3D tempNormal;	//平面法向垂直于直线wP1wP2，且平行于z=0
	tempNormal.x=wP1.y-wP2.y;
	tempNormal.y=wP2.x-wP1.x;
	tempNormal.z=0;
	normalizeVector3D(tempNormal);

	XgyVector3D view;
	getcamPose(camPos);
	view=setVector3D(camPos,wP1);
	normalizeVector3D(view);

	XgyVector3D normal;
	if(dotProduct3D(tempNormal,view)>0)	//若法向与视线同向
		scalarMul(-1.0,tempNormal,normal);	//反向
	else
		scalarMul(1.0,tempNormal,normal);

	double d=-1*(normal.x*wP1.x+normal.y*wP1.y+normal.z*wP1.z);
	pl.setPlane(normal.x,normal.y,normal.z,d);

	return true;
}
double getSunPosition::getVectorLength3D(XgyVector3D v)
{
	double l;
	l=0;
	l=v.x*v.x+v.y*v.y+v.z*v.z;
	l=sqrt(l);
	return l;
}
double getSunPosition::normalizeVector3D(XgyVector3D &v)
{
	double l;
	l=getVectorLength3D(v);
	if(l==0)
		return 0;
	v.x/=l;
	v.y/=l;
	v.z/=l;
	return l;
}
XgyVector3D getSunPosition::setVector3D(cv::Point3f p1,cv::Point3f p2)
{
	XgyVector3D v;
	v.x=p2.x-p1.x;
	v.y=p2.y-p1.y;
	v.z=p2.z-p1.z;
	return v;
}
void getSunPosition::getcamPose(cv::Point3f & campose)
{
	cv::Mat invR(3,3,CV_64F);
	invR=rotateMat.t();
	cv::Mat camPos(3,1,CV_64F);	//正交阵求逆
	camPos=invR*tranVector;

   // std::cout<<"invR="<<invR<<std::endl;
	//std::cout<<"camPos"<<camPos<<std::endl;
	campose.x=-1*camPos.at<double>(0,0);
	campose.y=-1*camPos.at<double>(1,0);
	campose.z=-1*camPos.at<double>(2,0);

}
double getSunPosition::dotProduct3D(XgyVector3D v1,XgyVector3D v2)
{
	double r=v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
	return r;
}
void getSunPosition::scalarMul(double num,XgyVector3D src,XgyVector3D &tar)
{
	tar.x=num*src.x;
	tar.y=num*src.y;
	tar.z=num*src.z;
}
bool getSunPosition::setSunAngleEx(std::vector<XgyLine> lineArray)
{

    if(lineArray.size()==0)
		return false;

	XgyPlane P1;		//确定方位角
	P1.setPlane(0,0,1,0);
	cv::Point3f startP,endP;		//方位角指示线的起点和终点
	if(!img2world(lineArray[0].x2,P1,startP))
		return false;
	if(!img2world(lineArray[0].x1,P1,endP))
		return false;

	XgyVector3D v1;
	v1=setVector3D(startP,endP);
	normalizeVector3D(v1);
	double beijian=PI/180*90;
	if(v1.y>=0)
		{sunAzimuth=acos(v1.x);
	}
	else
		{sunAzimuth=-1.0*acos(v1.x);

	}
	if(lineArray.size()==1)
		return false;
	else
	{
		XgyPlane P2;
		getPlanefromLine(lineArray[0], P2);
		cv::Point3f endZP;			//高度角指示线终点
		if(!img2world(lineArray[1].x2,P2,endZP))
			return false;
		XgyVector3D v2;
		v2=setVector3D(startP,endZP);
		normalizeVector3D(v2);
		if(v2.z<0)
			return false;
		else
		{
			sunZenith=acos(v2.z);//sunZenith为太阳方向与z轴的夹角
			sunAltitude=beijian- sunZenith;//sunAltitude为太阳方向与y轴的夹角
		}
	}
	return true;
}
void getSunPosition::printAngle()
{
	std::cout<<"sunAzimuth="<<sunAzimuth<<std::endl;
	std::cout<<"sunAltitude="<<sunAltitude<<std::endl;
}
void getSunPosition::setVecA(std::vector<XgyLine>& lineArray,std::vector<cv::Point2f> data)
{
	lineArray[0].x1=data[1];
	lineArray[0].x2=data[0];
	lineArray[1].x1=data[1];
	lineArray[1].x2=data[2];
}
void getSunPosition::setFlocallength(double f)
{
    focalLength=f;
}
void getSunPosition::getAngle(cv::Mat R,cv::Mat T,double f,std::vector<cv::Point2f> data,double&_sunAzimuth,double& _sunAltitude,double w,double h)
{
	setWandH(w,h);
    setFlocallength(f);
	setRT(R,T);
	std::vector<XgyLine> lineArray(2);
	setVecA(lineArray,data);
	setSunAngleEx(lineArray);
	_sunAzimuth=sunAzimuth;
	_sunAltitude=sunAltitude;
	//printAngle();
}
bool getSunPosition:: world2img(cv::Mat rotateMat,cv::Mat tranVector,double focalLength,cv::Point3d wP, cv::Point2d &iP)
{
	//CvMat *originalPos=cvCreateMat(3,1,CV_32FC1);
	cv::Mat originalPos(3,1,CV_64F);
	originalPos.at<double>(0,0)=wP.x;
	originalPos.at<double>(1,0)=wP.y;
	originalPos.at<double>(2,0)=wP.z;
	//cvmSet(originalPos,0,0,wP.x);
	//cvmSet(originalPos,1,0,wP.y);
	//cvmSet(originalPos,2,0,wP.z);
	cv::Mat rotationPos(3,1,CV_64F);
	rotationPos=rotateMat*originalPos;
	//CvMat *rotationPos=cvCreateMat(3,1,CV_32FC1);	//旋转变换
	cv::Mat translationPos(3,1,CV_64F);
	translationPos=rotationPos+tranVector;

	//CvMat *translationPos=cvCreateMat(3,1,CV_32FC1);	//位移变换
	//std::cout<<"test img2world:"<<std::endl;
	//std::cout<<translationPos.at<double>(0,0)<<", "<<translationPos.at<double>(1,0)<<", "<<translationPos.at<double>(2,0)<<std::endl;

	cv::Point2f	imgPos;	//透视变换
	iP.x=0.5*width+translationPos.at<double>(0,0)*focalLength/translationPos.at<double>(2,0);
	iP.y=0.5*height+translationPos.at<double>(1,0)*focalLength/translationPos.at<double>(2,0);

	//iP.x=0.5*width+cvmGet(translationPos,0,0)*focalLength/cvmGet(translationPos,2,0);
	//iP.y=0.5*height+cvmGet(translationPos,1,0)*focalLength/cvmGet(translationPos,2,0);

	//if(cvmGet(translationPos,2,0)<=focalLength)
	if(translationPos.at<double>(2,0)<=focalLength)
	{
		//cvReleaseMat(&originalPos);
		//cvReleaseMat(&translationPos);
		//cvReleaseMat(&rotationPos);
		return false;
	}
	//cvReleaseMat(&originalPos);
	//cvReleaseMat(&translationPos);
	//cvReleaseMat(&rotationPos);


	return true;
}
double getSunPosition::getXYlength(std::vector<cv::Point2d> data,double scale)
	{
		double xylength;
		double temp = 0;
		scale = 200;
		temp+=(data[1].x-data[0].x)*(data[1].x-data[0].x);
		temp+=(data[1].y-data[0].y)*(data[1].y-data[0].y);
		xylength=sqrt(temp)*scale;
		return xylength;
	}
double getSunPosition::getFocalLength(cv::Point2f vx, cv::Point2f vy)
{
    cv::Point2f v1 = vx;
    cv::Point2f v2 = vy;

    return sqrt( -1*(v1.x * v2.x + v1.y * v2.y) );
}
void getSunPosition::getIntrinsicParameter(double focal_length, cv::Point2f principal_point, cv::Mat & K)
{
    K = cv::Mat_<double>(3,3);
    K.at<double>(0,0) = focal_length;
    K.at<double>(0,1) = 0;
    K.at<double>(0,2) = principal_point.x;
    K.at<double>(1,0) = 0;
    K.at<double>(1,1) = focal_length;
    K.at<double>(1,2) = principal_point.y;
    K.at<double>(2,0) = 0;
    K.at<double>(2,1) = 0;
    K.at<double>(2,2) = 1;
}
void getSunPosition::getRotateMat(cv::Point2f vx, cv::Point2f vy, double focal_length, cv::Mat &R)
{
    cv::Mat r1 = cv::Mat_<double>(3,1);
    cv::Mat r2 = cv::Mat_<double>(3,1);
    cv::Mat r3 = cv::Mat_<double>(3,1);

    r1.at<double>(0,0) = vx.x;
    r1.at<double>(1,0) = vx.y;
    r1.at<double>(2,0) = focal_length;
    cv::normalize(r1,r1);

    r2.at<double>(0,0) = vy.x;
    r2.at<double>(1,0) = vy.y;
    r2.at<double>(2,0) = focal_length;
    cv::normalize(r2,r2);

    r3 = r1.cross(r2);
    R = cv::Mat_<double>(3,3);
    R.at<double>(0,0) = r1.at<double>(0,0);
    R.at<double>(1,0) = r1.at<double>(1,0);
    R.at<double>(2,0) = r1.at<double>(2,0);

    R.at<double>(0,1) = r2.at<double>(0,0);
    R.at<double>(1,1) = r2.at<double>(1,0);
    R.at<double>(2,1) = r2.at<double>(2,0);

    R.at<double>(0,2) = r3.at<double>(0,0);
    R.at<double>(1,2) = r3.at<double>(1,0);
    R.at<double>(2,2) = r3.at<double>(2,0);
    adjustRotateMat(R,R);
}

void getSunPosition::adjustRotateMat(cv::Mat srcR, cv::Mat &dstR)
{
    cv::Mat x_axis = cv::Mat_<double>(3,1);
    cv::Mat y_axis = cv::Mat_<double>(3,1);
    cv::Mat z_axis = cv::Mat_<double>(3,1);
    x_axis.at<double>(0,0) = srcR.at<double>(0,0);
    x_axis.at<double>(1,0) = srcR.at<double>(1,0);
    x_axis.at<double>(2,0) = srcR.at<double>(2,0);

    y_axis.at<double>(0,0) = srcR.at<double>(0,1);
    y_axis.at<double>(1,0) = srcR.at<double>(1,1);
    y_axis.at<double>(2,0) = srcR.at<double>(2,1);

    z_axis.at<double>(0,0) = srcR.at<double>(0,2);
    z_axis.at<double>(1,0) = srcR.at<double>(1,2);
    z_axis.at<double>(2,0) = srcR.at<double>(2,2);

    int biggest_index=1;
    double biggest_val = fabs((double)x_axis.at<double>(1,0));
	if (fabs((double)y_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 2;
		biggest_val = fabs((double)y_axis.at<double>(1, 0));
    }
	if (fabs((double)z_axis.at<double>(1, 0)) > biggest_val)
    {
        biggest_index = 3;
		biggest_val = fabs((double)z_axis.at<double>(1, 0));
    }
    int symble;
    cv::Mat v1 = cv::Mat_<double>(3,1);
    cv::Mat v2 = cv::Mat_<double>(3,1);
    cv::Mat v3 = cv::Mat_<double>(3,1);
    switch(biggest_index)
    {
    case 1:
        if(x_axis.at<double>(1,0) > 0) /**竖直向下*/
            symble = -1;
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * x_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * x_axis.at<double>(1,0);
        z_axis.at<double>(2,0) = symble * x_axis.at<double>(2,0);
        v1.at<double>(0,0) = y_axis.at<double>(0,0);
        v1.at<double>(1,0) = y_axis.at<double>(1,0);
        v1.at<double>(2,0) = y_axis.at<double>(2,0);

        v2.at<double>(0,0) = z_axis.at<double>(0,0);
        v2.at<double>(1,0) = z_axis.at<double>(1,0);
        v2.at<double>(2,0) = z_axis.at<double>(2,0);
        /**cross pruduct v1 * v2 */
        v3 = v1.cross(v2);
        //v3 = v2.cross(v1);
        x_axis = v3;
        break;
    case 2:
        if(y_axis.at<double>(1,0) > 0)
        {
            symble = -1;
        }
        else
        {
            symble = 1;
        }
        z_axis.at<double>(0,0) = symble * y_axis.at<double>(0,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(1,0);
        z_axis.at<double>(1,0) = symble * y_axis.at<double>(2,0);
        v1.at<double>(0,0) = z_axis.at<double>(0,0);
        v1.at<double>(1,0) = z_axis.at<double>(1,0);
        v1.at<double>(2,0) = z_axis.at<double>(2,0);

        v2.at<double>(0,0) = x_axis.at<double>(0,0);
        v2.at<double>(1,0) = x_axis.at<double>(1,0);
        v2.at<double>(2,0) = x_axis.at<double>(2,0);

        v3 = v1.cross(v2);
        y_axis = v3.clone();
        break;
    case 3:
        if(z_axis.at<double>(1,0) > 0)
        {
            z_axis.at<double>(0,0) = -1 * z_axis.at<double>(0,0);
            z_axis.at<double>(1,0) = -1 * z_axis.at<double>(1,0);
            z_axis.at<double>(2,0) = -1 * z_axis.at<double>(2,0);
            v1.at<double>(0,0) = z_axis.at<double>(0,0);
            v1.at<double>(1,0) = z_axis.at<double>(1,0);
            v1.at<double>(2,0) = z_axis.at<double>(2,0);
            v2.at<double>(0,0) = x_axis.at<double>(0,0);
            v2.at<double>(1,0) = x_axis.at<double>(1,0);
            v2.at<double>(2,0) = x_axis.at<double>(2,0);
            v3 = v1.cross(v2);
            y_axis = v3.clone();

        }

        break;
    default:
        std::cout<<">> case default"<<std::endl;
        break;

    }

    dstR.at<double>(0,0) = x_axis.at<double>(0,0);
    dstR.at<double>(1,0) = x_axis.at<double>(1,0);
    dstR.at<double>(2,0) = x_axis.at<double>(2,0);

    dstR.at<double>(0,1) = y_axis.at<double>(0,0);
    dstR.at<double>(1,1) = y_axis.at<double>(1,0);
    dstR.at<double>(2,1) = y_axis.at<double>(2,0);

    dstR.at<double>(0,2) = z_axis.at<double>(0,0);
    dstR.at<double>(1,2) = z_axis.at<double>(1,0);
    dstR.at<double>(2,2) = z_axis.at<double>(2,0);
}
void getSunPosition::getSunPos(const int H, const int W, const cv::Point2f data_point[11],cv::Point2d &sunPos, double &theta, double &phi)
{
    cv::Point2f point[11];
    for(int i=0; i<11; i++)
    {
        point[i] = data_point[i];
    }
    /**vx, vy 分别代表x方向,y方向的消失点*/
    cv::Point2f vx;
    cv::Point2f vy;
    cv::Point2f vz;/**z 方向的消失点*/
    cv::Point2f v_prime;
    cv::Point2d principal_point; /**相机主点**/
    cv::Mat K = cv::Mat_<double>(3,3); /**相机内参K**/
    cv::Mat w = cv::Mat_<double>(3,3); /** w = K.inv().t()*K.inv() */
    vx = get_line_intersection(point[0], point[1], point[2], point[3]);
    vy = get_line_intersection(point[4], point[5], point[6], point[7]);
    double f; /**焦距**/

    /**调整消失点，使得z-axis朝上**/
   // adjust_vanishing_point(vx, vy, point[10],vx, vy);

    /***失点vx, vy 所在直线与头的阴影点，脚点所在直线的交点为 v', 用v_prime表示*/
    v_prime = get_line_intersection(vx, vy, point[9], point[10]);

    /**求焦距f**/
    cv::Point2d v1, v2;
//    v1.x = vx.x - (double)W/2;
//    v1.y = vx.y - (double)H/2;
//    v2.x = vy.x - (double)W/2;
//    v2.y = vy.y - (double)H/2;

    v1.x = vx.x ;
    v1.y = vx.y ;
    v2.x = vy.x ;
    v2.y = vy.y ;
    f = get_focal_length2(v1, v2);
   // std::cout<<">> focal length f = "<<  f  <<std::endl;
    /**求相机主点**/
    principal_point.x = (double)W/2;
    principal_point.y = (double)H/2;
    /**求K**/
    K.at<double>(0,0) = f;
    K.at<double>(0,1) = 0;
    K.at<double>(0,2) = principal_point.x;
    K.at<double>(1,0) = 0;
    K.at<double>(1,1) = f;
    K.at<double>(1,2) = principal_point.y;
    K.at<double>(2,0) = 0;
    K.at<double>(2,1) = 0;
    K.at<double>(2,2) = 1;
    /**求w**/
    w = K.t().inv() * K.inv();
    //std::cout<<"w = "<<w<<std::endl;
    /**求z-axis消失点**/
    cv::Mat mx = (cv::Mat_<double>(3,1) << vx.x, vx.y, 1); /**将vx, vy变为齐次的*/
    cv::Mat my = (cv::Mat_<double>(3,1) << vy.x, vy.y, 1);
    cv::Mat xw = cv::Mat_<double>(1,3);
    cv::Mat yw = cv::Mat_<double>(1,3);
    xw = mx.t() * w;

    yw = my.t() * w;
    cv::Mat B = (cv::Mat_<double>(2,1) << -xw.at<double>(0,2), -yw.at<double>(0,2));
    cv::Mat A = (cv::Mat_<double>(2,2) << xw.at<double>(0,0), xw.at<double>(0,1), yw.at<double>(0,0), yw.at<double>(0,1));
    cv::Mat mz = cv::Mat_<double>(2,2);
   // std::cout<<"A = "<<A<<std::endl;
    mz = A.inv() * B;
   //mz = (A.t()*A).inv()*A.t()*B;
   /* std::cout<<"A.inv="<<A.inv()<<std::endl;
    std::cout<<"detA"<<cv::determinant(A)<<std::endl;
    std::cout<<"B="<<B<<std::endl;*/
    vz.x = mz.at<double>(0,0);
    vz.y = mz.at<double>(1,0);
    cv::Mat tmz = (cv::Mat_<double>(3,1) << vz.x, vz.y, 1);
    /*std::cout<<">> vy' * w * vx  = "<<std::endl;
    std::cout<< mx.t() * w * my<<std::endl;
    std::cout<<">> vy' * w * vz  = "<<std::endl;
    std::cout<< my.t() * w * tmz<<std::endl;
    std::cout<<">> vx' * w * vz  = "<<std::endl;
    std::cout<< mx.t() * w * tmz<<std::endl;*/

    /**v_prime与z轴消失点vz所在直线与头的阴影点，头点所在直线交点即太阳位置**/
    sunPos = get_line_intersection(v_prime, vz, point[8], point[9]);

    /**根据论文公式求theta和phi**/
    get_theta_and_phi(w, sunPos, v_prime, vx, vy, vz, theta, phi);

}
void getSunPosition::get_theta_and_phi(const cv::Mat w, const cv::Point2d sunPos, const cv::Point2d v_prime, \
                       const cv::Point2d vx, const cv::Point2d vy, const cv::Point2d vz,  double &theta, double &phi)
{
    cv::Point2d vs = get_line_intersection(vx, vy, vz, sunPos);
    cv::Mat mv_prime = (cv::Mat_<double>(3,1) << v_prime.x, v_prime.y, 1);
    cv::Mat mvx = (cv::Mat_<double>(3,1) << vx.x, vx.y, 1);
    cv::Mat mvy = (cv::Mat_<double>(3,1) << vy.x, vy.y, 1);
    cv::Mat mvz = (cv::Mat_<double>(3,1) << vz.x, vz.y, 1);
    cv::Mat mv = (cv::Mat_<double>(3,1) << sunPos.x, sunPos.y, 1);
    cv::Mat mvs = (cv::Mat_<double>(3,1) << vs.x, vs.y, 1);
   // std::cout<<">> mvz :"<<mvz<<std::endl;
   // std::cout<<">> mvs :"<<mvs<<std::endl;
    /**求theta*/
    cv::Mat tp = mvx.t()*w*mvs;
    cv::Mat dwn1 = mvs.t()*w*mvs;
    cv::Mat dwn2 = mvx.t()*w*mvx;
    double cosTheta = tp.at<double>(0,0)/(sqrt(dwn1.at<double>(0,0)) * sqrt(dwn2.at<double>(0,0)));
    theta = acos(cosTheta);
    /**求phi**/
    tp = mvz.t()*w*mv;
    dwn1 = mv.t()*w*mv;
    dwn2 = mvz.t()*w*mvz;
    double cosPhi = tp.at<double>(0,0)/(sqrt(dwn1.at<double>(0,0)) * sqrt(dwn2.at<double>(0,0)));
    phi = acos(cosPhi);
    if(phi > PI/2)
        phi = PI-phi;
 //   std::cout<<"theta = "<<(theta/3.14)*180<<" phi = "<<(phi/3.14)*180<<std::endl;
}
bool getSunPosition::getTranslationVector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, cv::Mat &T)
{

    if (fabs((double)(origin.x - v1.x))<20 && fabs((double)(origin.y - v1.y))<20)
        return false;

    /**x-axis is a line from origin to v1 in image space*/
    cv::Mat line_x_axis = cv::Mat_<double>(4,1);
    line_x_axis.at<double>(0,0) = origin.x;
    line_x_axis.at<double>(1,0) = origin.y;
    line_x_axis.at<double>(2,0) = v1.x;
    line_x_axis.at<double>(3,0) = v1.y;

    /**direction of x-axis */
    cv::Mat dir_x_axis = cv::Mat_<double>(2,1);
    dir_x_axis.at<double>(0,0) = v1.x - origin.x;
    dir_x_axis.at<double>(1,0) = v1.y - origin.y;
    cv::Mat length = cv::Mat_<double>(1,1);
    length = dir_x_axis.t()*dir_x_axis;
    dir_x_axis.at<double>(0,0) /= sqrt(length.at<double>(0,0));
    dir_x_axis.at<double>(1,0) /= sqrt(length.at<double>(0,0));

    /**p为x轴上一点在图像上的投影坐标,其在世界坐标系内的坐标为（scale*d,0,0）**/
    cv::Point2f p;
    /**p点同原点在图像上的距离**/
    int d=20;
    int scale = d * SCALE;

    p.x = (int)(origin.x + d*dir_x_axis.at<double>(0,0) + 0.5);
    p.y = (int)(origin.y + d*dir_x_axis.at<double>(1,0) + 0.5);

    /**translation vector*/
	if (fabs((double)(p.x - origin.x)) > fabs((double)(p.y - origin.y)))
    {
        T.at<double>(2,0) = scale  * (R.at<double>(0,0)*focal_length - R.at<double>(2,0)*p.x)/(double)(p.x - origin.x);
    }
    else
    {
        T.at<double>(2,0) = scale  * (R.at<double>(1,0)*focal_length - R.at<double>(2,0)*p.y)/(double)(p.y - origin.y);
    }

    if(T.at<double>(2,0) < 0)
    {
        T.at<double>(2,0) =  -1 * T.at<double>(2,0);
    }
    if(T.at<double>(2,0) < focal_length)
    {
        return false;
    }
    T.at<double>(0,0) = origin.x * T.at<double>(2,0) / focal_length;
    T.at<double>(1,0) = origin.y * T.at<double>(2,0) / focal_length;
    return true;
}
cv::Point2f getSunPosition::getVanishPoint(cv::Point2f v0, cv::Point2f v1, cv::Point2f v2, cv::Point2f v3)
{
    cv::Point2f vp;
    vp = getLineIntersection(v0, v1, v2, v3);
    return vp;
}
cv::Point2f getSunPosition::getLineIntersection(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
{
    cv::Point2f vp;
    float x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = p0.x;   y0 = p0.y;
    x1 = p1.x;   y1 = p1.y;
    x2 = p2.x;   y2 = p2.y;
    x3 = p3.x;   y3 = p3.y;
    vp.y = ( (y0-y1)*(y3-y2)*x0 + (y3-y2)*(x1-x0)*y0 + (y1-y0)*(y3-y2)*x2 + (x2-x3)*(y1-y0)*y2 ) / ( (x1-x0)*(y3-y2) + (y0-y1)*(x3-x2) );
    vp.x = x2 + (x3-x2)*(vp.y-y2) / (y3-y2);
    return vp;
}
bool getSunPosition::adjust_vanishing_point(const cv::Point2f  _v1, const cv::Point2f _v2, const cv::Point2f origin, cv::Point2f &v1, cv::Point2f &v2)
{
    cv::Point2f tv1, tv2;
    tv1 = _v1;
    tv2 = _v2;
    v1 = _v1;
    v2 = _v2;

    cv::Mat lvector[3];//坐标轴方向向量
	lvector[0] = cv::Mat_<double>(2, 1);
	lvector[1] = cv::Mat_<double>(2, 1);
	lvector[2] = cv::Mat_<double>(2, 1);

	lvector[0].at<double>(0,0) = tv1.x - origin.x;
	lvector[0].at<double>(1,0) = tv1.y - origin.y;
	lvector[1].at<double>(0,0) = tv2.x - origin.x;
	lvector[1].at<double>(1,0) = tv2.y - origin.y;

	cv::normalize(lvector[0],lvector[0]);
	cv::normalize(lvector[1],lvector[1]);

	double angle_cos = (lvector[0].at<double>(0, 0) * lvector[1].at<double>(0, 0) + \
                     lvector[0].at<double>(1, 0) * lvector[1].at<double>(1, 0)) / (sqrt(lvector[0].at<double>(0, 0)*lvector[0].at<double>(0, 0) + \
                    lvector[0].at<double>(1, 0)*lvector[0].at<double>(1, 0))*sqrt(lvector[1].at<double>(0, 0)*lvector[1].at<double>(0, 0) + \
                                                            lvector[1].at<double>(1, 0)*lvector[1].at<double>(1, 0)));//夹角cos值
	double xy_angle = acos(angle_cos);
	double angle_sin = sin(xy_angle);
	cv::Mat R = cv::Mat_<double>(2, 2);//角度旋转矩阵
	R.at<double>(0, 0) = angle_cos;
	R.at<double>(0, 1) = -angle_sin;
	R.at<double>(1, 0) = angle_sin;
	R.at<double>(1, 1) = angle_cos;

	cv::gemm(R.inv(), lvector[0], 1, 0, 0, lvector[2], 0);
	double dx = lvector[2].at<double>(0, 0) - lvector[1].at<double>(0, 0), dy = lvector[2].at<double>(1, 0) - lvector[1].at<double>(1, 0);
    std::cout<<"dx = "<<std::endl;
    std::cout<<dx<<std::endl;
    std::cout<<"dy = "<<std::endl;
    std::cout<<dy<<std::endl;
	if (fabs(dx) > 0.1 || fabs(dy) > 0.1)
    {
        v1 = _v2;
        v2 = _v1;
        return true;
    }
    return false;
}


void getSunPosition::adjustInteraction(cv::Point2f src_point[11], cv::Point2f dst_point[11])
{
    cv::Point2f v1, v2;
    v1 = getVanishPoint(src_point[0],src_point[1], src_point[2], src_point[3]);
    v2 = getVanishPoint(src_point[4],src_point[5], src_point[6], src_point[7]);
    for(int i=0; i<11; i++)
    {
        dst_point[i] = src_point[i];
    }
    if(adjust_vanishing_point(v1, v2, src_point[10], v1, v2) == true)
    {
        cv::Point2f temp;
        for(int i=0; i<4; i++)
        {
            temp = dst_point[i+4];
            dst_point[i+4] = dst_point[i];
            dst_point[i] = temp;

        }
    }
    return ;

}
void getSunPosition::setVideoFileName(std::string _filename)
{
	fileName = _filename;

}
void getSunPosition::setnFrameStart(int nfs)
{
	nFrameStart = nfs;
}
int getSunPosition::getnFrameStart()
{
	return nFrameStart;
}
void getSunPosition::setnFrameEnd(int nfe)
{
	nFrameEnd = nfe;
}
int getSunPosition::getnFrameEnd()
{
	return nFrameEnd;
}
cv::Mat getSunPosition::getFrameStart()
{
	cv::VideoCapture capture(fileName); // load video
	std::cout << fileName << std::endl;
	if (!capture.isOpened()) {
		std::cout << "Failed to open video." << std::endl;
		exit(1);
	}
	
	for(int i = 1 ; i <=nFrameStart ; i++)//找到交互初始帧
	{
		capture >> FrameStart;
	}
	return FrameStart;

}

 int getSunPosition::sunPositionProcess(cv::Point2f point_data[])
 {
    cv::VideoCapture capture(fileName); // load video
    std::cout << fileName << std::endl;
    if (!capture.isOpened()) {
        std::cout << "Failed to open video." << std::endl;
        return -1;
    }
    std::ofstream outPut("out.txt");
    if(!outPut.is_open())
    {
        std::cout<<"error occured while creating file out.txt"<<std::endl;
        std::cout<<"program closed..."<<std::endl;
        return -1;
    }

    
	cv::Mat frame = getFrameStart();
    cv::Point2f line_point[4][LINE_POINT_NUM];//线上的坐标点
    int out_point_mark[11] , out_line_mark[4][LINE_POINT_NUM];//记录出界的信息点
    for(int i = 0; i<11; i++)//默认所有点没出界
    {
        out_point_mark[i]=1;
    }
    for(int i = 0 ; i < 4 ;i++)
    {
        for(int j = 0 ; j < LINE_POINT_NUM ;j++)
        {
            out_line_mark[i][j]=1;
        }
    }
    extend(point_data);//数据线过短时加以延长
    find_line_point(point_data,line_point);//初始化线上的点

    cv::VideoWriter writer("dst.avi", CV_FOURCC('M', 'J', 'P', 'G'), 8.0, cv::Size(frame.cols, frame.rows));
    bool if_frame_break = false;
    cv::Point2f v1,v2,vi;
    double focal_length;
    cv::Mat R = cv::Mat_<double>(3,3);
    std::vector<cv::Point2d> datatemp;
    datatemp.push_back(point_data[10]);
    datatemp.push_back(point_data[11]);

    double xyLength = getXYlength(datatemp,200);

    for (int frame_num = 0; frame_num < nFrameEnd&& !frame.empty(); frame_num++, capture >> frame)
    {
        std::cout << "Frame No."<< frame_num+nFrameStart << std::endl;
        outPut << "#Frame No." <<frame_num+nFrameStart <<std::endl;

        cv::Point2f point[11];
        for(int i=0; i<11; i++)
        {
            point[i] = point_data[i]-cv::Point2f(frame.cols*0.5, frame.rows*0.5);
        }
       // adjustInteraction(point, point);
        v1 = getVanishPoint(point[0], point[1], point[2], point[3]);
        v2 = getVanishPoint(point[4], point[5], point[6], point[7]);
        focal_length = getFocalLength(v1, v2);


        //get_vp_for_R(frame, pointVec, v1, v2, focal_length);

        //adjust_vanishing_point(v1, v2, point[10], v1, v2);

        cv::Point3f Xc, Yc, Zc;

        get_rotation_mat( v1, v2,  focal_length, R);

		//std::cout<<">> no adjust R = "<<R<<std::endl;
        bool flag=adjust_rotation_matrix(R, R);
        //std::cout<<">> adjust R "<<R<<std::endl;

        /**compute translation vector*/
        cv::Point2f origin;
        origin.x = point_data[10].x-0.5*frame.cols;
        origin.y = point_data[10].y-0.5*frame.rows;
        /**bool get_translation_vector(const cv::Point2f origin, const cv::Point2f v1, const cv::Mat R, const double focal_length, double T[]);**/
        double T[3] = {0,0,0};
        if(!get_translation_vector(origin, v1, R, focal_length, T) )
        {
            std::cout<<">> error while computing T"<<std::endl;
            continue;
        }
        //std::cout<<"T = "<<std::endl;
       // std::cout<<T[0]<<", "<<T[1]<<", "<<T[2]<<std::endl;



        /**get prarmeter in gluLookAt() function for openGL**/

        cv::Mat t = cv::Mat_<double>(3,1);/**平移矩阵*/
       /* if(std::isnan(T[0]))
            break;*/

        t.at<double>(0,0) = T[0];
        t.at<double>(1,0) = T[1];
        t.at<double>(2,0) = T[2];

//
//        std::cout<<">> R = "<<std::endl;
//        std::cout<< R <<std::endl;
//        std::cout<<">> T = "<<std::endl;
//        std::cout<< t <<std::endl;

        /***********************************  ************************************************************/
        /**                      R, T 均已求出.                                                           *
        *                                                                                                 *
        ***************************************************************************************************/
        double phi, theta;
        double xxxx, yyyy;

        cv::Point2d sunPos;
        getSunPos(frame.rows, frame.cols, point_data, sunPos, theta, phi);
        xxxx = (float)sunPos.x;
        yyyy = (float)sunPos.y;


        std::vector<cv::Point2f> triangle;
        triangle.push_back(point_data[9]);
        triangle.push_back(point_data[10]);
        triangle.push_back(point_data[8]);
		//std::cout<<"R="<<R<<std::endl;
		//std::cout<<"T="<<t<<std::endl;
		//std::cout<<"f="<<focal_length<<std::endl;
        getAngle( R, t,focal_length ,triangle,theta,phi,frame.cols,frame.rows);
        /**将参数坐标系转换成笛卡尔坐标系**/
        double dex, dey, dez;


         xyLength = xyLength;

        //std::cout<<"xyLength:"<<xyLength<<std::endl;

        para_to_decare( xyLength,phi, theta, dex, dey, dez);
        //para_to_decare(10000,0.908, -0.082581, dex, dey, dez);
        std::vector<cv::Point3d> objectPoints;
        std::vector<cv::Point2d> imagePoints(3);
        cv::Point3d Position;

        Position.x = xyLength;
        Position.y = 0;
        Position.z = 0;
        objectPoints.push_back(Position);

        Position.x = 0;
        Position.y = xyLength;
        Position.z = 0;
        objectPoints.push_back(Position);
       /* Position.x = 0;
        Position.y = 0;
        Position.z = 100000;
        objectPoints.push_back(Position);*/
        Position.x = dex;
        Position.y = dey;
        Position.z = dez;
        objectPoints.push_back(Position);

        cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
        cv::Rodrigues(R,rVec);
        cv::Mat K = cv::Mat::zeros(cv::Size(3,3),CV_64F);
        K.at<double>(0,0) = focal_length;
        K.at<double>(1,1) = focal_length;
        K.at<double>(0,2) = frame.cols/2.0;
        K.at<double>(1,2) = frame.rows/2.0;
        K.at<double>(2,2) = 1.0;

        //cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);   // Distortion vector
        cv::Mat distCoeffs = cv::Mat::zeros(cv::Size(5,1),CV_64F);
        cv::projectPoints(objectPoints,rVec,t,K,distCoeffs,imagePoints);
//		world2img(R, t,focal_length,objectPoints[1],imagePoints[1]); //compute the image
//		world2img(R, t,focal_length,objectPoints[0],imagePoints[0]);
//		world2img(R, t,focal_length,objectPoints[2],imagePoints[2]);
		//world2img(R, t,focal_length,objectPoints[3],imagePoints[3]);
      /*  std::cout<<">> x-axis = "<<std::endl;
        std::cout<<imagePoints[0].x<<", "<<imagePoints[0].y<<std::endl;
        std::cout<<">> y-axis = "<<std::endl;
        std::cout<<imagePoints[1].x<<", "<<imagePoints[1].y<<std::endl;
        std::cout<<">> z-axis = "<<std::endl;
        std::cout<<imagePoints[2].x<<", "<<imagePoints[2].y<<std::endl;
        std::cout<<">> sun position = "<<std::endl;
        std::cout<<imagePoints[3].x<<", "<<imagePoints[3].y<<std::endl;*/
//        cv::line(frame,point_data[10],imagePoints[0],cv::Scalar(0,255,0),2,8);
//        cv::line(frame,point_data[10],imagePoints[1],cv::Scalar(255,0,0),2,8);
//        cv::line(frame,point_data[10],imagePoints[2],cv::Scalar(0,255,255),2,8);
     //  cv::line(frame,point_data[10],imagePoints[3],cv::Scalar(0,0,0),2,8);
//        cv::line(frame,point_data[10], cv::Point(592,207), cv::Scalar(34,35,23));
//        cv::line(frame,point_data[10], cv::Point(-2046, -14615), cv::Scalar(255,135,23));
        /**sun direction in world coordinate*/
        cv::Mat Pw = cv::Mat_<double>(4,1);  /**世界坐标系下太阳位置**/
        static cv::Mat P0c = cv::Mat_<double>(4,1);/**初始帧相机坐标*/
        cv::Mat Pc = cv::Mat_<double>(4,1); /**当前帧相机坐标*/
        //cv::Mat t = cv::Mat_<double>(3,1);  /**平移矩阵*/
        static cv::Mat M0 = cv::Mat_<double>(4,4);
        cv::Mat M  = cv::Mat_<double>(4,4);

        Pw.at<double>(0,0) = dex;
        Pw.at<double>(1,0) = dey;
        Pw.at<double>(2,0) = dez;
        Pw.at<double>(3,0) = 1.0;

        t.at<double>(0,0) = T[0];
        t.at<double>(1,0) = T[1];
        t.at<double>(2,0) = T[2];

        if(frame_num == 0) /**第一帧*/
        {
            /**switch to camera coordinate*/
            switch_to_cam_coordinate(Pw, R, t, P0c, M0);
            cv::Mat P = cv::Mat_<double>(3,1);
            P.at<double>(0,0) = P0c.at<double>(0,0);
            P.at<double>(1,0) = P0c.at<double>(1,0);
            P.at<double>(2,0) = P0c.at<double>(2,0);
            outPut<<"# Sun Position in World Coordinate = "<<std::endl;
            outPut<<"("<<Pw.at<double>(0,0)<<", "<<Pw.at<double>(1,0)<<", "<<Pw.at<double>(2,0)<<")"<<std::endl;

            cv::normalize(P,P);
            outPut<<"# Sun position in camera coordinate = "<<std::endl;
            outPut<<"("<<P.at<double>(0,0)<<", "<<P.at<double>(1,0)<<", "<<P.at<double>(2,0)<<")"<<std::endl;
            //outPut<<P<<std::endl;
//            cv::normalize(P,P);
//            outPut  << "#Direction = " <<std::endl;
//            outPut<<P<<std::endl;
        }
        else
        {
            /**switch to camera coordinate*/
            switch_to_cam_coordinate(Pw, R, t, Pc,M);
            cv::Mat P = cv::Mat_<double>(3,1);
            P.at<double>(0,0) = Pc.at<double>(0,0);
            P.at<double>(1,0) = Pc.at<double>(1,0);
            P.at<double>(2,0) = Pc.at<double>(2,0);

//            std::cout<<">> Relation Matrix = "<<std::endl;
//            std::cout<<M*M0.inv()<<std::endl;
//
//            outPut << "#Relation Matrix = " <<std::endl;
//            outPut << M*M0.inv() <<std::endl;

//            std::cout<<">> T = "<<std::endl;
//            std::cout<<t<<std::endl;
            cv::normalize(P,P);
            outPut<<"# Sun Position in camera coordinate = "<<std::endl;
            outPut<<"("<<P.at<double>(0,0)<<", "<<P.at<double>(1,0)<<", "<<P.at<double>(2,0)<<")"<<std::endl;
//            cv::normalize(P, P);
//            std::cout<<P<<std::endl;
        }


        cv::Point2f point_data1[11];
        for (int i = 0; i < 11; i++)
        {
            point_data1[i] = point_data[i];
        }

        int out_point_num = -1;  //记录是否有信息点出界

        tracking(frame,point_data1,out_point_mark,line_point,out_line_mark);//光流跟踪
        fitline_reAllocation(point_data1,line_point);//直线拟合，重分配点

        for(int i=0 ; i < 11 ; i++)//判断是否有点出界
        {
            if(out_point_mark[i])
            {
                if(point_data1[i].x > frame.cols || point_data1[i].x < 0 || point_data1[i].y > frame.rows || point_data1[i].y < 0)
                {
                    out_point_mark[i] = 0 ;
                    out_point_num = i ;
                }
            }

        }
        for(int i = 0 ; i < 4 ;i++)
        {
            for(int j = 0 ; j < LINE_POINT_NUM ;j++)
            {
                if(out_line_mark[i][j])
                {
                    if(line_point[i][j].x > frame.cols ||line_point[i][j].x < 0 || line_point[i][j].y > frame.rows || line_point[i][j].y < 0)
                    {
                        out_line_mark[i][j] = 0 ;
//                           QMessageBox::warning(NULL, "warning", "Line's points is out of the video's boundary", QMessageBox::Yes , QMessageBox::Yes);
//                           if_frame_break = true;
                    }
                }

            }
        }
        if_frame_break = reAllocation(frame, point_data1,out_point_mark,line_point,out_line_mark);


            if (out_point_num >= 8)
            {
                std::cout<<"people's point is out of frame"<<std::endl;
                if_frame_break = true;
            }//人及影子的信息点出界直接判断不能继续运行

            if(if_frame_break == true)
            {
                std::cout<<"some point is out of frame"<<std::endl;
                break ;
            }


        for (int i = 0; i < 11; i++)
        {
            point_data[i] = point_data1[i];
        }

		//draw_coordinate_sys(point_data, frame, xxxx ,yyyy);
		draw_coordinate_sys(point_data,frame,imagePoints ,flag);
	//	draw_coordinate_sys(point_data, frame,flag , xxxx ,yyyy);

        draw_line_2(frame,point_data,out_point_mark,line_point,out_line_mark);

        writer << frame;
        cv::imshow("readvideo",frame);
        //cv::VideoWriter writer("dst.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15.0, cv::Size(frame.cols, frame.rows));
        cv::waitKey(200);
        std::cout<<"process succeeded..."<<std::endl;

    }
    outPut.close();
    gray_prev.release();
    return 0;
 }