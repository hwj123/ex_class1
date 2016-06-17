#include"getSunPosition.h"
#include<string>
using namespace std;

int main(int args, char *argv[])
{
    if (args != 3)
    {
        cout << "Error usage!" << endl;
        return -1;
    }
    //string sInitNum(argv[1]);
    int init_num = 1;/**默认从第一帧开始交互*/
    //init_num = atoi(sInitNum.data());
    string fileName(argv[1]);
    string pts(argv[2]);
    cv::Point2f points[11];
    char lb = '(';
    char rb = ')';
    char slb = '[';
    char srb = ']';
    char comma = ',';

    int point_num = 0;

    size_t pos1 = 0, pos2 = 0, pos3;

    pos1 = pts.find(slb, 0);
    if(pos1 != string::npos)
    {
           pos2 = pts.find(srb, pos1);
    }
    if (pos1 != string::npos && pos2 != string::npos)
    {
        init_num = atoi(pts.substr(pos1+1, pos2 - pos1 -1).c_str());
    }
    pos1 = 0;
     pos2 = 0;
    do
    {
        pos1 = pts.find(lb, pos2);
        if (pos1 != string::npos)
        {
            pos2 = pts.find(rb, pos1);
        }
        if (pos1 != string::npos && pos2 != string::npos)
        {
            pos3 = pts.find(comma, pos1);
            if (pos3 != string::npos)
            {
                points[point_num].x = atof(pts.substr(pos1 + 1, pos3 - pos1 - 1).c_str());
                points[point_num].y = atof(pts.substr(pos3 + 1, pos2 - pos3 - 1).c_str());
                cout << "point " << point_num << "[" << points[point_num].x << "," << points[point_num].y << "]" << endl;
                ++point_num;
            }
            else
            {
                cout << "Error parameter format." << endl;
                return -1;
            }
        }
    }
    while (pos1 != string::npos && pos2 != string::npos);
    std::cout<<"init num = "<<init_num<<std::endl;
	/*getSunPosition gsp;
	gsp.setnFrameEnd(20);
    gsp.sunPositionProcess(init_num, fileName, points);*/
	getSunPosition gsp1;
	gsp1.setVideoFileName(fileName);
	gsp1.setnFrameEnd(20);
	gsp1.setnFrameStart(1);
	gsp1.sunPositionProcess(points);
	system("pause");
    return 0;

}