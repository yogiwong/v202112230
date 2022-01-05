/*************************************************
Copyright: 光电实验室
Create Date: 2021-11-08
Description: 对角膜进行处理,形成角膜地形图
Version: 1.0:实现盘中心定位、提取各环的轮廓
		 2.0:拟合各个环【未完成】

Return: 0-正常运行; 901-无法打开文件; 902-图片为空; 903-无法定位Placido氏盘的中心; 904-Placido氏盘的中心定位错误;
Description：【▲表示有 参数/步骤 可以调整】
**************************************************/

#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include<math.h>
#include <algorithm>

using namespace std;
using namespace cv;

struct piecewise_poly_para
{
	vector<cv::Mat> A_set;
	vector<int> div_point;
};

struct placido_ring
{
	//index 第几个环，从0开始
	int index;
	vector <cv::Point> ring;
	//dis2cen 环上的点到中心的距离 distance to center
	vector <double> dis2cen;
};

void GammaTransform(const cv::Mat& src, cv::Mat& dst, double gamma)
{

	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)i / 255.0, gamma) * 255.0f);
	}
	dst = src.clone();
	int channels = src.channels();
	switch (channels)
	{
	case 1:
	{
		cv::MatIterator_<uchar> it = dst.begin<uchar>();
		cv::MatIterator_<uchar> end = dst.end<uchar>();
		while (it != end)
		{
			*it = lut[(*it)];
			it++;
		}
		break;
	}
	case 3:
	{
		cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b> end = dst.end<cv::Vec3b>();
		while (it != end)
		{
			(*it)[0] = lut[(*it)[0]];
			(*it)[1] = lut[(*it)[1]];
			(*it)[2] = lut[(*it)[2]];
			it++;
		}
		break;
	}
	default:
		break;
	}
}

// 轮廓按照面积大小升序排序
bool ascendSort(vector<cv::Point> a, vector<cv::Point> b) {
	return a.size() < b.size();
}


// 轮廓按照面积大小降序排序
bool descendSort(vector<cv::Point> a, vector<cv::Point> b) {
	return a.size() > b.size();
}


// 去除小面积
cv::Mat bwareaopen(cv::Mat bwImg, int npixsmall2)
{
	cv::Mat labels, stats, centroids;
	int nccomps = cv::connectedComponentsWithStats(bwImg, labels, stats, centroids, 8, CV_32S);
	cv::Mat bwImg2 = cv::Mat::zeros(bwImg.size(), CV_8UC1);
	bwImg.copyTo(bwImg2);
	for (int y = 0; y < bwImg2.rows; ++y)
	{
		for (int x = 0; x < bwImg2.cols; ++x)
		{
			int label = labels.at<int>(y, x); // labels里面是按0-n标注的检测顺序标签
			int label_size = stats.at<int>(label, cv::CC_STAT_AREA); // stats里面是按0-n标签对应的检测区域大小
			if (label_size < npixsmall2)
			{
				bwImg2.at<uchar>(y, x) = 0;
			}
		}
	}
	return bwImg2;
}


// 判断输入图片是否为灰度图并将彩图输出灰度图
cv::Mat to_gray(cv::Mat img)
{
	if (img.channels() == 1)
	{
		;
	}
	else if (img.channels() == 3)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	}
	else
	{
		printf("Incorrect number of image channels\n"); // 出现错误：通道数错误
	}
	return img;
}


// 漫水填充法
cv::Mat floodFill_four_corners(cv::Mat src, double Boundary_distance, int newVal)
{
	Mat dst = src.clone();
	for (int i = 0; i < 4; ++i) //遍历4个角
	{
		// 对四个角进行填充
		cv::floodFill(dst, cv::Point((i % 2) * (dst.cols - Boundary_distance), (i / 2) * (dst.rows - Boundary_distance)), cv::Scalar(newVal));
	}
	return dst;
}


// 扫描ring
vector<vector<cv::Point>> label_ring(cv::Mat img)
{
	cv::threshold(img, img, 100, 255, cv::THRESH_BINARY); // ▲
	vector<vector<cv::Point>> ring_set;
	cv::Mat img_ = img.clone();

	//原点坐标，当采集某ring中出现了缺失值就在这个轮廓的yy的纵坐标所有点都为一个原点，细化后补齐
	cv::Point no = cv::Point();

	//xx：横坐标，yy：纵坐标，
	//index：ring点集索引，只采集前(ring_num = 18个环的数据。flag：连续采集到0的次数，
	int xx, yy, index, flag, pre_value, ring_num = 18;
	for (index = 0; index < ring_num; index++)
	{
		vector<cv::Point> ring;
		ring_set.push_back(ring);
	}
	index = 0;
	//uchar* p;
	for (yy = 0; yy < img_.rows; yy++)
	{
		//if(yy!=0)
			//cout << yy << endl;
		index = 0;
		flag = 0;
		pre_value = 0;
		//uchar* p;
		//p = img_.ptr<uchar>(yy);
		for (xx = 5; xx < img_.cols; xx++)
		{
			if (img_.at<uchar>(cv::Point(xx, yy)) == 255)
			{
				pre_value = 255;
				flag = 0;
				if (index > ring_num - 1)
				{
					break;
				}
				else
				{
					ring_set[index].push_back(cv::Point(xx, yy));
				}
			}
			else
			{
				//判断是否为外边缘，如果为外边缘index到下一个ring的vector
				if (pre_value == 255)
				{
					pre_value = 0;
					index++;
					flag++;
				}
				else
				{
					pre_value = 0;
					//连续对黑色区域采样了40个点之后证明没有这个ring出现了断点，压栈一个原点
					if (index == 8 || index == 0 || index == 7)
					{
						if (flag == 45)
						{
							ring_set[index].push_back(no);
							index++;
							flag = 0;
						}
						else
						{
							flag++;
						}
					}
					else
					{
						if (flag == 35)
						{
							if (index > ring_num - 1)
								break;
							ring_set[index].push_back(no);
							index++;
							flag = 0;
						}
						else
						{
							flag++;
						}
					}
				}
			}
		}
	}
	return ring_set;
}


// 将子集全部压栈到全集中
vector<cv::Point> push_all(vector<cv::Point> subset, vector<cv::Point>& set)
{
	for (int i = 0; i < subset.size(); i++)
	{
		set.push_back(subset[i]);
	}
	return set;
}


/********************************************************************************************************
	* @brief 对输入图像进行细化,骨骼化
	* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
	* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
	*/
Mat thin_getPoints(cv::Mat src, cv::Mat& dst)
{
	src = src / 255;
	const int maxIterations = -1; // 限制迭代次数
	assert(src.type() == CV_8UC1);
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
			break;
		std::vector<uchar*> mFlag; //用于标记需要删除的点  
		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}

		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}
	}

	// 显示骨架、连接处、交叉点、端点
	dst = dst * 255; // 显示骨架
	return dst;
}

// 排序; 代码来源：https://blog.csdn.net/lyq_12/article/details/80755261
/********************************************************************************************************
	* @brief 对每条连通域上的各点根据Cho(x/y)进行Sort_by(0/1)排序
	* @param: inputContours为输入轮廓
	* @param: Cho为排序的对象; (x/X/0)表示对x进行排序; (y/Y/1)表示对y进行排序
	* @param: Sort_by排序的方式; (0/descend)表示从大到小排序,即降序; (1/ascend)表示从小到大排序,即升序;
	* @return: outputContours为输出轮廓
	*/
vector<vector<Point>>  SortContourPoint(vector<vector<Point>> inputContours)
{
	vector<Point> tempContoursPoint;
	vector<vector<Point>> outputContours;
	for (int i = 0; i < inputContours.size(); i++)
	{
		tempContoursPoint.clear(); //每次循环注意清空
		// 除2是因为：提取的轮廓有重复的内外两层，需要删去一层
		//for (int j = 0; j < inputContours[i].size() / 2; j++)
		for (int j = 0; j < inputContours[i].size() ; j++)
		{
			//for (int k = j; k < inputContours[i].size() / 2; k++)
			for (int k = j; k < inputContours[i].size() ; k++)
			{
				if (inputContours[i][j].y >inputContours[i][k].y)
				{
					swap(inputContours[i][j], inputContours[i][k]);
				}
			}
			tempContoursPoint.push_back(inputContours[i][j]);
		}
		//降重
		outputContours.push_back(tempContoursPoint);
		vector<cv::Point>().swap(tempContoursPoint);
		tempContoursPoint.push_back(outputContours[i][0]);
		for (int j = 1; j < outputContours[i].size(); j++)
		{
			if (outputContours[i][j].y != outputContours[i][j - 1].y)
			{
				tempContoursPoint.push_back(outputContours[i][j]);
			}
		}
		outputContours[i].swap(tempContoursPoint);
	}
	return outputContours;
}

/*
* 多项式拟合
*/

Mat polyfit(const vector<cv::Point> p, const int k = 4)
{
	int n = p.size();
	//范德蒙矩阵，行为：[1  ， x  ,  x^2  ,  x^3  ,  ...  ,  x^k],超定方程组系数矩阵。
	//求解超定方程组时，需要在等式两边同时相乘一个超定方程系数矩阵的转置
	//即：$\phi ^T \phi A=\phi ^T Y$
	cv::Mat phi = cv::Mat::zeros(cv::Size(k + 1, n), CV_64F);
	cv::Mat phi_T = phi.clone();
	cv::Mat A = cv::Mat::zeros(cv::Size(1, k + 1), CV_64F);
	cv::Mat Y = cv::Mat::zeros(cv::Size(1, n), CV_64F);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j <= k; j++)
		{
			phi.at<double>(i, j) = double(pow(double(p[i].x), double(j)));
		}
		Y.at<double>(i, 0) = double(p[i].y);
	}
	cv::transpose(phi.clone(), phi_T);
	cv::solve((phi_T * phi), (phi_T * Y), A);
	return A;
}

/*
* 多项式拟合表达式
*/
float polycaulate(const double x, const Mat A)
{
	double y = 0.0;
	for (int i = 0; i < A.rows; i++)
	{
		y = y + (A.at<double>(i, 0) * double(pow(double(x), double(i))));
	}
	return y;
}

//多项式拟合
vector<cv::Point> poly_predict(const cv::Mat A, const int n, const cv::Point first_point,
	const int x_begin,const int x_end, const bool flag = false)
{
	vector<cv::Point> fit_line;
	for (int ii = x_begin; ii <=x_end;ii++)
	{

	}
	return fit_line;
}





//分段多项式拟合中的内联函数，主要是把某个段的待拟合的点弄成一个点集
inline vector<cv::Point> subindex(const vector<cv::Point> v_now, const int i_piece, const int step)
{
	//i_piece :第i_piece段待拟合子点集
	vector<cv::Point> subset;
	int begin_index, end_index;
	begin_index = i_piece * step;
	end_index = (i_piece + 1) * step;
	if ((v_now.size()-1) < end_index)
	{
		end_index = (v_now.size()-1);
	}

	for (int i = begin_index; i <= end_index; i++)
	{
		subset.push_back(v_now[i]);
	}
	return subset;
}

//分段多项式拟合
piecewise_poly_para PiecewisePoly_fit(const vector<cv::Point> v_now, const int n_piece)
{
	//v_now 待拟合曲线点集
	//n_piece 分段拟合的段数，分段思路是
	piecewise_poly_para A;
	int n = v_now.size();
	int step = int(n / n_piece);
	for (int i = 0; i < n_piece; i++)
	{
		A.div_point.push_back(v_now[step * i].x);
	}
	A.div_point.push_back(v_now[v_now.size()-1].x);
	for (int i = 0; i < n_piece; i++)
	{
		vector<cv::Point> subset;
		subset = subindex(v_now, i, step);
		cv::Mat A_;
		A_ = polyfit(subset, 4);
		A.A_set.push_back(A_);
	}
	return A;
}

//分段多项式拟合预测
vector<cv::Point> PiecewisePoly_predict(const piecewise_poly_para A, const int n_piece, const int n, 
	                     const int step,const cv::Point first_point,const bool flag=false)
{
	//n 拟合点数
	//first_point 扫描出来点的第一个点的坐标
	//flag 是否需要使用对第一个点之前的点进行拟合
	vector<cv::Point> fit_line;
	vector<vector<cv::Point>> temp;
	for (int i = 0; i < n; i++)
	{
		int vv;
		for (int j = 0; j < n_piece; j++)
		{
			if (first_point.y>=25 && flag)
			{
				if (i <= first_point.y)
				{
					fit_line.push_back(cv::Point(first_point.x,i));
				}
				else
				{
					if ((i >= A.div_point[j]) && (i < A.div_point[j + 1]))
					{
						vv = int(polycaulate(double(i), A.A_set[j]));
						fit_line.push_back(cv::Point(vv, i));
					}
					else if (i >= A.div_point[n_piece])
					{
						vv = int(polycaulate(double(i), A.A_set[n_piece - 1]));
						fit_line.push_back(cv::Point(vv, i));
					}
					else if (i < A.div_point[0])
					{
						vv = int(polycaulate(double(i), A.A_set[0]));
						fit_line.push_back(cv::Point(vv, i));
					}
				}
			}
			else
			{
				if ((i >= A.div_point[j]) && (i < A.div_point[j + 1]))
				{
					vv = int(polycaulate(double(i), A.A_set[j]));
					fit_line.push_back(cv::Point(vv, i));
				}
				else if (i >= A.div_point[n_piece])
				{
					vv = int(polycaulate(double(i), A.A_set[n_piece - 1]));
					fit_line.push_back(cv::Point(vv, i));
				}
				else if (i < A.div_point[0])
				{
					vv = int(polycaulate(double(i), A.A_set[0]));
					fit_line.push_back(cv::Point(vv, i));
				}
			}
		}
	}
	temp.push_back(fit_line);
	vector<cv::Point>().swap(fit_line);
	temp=SortContourPoint(temp);
	fit_line = temp[0];
	return fit_line;
}


/*
* 二分查找
*/
int binary_search(const vector<cv::Point> point,const int target)
{
	int left = 0;
	int right = point.size()-1;
	while (left <= right)
	{
		int mid = left + ((right - left) >> 1);
		if (point[mid].y > target)
		{
			right = mid - 1;
		}
		else if (point[mid].y < target)
		{
			left = mid + 1;
		}
		else
		{
			return 1;
			//break;
		}
	}
	return -1;
}

/*
* 相关系数
*/
double pearson(vector<double> xx, vector<double> yy)
{
	double cof, xm=0.0, ym=0.0;
	for (int i = 0; i < xx.size(); i++)
	{
		xm = xm + xx[i];
		ym = ym + yy[i];
	}
	xm = xm / double(xx.size());
	ym= ym / double(yy.size());
	cv::Mat x = cv::Mat::zeros(cv::Size(1, xx.size()),CV_64F);
	cv::Mat y = cv::Mat::zeros(cv::Size(1, yy.size()), CV_64F);
	for (int i = 0; i < xx.size(); i++)
	{
		x.at<double>(cv::Point(0, i)) = xx[i] - xm;
		y.at<double>(cv::Point(0, i)) = yy[i] - ym;
	}
	cv::Mat x_= cv::Mat::zeros(cv::Size(xx.size(),1), CV_64F);
	cv::Mat y_ = cv::Mat::zeros(cv::Size(yy.size(),1), CV_64F);
	cv::transpose(x, x_);
	cv::transpose(y, y_);
	//double x1, x2, x3;
	cv::Mat x1 = x_ * y;
	cv::Mat x2 = x_ * x;
	cv::Mat x3 = y_ * y;
	cof = x1.at<double>(0,0) / (sqrt(x2.at<double>(0,0))*sqrt(x3.at<double>(0, 0)));
	return cof;
}

///*
//* 重拍程序
//*/
//inline void rephoto()
//{
//
//}

/*
* 找下一个环
* 为了防止中间有环被省略了，所以发现了一条新轮廓之后要每隔5个点之后回去看看有没有白线
*/
vector<cv::Point> find_next_ring(const cv::Mat polar_thin,const vector<cv::Point> last_ring ,
	const int ring_index,const int gap9=50,const float search_rate=1.0)
{
	//一条环的全部
	vector<cv::Point> v_now;
	cv::Mat polar_thin_ = polar_thin.clone();
	//cv::Mat polar_thin_=
	//last_x是上一个环的横坐标，也就是本次扫描的x的起始位置
	int yy = 0, xx, gap,
		last_y=polar_thin.rows + 10,last_x;
	if (ring_index == 7 || ring_index == 8)
		gap = 60;
	else if (ring_index == 9)
		gap = gap9;
	else
		gap = 40;
	while (yy < last_ring.size())
	{ 
		if (yy > int(polar_thin.rows * search_rate))
		{
			break;
		}
		last_y = polar_thin.rows + 10;
		//当前检测到的轮廓
		vector<vector<cv::Point>> temp;
		last_x = last_ring[yy].x;
		int temp_size = 0;
		vector<vector<cv::Point>> temp_couter;
		for (xx = last_x+15; xx < gap+last_x; xx++)
		{
			int values = polar_thin.at<uchar>(cv::Point(xx, yy));
			if (values == 255)
			{
				vector<vector<cv::Point>>().swap(temp);
				cv::Mat polar_thin_flood = polar_thin.clone();
				cv::floodFill(polar_thin_flood, cv::Point(xx, yy), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // 删除当前环数
				cv::floodFill(polar_thin_, cv::Point(xx, yy), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // 删除当前环数
				cv::bitwise_xor(polar_thin_flood, polar_thin.clone(), polar_thin_flood); // 用异运算提取该环
				cv::findContours(polar_thin_flood, temp, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
				temp = SortContourPoint(temp);
				v_now = push_all(temp[0], v_now);
				last_y = temp[0][temp[0].size()-1].y;
				temp_size = temp[0].size();
				break;
			}
		}
		if (last_y == polar_thin.rows + 10)
		{
			yy++;
		}
		else if ((last_y + 3) < polar_thin.rows-2)
		{
			yy = last_y + 2;
			//yy用last_y更新之后将last_y重新初始化
			last_y = polar_thin.rows + 10;
		}
		else
		{
			break;
		}
	}
	return v_now;
}

//索引所有的placido环
vector<vector<cv::Point>> find_all_ring(cv::Mat polar_thin)
{
	vector<vector<cv::Point>> ring_set;
	vector<cv::Point> last_ring, init_ring,v_now,v_now_;
	//last_ring ：上一个环，主要是用于下一个环的索引，当索引第0个环的时候就是等于init_ring，否则就是ring_set 的i-1环
	// init_ring ：初始化环，主要是用于
	for (int ii = 0; ii < polar_thin.rows; ii++)
	{
		init_ring.push_back(cv::Point(5,ii));
	}
	for (int i = 0; i <= 18; i++)
	{
		vector<cv::Point>().swap(last_ring);
		if (i == 0)
		{
			last_ring = init_ring;
		}
		else
		{
			last_ring = ring_set[i - 1];
		}
		

	}
	return ring_set;
}

//求解vector均值
double vector_means(vector<int> v)
{
	double vmeans = 0.0, vsize;
	vsize = double(v.size());
	for (int ii = 0; ii < v.size(); ii++)
	{
		vmeans += double(v[ii]);
	}
	vmeans /= vsize;
	return vmeans;
}

//求解vector的标准差
double vector_std(vector<int> v)
{
	double vstd=0.0, vmeans,vsize;
	vmeans = vector_means(v);
	vsize = double(v.size());
	for (int ii = 0; ii < v.size(); ii++)
	{
		vstd += (double(v[ii])-vmeans)* (double(v[ii]) - vmeans);
	}
	vstd /= (vsize-1.0);
	vstd = sqrt(vstd);
	return vstd;
}

//取反函数
inline bool notfunction(bool flag)
{
	if (flag==true)
		flag = false;
	else
		flag = true;
	return flag;
}

//检测扫描环的时候是不是扫描多了，如果扫描到了下一个环，就将其删掉
vector<cv::Point> drop_next_ring(vector <cv::Point> v_now,vector<cv::Point> last_ring)
{
	int ii;
	bool flag = true;
	vector <cv::Point> v_now_;
	vector<int> v_xdelta,v_ydelta;
	vector<cv::Point> ::iterator vbeg = v_now.begin(), vend = v_now.end();
	//计算扫描到的点集当前点和下一个点的x与y的差值
	//v_xdelta and v_ydelta 是差值的集合
	while (vbeg != (vend-1))
	{
		int xdelta,ydelta;
		xdelta = abs((vbeg + 1)->x) - ((vbeg)->x);
		ydelta = abs((vbeg + 1)->y) - ((vbeg)->y);
		v_xdelta.push_back(xdelta);
		v_ydelta.push_back(ydelta);
		vbeg++;
	}
	//
	for (ii = 0; ii < v_now.size()-1; ii++)
	{
		if ((v_xdelta[ii]<18 && v_xdelta[ii]>-18) && flag==true)
			v_now_.push_back(v_now[ii]);
		else if ((v_xdelta[ii] >= 18 || v_xdelta[ii] <= -18) && (v_ydelta[ii] > 50))
		{
			if (((v_now[ii].x - last_ring[v_now[ii].y].x) <= 35 && (v_now[ii].x - last_ring[v_now[ii].y].x) >= 18) 
				&& flag==true)
			{
				v_now_.push_back(v_now[ii]);
			}
			else 
			{
				//if(ii==224)
				//cout <<11<<ii <<endl;
				flag=notfunction(flag);
				continue;
			}
		}
		else if((v_xdelta[ii] >= 18 || v_xdelta[ii] <= -18) && (v_ydelta[ii] <= 50))
		{
			//cout << 22<<ii << endl;
			flag = notfunction(flag);
			continue;
		}
	}
	return v_now_;
}

// 主函数
int main()
{
	//********** 图片路径 **********//
	string basic_path = "D:/placido/data/normal/"; // 保存的基本路径
	//string basic_path = "F:/Arr/21.11.08/phfine/"; // 保存的基本路径
	string folder = basic_path + "1/"; // 导入文件夹路径
	string gauss = basic_path + "gauss/";//高斯滤波结果保存路径
	string adathr = basic_path + "adath/";//自适应阈值分割结果保存路径
	string closer = basic_path + "close/";//自适应分割后进行闭运算，用于下一步漫水填充
	string localr = basic_path + "local/";
	string coutour_midr = basic_path + "cen/";//寻找最中心用于定位圆心的轮廓
	string scanr = basic_path + "scan/";//扫描后
	string dealth = basic_path + "dealth/";
	string polarr = basic_path + "polar/";
	string imgpr = basic_path + "img_point/";
	string roir = basic_path + "roi/";
	string tr = basic_path + "thin/";
	string tr_c = basic_path + "thin_contours/";
	string bg_path = basic_path + "bg/";



	//********** 导入图片 **********//
	vector<cv::String> imagePathList; // 存储所有文件绝对路径
	cv::glob(folder, imagePathList); // 遍历文件
	cout << "there are " << imagePathList.size() << " files in this root path!!! " << endl;
	for (int index =16; index < imagePathList.size(); ++index)
	{
		/*
		* 图片预处理模块：
		* 1.检验读取到的图片是否为空，若为空则跳下一张。并将读取到的照片转为灰度图
		* 2.直方图均衡化后进行5*5的sigma=0.9的高斯滤波，并保存结果图在gauss路径中
		*/
		string file = imagePathList[index].substr(folder.length(), imagePathList[index].size());//图片文件名
		cout << index << ":   " << imagePathList[index] << std::endl;
		cv::Mat img = cv::imread(imagePathList[index]);
		cv::Mat img_gray;
		if (img.empty() != 0)
		{
			std::cout << "is  empty file !!!!" << std::endl;
			//return 901;
			continue;
		}
		else
		{
			img_gray = to_gray(img); // 灰度化
		}



		// 高斯滤波
		cv::Mat img_gauss = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::GaussianBlur(img_gray, img_gauss, cv::Size(11, 11), 1.2); // 高斯滤波▲
		//GammaTransform(img_gauss,img_gauss,1.5);
		cv::imwrite(gauss + file, img_gauss);



		// 自适应阈值分割
		cv::Mat img_adath = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::adaptiveThreshold(img_gauss, img_adath, 255,
			cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 7, -1.2); // 自适应阈值分割▲
		cv::medianBlur(img_adath, img_adath, 3); // 中值滤波▲
		cv::imwrite(adathr + file, img_adath);



		// 漫水填充法取最中间的若干个环
		//先对二值图进行闭运算，为了将中心可能出现部分小断环闭合起来
		int flood_dist = 10; // 漫水填充距离边界的距离▲
		cv::Mat img_close = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat img_flood = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // 闭运算卷积核▲
		int area_adpt = cv::countNonZero(img_adath); // 统计分割区域的面积
		if (area_adpt > (img_adath.cols * img_adath.rows) * 0.15) // 若分割出的面积大于图片尺寸一定比例时,则不进行闭运算▲
		{
			img_close = img_adath.clone(); // 直接用自适应图代替闭运算的图
		}
		else
		{
			cv::morphologyEx(img_adath, img_close, cv::MORPH_CLOSE, close_kernel); // 闭运算,连接断裂处
		}
		img_flood = img_close.clone(); // 克隆图片,便于后续调整
		img_close = bwareaopen(img_close, 5); // 删除像素小于5的像素块▲
		img_flood = floodFill_four_corners(img_flood, flood_dist, 255); // 把背景填白
		cv::medianBlur(img_flood, img_flood, 3); // 中值滤波▲
		int area_flood = cv::countNonZero(~img_flood); // 统计分割区域的面积
		if (area_flood < 250) // 判断是否没有圆▲
		{
			img_flood = img_close.clone(); // 克隆图片,便于后续调整
			cv::morphologyEx(img_flood, img_flood, cv::MORPH_CLOSE, close_kernel); // 闭运算,连接断裂处
			img_flood = floodFill_four_corners(img_flood, flood_dist, 255); // 把背景填白
			img_flood = floodFill_four_corners(img_flood, flood_dist, 0); // 把背景填黑
		}
		else if (area_flood < 2500) // 判断是否剩下一个小圆▲
		{
			img_flood = ~img_flood; // 取反,方便后续操作
			cv::medianBlur(img_flood, img_flood, 3); // 中值滤波去噪点▲
		}
		else
		{
			img_flood = floodFill_four_corners(img_flood, flood_dist, 0); // 把背景填黑
		}
		cv::imwrite(closer + file, img_close);


		// 提取中心轮廓
		vector<vector<cv::Point>> cou_cen;//中心轮廓
		cv::Point circle_mid;
		cv::Mat img_cou = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat e_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); // 卷积核▲
		cv::Mat imgoral = img.clone();
		cv::bitwise_not(img_flood, img_flood); // 非运算
		cv::findContours(img_flood, cou_cen, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // 找轮廓
		sort(cou_cen.begin(), cou_cen.end(), ascendSort); // 对轮廓进行排序
		int i, j = 0;
		for (i = 0; i < cou_cen.size(); ++i)
		{
			//去掉轮廓数太少的孤立噪声点
			if (cou_cen[i].size() > 50)
			{
				//判断轮廓的第一个点是否在相对比较中心的地方，防止采到一些奇奇怪怪的东西
				if ((cou_cen[i][0].x > 50) && (cou_cen[i][0].x < img_gray.cols - 50)) // ▲
				{
					if ((cou_cen[i][0].y > 50) && (cou_cen[i][0].y < img_gray.rows - 50)) // ▲
					{
						cv::Mat img_cou = cv::Mat::zeros(img_gray.size(), CV_8UC1);
						cv::drawContours(img_cou, cou_cen, i, cv::Scalar(255), cv::FILLED);
						vector<cv::Vec3f> circles;
						cv::HoughCircles(img_cou, circles, cv::HOUGH_GRADIENT, 1, 200, 255, 11, 6, 43); // ▲
						if (circles.size() != 0)
						{
							cv::Moments m = cv::moments(img_cou, true); // 提取质心的信息
							circle_mid.x = m.m10 / m.m00; // 存储质心的x坐标
							circle_mid.y = m.m01 / m.m00; // 存储质心的y坐标
							cv::circle(imgoral, cv::Point(circle_mid), 2, cv::Scalar(0, 0, 255), -1); // 标记圆心▲
							cv::imwrite(coutour_midr + file, img_cou);
							cv::imwrite(localr + file, imgoral);
							/*
														cv::erode(img_cou, img_cou, e_kernel);
														cv::bitwise_and(img_close, ~img_cou, img_close);
														cv::imwrite(dealth + file, img_close);*/
							break;
						}
					}
				}
			}
		}
		if ((circle_mid.x == 0) && (circle_mid.y == 0))
		{
			printf("Unable to locate the center of Placido's disk\n"); // 出现错误:无法定位Placido氏盘的中心
			//return 903; // 返回错误值
			continue;
		}
		else if ((circle_mid.x == 0) || (circle_mid.y == 0))
		{
			printf("Central positioning error of Placido's disk\n"); // 出现错误:Placido氏盘的中心定位错误
			//return 904; // 返回错误值
			continue;
		}



		/*
		* 去眼睑：
		* 原理：
		* 1.对二值图进行采样，具体采样过程见：https://github.com/psurya1994/polar-scanning-algorithm
		* 2.通过采样能够在眼睑的位置停止，采样图极坐标化后做闭运算，把所有采样点全部粘起来，取得mask
		* 3.采样图与二值图and操作，得到roi
		*/

		// 采样，找出
		double pi = 3.1415926;
		cv::Mat img_B = img_close.clone(); // 获取边界图(B:get boundary)
		cv::Mat img_point = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		int Samp_num = 300; // 环上的采样点的数量(Number of ring sampling points)▲
		int Samp_rmax = (img_B.cols <= img_B.rows ? img_B.cols : img_B.rows); // 最大采样半径(Maximum sampling radius)▲
		vector<cv::Point> B_point; // 边缘点(B:boundary)
		vector<float> B_dist; // 边界点到中心的距离(B:boundary;dist:distance)
		close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::morphologyEx(img_B, img_B, cv::MORPH_CLOSE, close_kernel);
		cv::medianBlur(img_B, img_B, 3); // 中值滤波
		for (double theta = 0.0; theta < 360.0; (theta += 360 / Samp_num)) // 遍历角度
		{
			//nn：记录不发生跳变次数，当nn大于等于(nn_max = 16)的时候结束对该角度的采样
			int Samp_r = 4, nn = 0, nn_max = 16; // 采样半径(sampling radius)▲
			double theta_rad = (theta * pi) / 180; // 转换为弧度制
			cv::Point search_point = circle_mid + cv::Point(floor(Samp_r * cos(theta_rad)), floor(Samp_r * sin(theta_rad))); // 搜索点的坐标
			bool prev_spvalue = false; // 上一个搜寻点的值(previous search point value)
			bool inside_bounds = false; // 判断搜索点是否在图内(if a search given point is inside the bounds of the image.)
			do
			{
				if (nn >= nn_max)
					break;
				if (bool(img_B.at<uchar>(search_point)) != prev_spvalue) // 判断是否与上一个像素相同
				{
					B_point.push_back(search_point); // 保存边缘点
					float dist = sqrt(pow((search_point.x - circle_mid.x), 2) 
						+ pow((search_point.y - circle_mid.y), 2)); // 计算边界点到中心的距离
					B_dist.push_back(round(dist)); // 保存边界点到中心的距离
					prev_spvalue = (!prev_spvalue); // 反转上一个搜寻点的值
					Samp_r += 1; // 提高运算速度,不检测一个像素宽(▲可选择删除)
					cv::circle(imgoral, cv::Point(search_point), 0, cv::Scalar(0, 255, 255), -1);
					cv::circle(img_point, cv::Point(search_point), 0, 255, -1);
					nn = 0;
				}
				else
				{
					++nn;
				}
				Samp_r += 1; // 半径往外扩
				search_point = circle_mid + cv::Point(floor(Samp_r * cos(theta_rad)), floor(Samp_r * sin(theta_rad))); // 提取下一个搜索点
				inside_bounds = inside_bounds = ((search_point.x < img_B.cols) && (search_point.x >= 0)
					&& (search_point.y < img_B.rows) && (search_point.y >= 0)); // 重新判断下一个搜素点是否在图内

			} while (inside_bounds); // 遍历该角度的从中心到边界的像素
		}
		cv::Mat cl_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); // ▲
		for (i = 0; i < 10; ++i) // ▲
			cv::morphologyEx(img_point, img_point, cv::MORPH_CLOSE, cl_kernel);
		cv::medianBlur(img_point, img_point, 3); // ▲
		cv::Mat img_roi;
		cv::bitwise_and(img_point, img_close, img_roi);
		img_roi = bwareaopen(img_roi, 20); // ▲
		cv::imwrite(scanr + file, imgoral);
		cv::imwrite(imgpr + file, img_point);
		cv::imwrite(roir + file, img_roi);



		// 极坐标变换
		cv::Mat polar, invpolar, polar_point, toph, polar_gradx, polar_gradxi, polar_gradxe;
		cv::warpPolar(img_roi, polar, cv::Size(img_gray.cols * 2, img_gray.rows * 2), circle_mid,
			circle_mid.x * 0.85, cv::INTER_LINEAR + cv::WARP_POLAR_LINEAR); // 极坐标转换半径不能太大，太大会将边缘也弄进来 ▲
		cv::medianBlur(polar, polar, 3); // ▲
		cv::threshold(polar, polar, 206, 255, cv::THRESH_BINARY); // 去除两个半圆▲
		cv::imwrite(polarr + file, polar);



		// 骨骼化
		cv::Mat polar_thin, polar_thin_contours;
		thin_getPoints(polar.clone(), polar_thin);
		cv::imwrite(basic_path +"thino/o" + file, polar_thin);



		// 删除横向信息边缘
		cv::Matx33d la_kernel(1,0,-1,2,0,-2,1,0,-1); // ▲
		//cv::Matx12d la_kernel(-1,1);
		cv::Mat polar_thin_=polar_thin.clone(), la_kernel_t,polar_thin__= polar_thin.clone();
		//cv::transpose(la_kernel, la_kernel_t);
		//cv::Matx33d la_kernel(-1,1,0,0,0,0,0,0,0);
		cv::filter2D(polar_thin.clone(), polar_thin, -1, la_kernel);
		//polar_thin__ = polar_thin.clone();
		//cv::filter2D(polar_thin, polar_thin, -1, la_kernel);
		//polar_thin = bwareaopen(polar_thin, 35);// ▲
		polar_thin = bwareaopen(polar_thin, 20);
		polar_thin__ = polar_thin.clone();
		cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
		cv::morphologyEx(polar_thin,polar_thin,cv::MORPH_CLOSE,kernel_open);
		//cv::subtract(polar_thin,polar_thin_,polar_thin_);
		cv::imwrite(tr + file , polar_thin);



		// 环索引拟合
		vector<vector<cv::Point>> ring_set;
		vector<Point> v_now; // 储存当前环的轮廓
		vector<vector<int>> ring_xdelta;
		vector<int> v_xdelta;
		int x = 5, y = 0; // x,y坐标
		int x_dist_max = 90; // 找环的最大范围
		int x_before = 0; // 前一环直线的质心坐标
		int dist_max = 20; // 线的质心之间最大的容忍范围.
		int polar_thin_rows = polar_thin.rows; // 提取图片宽度
		int polar_thin_cols = polar_thin.cols; // 提取图片宽度
		int ring_index=0;
		double vstd, vmeans;
		int last_ring_xmeans;
		vector <vector<double>> ring_sigma;
		int x_delta = 0,last_x_means=0;
		while (y < polar_thin_rows)
		{
			vector<vector<cv::Point>> temp;
			int last_y = polar_thin_rows + 10;
			for (x = 5; x < 60; x++)
			{
				int value = polar_thin.at<uchar>(cv::Point(x,y));
				if (value == 255)
				{
					cv::Mat polar_flood = polar_thin.clone();
					cv::floodFill(polar_flood, cv::Point(x, y), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // 删除当前环数
					cv::bitwise_xor(polar_flood, polar_thin.clone(), polar_flood); // 用异运算提取该环
					cv::findContours(polar_flood,temp, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
					temp=SortContourPoint(temp);
					v_now = push_all(temp[0], v_now);
					last_y=temp[0][temp[0].size() - 1].y;

					break;
				}
			}
			if (last_y == polar_thin_rows + 10)
			{
				++y;
			}
			else if ((last_y + 2) < polar_thin_rows)
			{
				y = last_y + 2;
			}
			else
			{
				break;
			}
		}
		cv::floodFill(polar_thin, cv::Point(v_now[0].x, v_now[0].y), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8);
		vector<cv::Point> p;
		//set<cv::Point> temp_set(v_now.begin(),v_now.end());
		//v_now.assign(temp_set.begin(),temp_set.end());
		cout << "there are " << v_now.size() << " points in this line " << endl;
		for (i = 0; i < v_now.size(); i++)
		{
			p.push_back(cv::Point(v_now[i].y, v_now[i].x));
		}
		int step = v_now.size()/6;
		piecewise_poly_para A;
		A = PiecewisePoly_fit(p,6);
		//cout <<A.A_set[0] <<endl;
		vector<cv::Point>fit_line;
		fit_line = PiecewisePoly_predict(A,6,polar.rows,step,v_now[0]);
		cv::Mat wh = cv::Mat::zeros(polar_thin.size(),CV_8U);
		//画线
		for (int i = 0; i < v_now.size(); i++)
		{
			cv::circle(wh,v_now[i],0,cv::Scalar(255),-1); 
		}
		for (int i = 0; i < fit_line.size(); i++)
		{
			cv::circle(wh, fit_line[i], 0, cv::Scalar(100), -1);
		}
		ring_set.push_back(fit_line);
		//cv::Mat wh = cv::Mat::zeros(polar_thin.size(), CV_8U);
		for (int i = 1; i < 18; i++)
		{
			if (i == 16)
				cout << endl;
			cv::floodFill(polar_thin, cv::Point(v_now[0].x, v_now[0].y),
				cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8);
			vector<cv::Point> v_now_;
			vector<cv::Point>().swap(v_now);
			v_now = find_next_ring(polar_thin, ring_set[i-1], i);
			//将扫描错的清掉
			v_now_ = drop_next_ring(v_now,ring_set[i-1]);
			v_now_.swap(v_now);
			rescan:
			cout << "ring "<< i <<"  :  " << v_now.size() << endl;
			//对于在第8个环之内的并且扫描出来的点数是少于502*0.65=326个点的时候，认为这张图收到影响太大，建议重拍
			//注意 如果眼睫毛影响到太里面的环，那么外面的环一般情况下受到的影响会更大，所以建议去重拍
			if (i < 9 && v_now.size() < int(0.65 * polar_thin_rows))
			{
				cout << "the count of this ring: " << i << " is too small" << endl;
				/*
				* 重拍程序接口
				*/
				break;
			}
			//点数太少了，看看能不能满足大于502*0.3个点
			else if (i>=9 && v_now.size() < int(0.3* polar_thin_rows))
			{
				int num = 0, ii=0;
				vector<cv::Point> ::iterator v_now_beg = v_now.begin(), v_now_end = v_now.end();
				//统计当点数太少的时候，看看前300个点之前有没有相对比较完整的线
				while ((v_now_beg != v_now_end)&&((v_now_beg->y) <= 300 ))
				{
					num++;
					v_now_beg++;
				}
				if (num>=180) 
				{
					vector<cv::Point>().swap(p);
					vector<cv::Point>().swap(fit_line);
					v_now_beg = v_now.begin();
					while ((v_now_beg != v_now_end) && ((v_now_beg->y) <= 300))
					{
						p.push_back(cv::Point(v_now_beg->y, v_now_beg->x));
						v_now_beg++;
					}
					piecewise_poly_para AA;
					//对于在上半段点数大于180个点，进行分段多项式拟合
					AA = PiecewisePoly_fit(p, 3);
					fit_line = PiecewisePoly_predict(AA, 3, p[p.size()-1].x, p.size() / 3,v_now[0],true);
					v_now_beg = v_now.begin(), v_now_end = v_now.end();
					while ((v_now_beg != v_now_end) && ((v_now_beg->y) <= 300))
					{
						cv::circle(wh, *v_now_beg, 0, cv::Scalar(255), -1);
						v_now_beg ++;
					}
					for (int ii = 0; ii < fit_line.size(); ii++)
					{
						cv::circle(wh, fit_line[ii], 0, cv::Scalar(100), -1);
					}
				}
				else
				{
					cout << "the count of this ring: " << i << " is too small" << endl;
					break;
				}
			}
			//满足点数不会出现过少的情况
			else
			{
				vector<cv::Point>().swap(p);
				vector<cv::Point>().swap(fit_line);
				for (int ii = 0; ii < v_now.size(); ii++)
				{
					p.push_back(cv::Point(v_now[ii].y, v_now[ii].x));
				}
				if (v_now.size() <= 501 && v_now.size() >= 400)
				{
					//大于400直接用多项式拟合
					//当扫描出来的第一个点的纵坐标如果是大于25的时候，就用开头端的x就是用第一个x，不能直接进行拟合
					cv::Mat AA = polyfit(p, 8);
					for (int ii = 0; ii < polar_thin.rows; ii++)
					{
						int vv;
						//当扫描的第0个点的y大于等于25并且待拟合的点的y小于25
						if ((v_now[0].y >= 25) && (ii<=25))
							vv = v_now[0].x;
						
						else
							vv = int(polycaulate(double(ii), AA));
						fit_line.push_back(cv::Point(vv, ii));
					}
				}
				else if (v_now.size() < 400)
				{
					//当前点和下一个点之间的y差值
					vector<int> delta;
					int max_delta=0, max_y,max_ii;
					for (int ii = 0; ii < (v_now.size() - 2); ii++)
					{
						delta.push_back(v_now[ii+1].y-v_now[i].y);
						if ((v_now[ii + 1].y - v_now[ii].y) > max_delta)
						{
							max_delta = (v_now[ii + 1].y - v_now[ii].y);
							max_y = v_now[ii].y;
							max_ii = ii;
						}
					}
					//当出现了超大的间隔，说明这个还受到了上眼睑的影响，
					//如果影响像素点小于等于150的时候，认为影响相对比较少，可以直接对整个线做分段拟合
					if (max_delta <= 150)
					{
						piecewise_poly_para AA;
						vector<cv::Point>().swap(fit_line);
						if (v_now[v_now.size() - 1].y <= 485)
						{
							AA = PiecewisePoly_fit(p, 4);
							fit_line = PiecewisePoly_predict(AA, 4, v_now[v_now.size() - 1].y + 10, 
								p.size() / 4, v_now[0], true);
						}
						else
						{
							AA = PiecewisePoly_fit(p, 4);
							fit_line = PiecewisePoly_predict(AA, 4, polar_thin.rows, 
								p.size() / 4, v_now[0], true);
						}
					}
					else 
					{
						//last_最大间隔之后，剩下多少个点
						//如果剩下的点数是大于等于50，则对当前环的上半段的点和下半段剩余的点分别做拟合
						//如果剩下的点小于50个，那么只对上半段的点做拟合
						int last_ = v_now.size() - max_ii;
						if (last_ >= 50)
						{
							if (v_now[v_now.size() - 1].y >= 485)
							{
								cv::Mat AA1, AA2;
								vector<cv::Point> p1, p2;
								for (int ii = 0; ii <= max_ii; ii++)
								{
									p1.push_back(p[ii]);
								}
								AA1 = polyfit(p1, 4);
								for (int ii = max_ii + 1; ii < v_now.size(); ii++)
								{
									p2.push_back(p[ii]);
								}
								AA2 = polyfit(p2, 4);
								//如果后半段剩余的点的最后一个剩余点距离502相差太太远，强行拟合会出现龙格现象，所以对于这种情况只拟合到
								for (int ii = 0; ii < polar_thin.rows; ii++)
								{
									int vv;
									if (ii <= max_y)
									{
										if (((v_now[0].y >= 25) && (ii <= 25)))
											vv = v_now[0].x;
										else
											vv = int(polycaulate(double(ii), AA1));
										fit_line.push_back(cv::Point(vv, ii));
									}
									else if (ii >= (max_y + max_delta - 8))
									{
										vv = int(polycaulate(double(ii), AA2));
										fit_line.push_back(cv::Point(vv, ii));
									}
									else
										continue;
								}
							}
							else
							{
								cv::Mat AA1, AA2;
								vector<cv::Point> p1, p2;
								for (int ii = 0; ii <= v_now[v_now.size() - 1].y+10; ii++)
								{
									p1.push_back(p[ii]);
								}
								AA1 = polyfit(p1, 4);
								for (int ii = max_ii + 1; ii < v_now.size(); ii++)
								{
									p2.push_back(p[ii]); 
								}
								AA2 = polyfit(p2, 4);
								for (int ii = 0; ii < polar_thin.rows; ii++)
								{
									int vv;
									if (ii <= max_y)
									{
										vv = int(polycaulate(double(ii), AA1));
										fit_line.push_back(cv::Point(vv, ii));
									}
									else if (ii >= (max_y + max_delta - 8))
									{
										vv = int(polycaulate(double(ii), AA2));
										fit_line.push_back(cv::Point(vv, ii));
									}
									else

										continue;
								}
							}
						}
						else 
						{
							piecewise_poly_para AA;
							vector<cv::Point>().swap(fit_line);
							//vector<cv::Point> p_;
							AA = PiecewisePoly_fit(p, 3);
							fit_line = PiecewisePoly_predict(AA, 3, v_now[max_ii].y, p.size() / 3,v_now[0],true);
						}
					}

				}
				//如果出现了v_now大于502，那么说明扫描到了下一条线
				else if (v_now.size() >=502)
				{
					int max_delta = 0;
					//当检测扫描到的点的点数为502的时候，可能出现有这条线是一条完整的线，或者扫描到了部分下一条线
					if (v_now.size() == 502)
					{
						//max_delta是指前后两个点的y坐标的差值的最大值，如果这条线是完整的，那么最大间隔肯定不会大于1
						for (int ii = 0; ii < (v_now.size() - 2); ii++)
						{
							if ((v_now[ii + 1].y - v_now[ii].y) > max_delta)
							{
								max_delta = (v_now[ii + 1].y - v_now[ii].y);
								//如果出现最大间隔是大于等于2的，就说明出现了
								if (max_delta >= 2)
								{
									break;
								}
							}
						}
						//如果扫描到前后两个点的纵坐标间隔是大于1，就说明这502
						if (max_delta >= 2)
						{
							vector<cv::Point>().swap(v_now);
							v_now = find_next_ring(polar_thin, ring_set[i - 1], i, 35);
							cout << "begin rescan" << endl;
							goto rescan;
						} 
						else
						{
							fit_line = v_now;
						}
					}
					else
					{
						vector<cv::Point>().swap(v_now);
						v_now = find_next_ring(polar_thin, ring_set[i - 1], i, 35);
						cout << "begin rescan" << endl;
						goto rescan;
					}

				}
				//画线
				for (int ii = 0; ii < v_now.size(); ii++)
				{
					cv::circle(wh, v_now[ii], 0, cv::Scalar(255), -1);
				}
				for (int ii = 0; ii < fit_line.size(); ii++)
				{
					cv::circle(wh, fit_line[ii], 0, cv::Scalar(100), -1);
				}
				ring_set.push_back(fit_line);
				vector<cv::Point>().swap(fit_line);
				if (i == 7)
					cout << endl;
			}
		}
		cv::imwrite(basic_path + "bg/" + file + ".png", wh);
		//cv::dnn::readNetFromTorch
	}
	return 0;
}
