/*************************************************
Copyright: ���ʵ����
Create Date: 2021-11-08
Description: �Խ�Ĥ���д���,�γɽ�Ĥ����ͼ
Version: 1.0:ʵ�������Ķ�λ����ȡ����������
		 2.0:��ϸ�������δ��ɡ�

Return: 0-��������; 901-�޷����ļ�; 902-ͼƬΪ��; 903-�޷���λPlacido���̵�����; 904-Placido���̵����Ķ�λ����;
Description��������ʾ�� ����/���� ���Ե�����
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
	//index �ڼ���������0��ʼ
	int index;
	vector <cv::Point> ring;
	//dis2cen ���ϵĵ㵽���ĵľ��� distance to center
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

// �������������С��������
bool ascendSort(vector<cv::Point> a, vector<cv::Point> b) {
	return a.size() < b.size();
}


// �������������С��������
bool descendSort(vector<cv::Point> a, vector<cv::Point> b) {
	return a.size() > b.size();
}


// ȥ��С���
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
			int label = labels.at<int>(y, x); // labels�����ǰ�0-n��ע�ļ��˳���ǩ
			int label_size = stats.at<int>(label, cv::CC_STAT_AREA); // stats�����ǰ�0-n��ǩ��Ӧ�ļ�������С
			if (label_size < npixsmall2)
			{
				bwImg2.at<uchar>(y, x) = 0;
			}
		}
	}
	return bwImg2;
}


// �ж�����ͼƬ�Ƿ�Ϊ�Ҷ�ͼ������ͼ����Ҷ�ͼ
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
		printf("Incorrect number of image channels\n"); // ���ִ���ͨ��������
	}
	return img;
}


// ��ˮ��䷨
cv::Mat floodFill_four_corners(cv::Mat src, double Boundary_distance, int newVal)
{
	Mat dst = src.clone();
	for (int i = 0; i < 4; ++i) //����4����
	{
		// ���ĸ��ǽ������
		cv::floodFill(dst, cv::Point((i % 2) * (dst.cols - Boundary_distance), (i / 2) * (dst.rows - Boundary_distance)), cv::Scalar(newVal));
	}
	return dst;
}


// ɨ��ring
vector<vector<cv::Point>> label_ring(cv::Mat img)
{
	cv::threshold(img, img, 100, 255, cv::THRESH_BINARY); // ��
	vector<vector<cv::Point>> ring_set;
	cv::Mat img_ = img.clone();

	//ԭ�����꣬���ɼ�ĳring�г�����ȱʧֵ�������������yy�����������е㶼Ϊһ��ԭ�㣬ϸ������
	cv::Point no = cv::Point();

	//xx�������꣬yy�������꣬
	//index��ring�㼯������ֻ�ɼ�ǰ(ring_num = 18���������ݡ�flag�������ɼ���0�Ĵ�����
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
				//�ж��Ƿ�Ϊ���Ե�����Ϊ���Եindex����һ��ring��vector
				if (pre_value == 255)
				{
					pre_value = 0;
					index++;
					flag++;
				}
				else
				{
					pre_value = 0;
					//�����Ժ�ɫ���������40����֮��֤��û�����ring�����˶ϵ㣬ѹջһ��ԭ��
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


// ���Ӽ�ȫ��ѹջ��ȫ����
vector<cv::Point> push_all(vector<cv::Point> subset, vector<cv::Point>& set)
{
	for (int i = 0; i < subset.size(); i++)
	{
		set.push_back(subset[i]);
	}
	return set;
}


/********************************************************************************************************
	* @brief ������ͼ�����ϸ��,������
	* @param srcΪ����ͼ��,��cvThreshold�����������8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
	* @return Ϊ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
	*/
Mat thin_getPoints(cv::Mat src, cv::Mat& dst)
{
	src = src / 255;
	const int maxIterations = -1; // ���Ƶ�������
	assert(src.type() == CV_8UC1);
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //��¼��������  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
			break;
		std::vector<uchar*> mFlag; //���ڱ����Ҫɾ���ĵ�  
		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
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
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}

		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
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
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}
	}

	// ��ʾ�Ǽܡ����Ӵ�������㡢�˵�
	dst = dst * 255; // ��ʾ�Ǽ�
	return dst;
}

// ����; ������Դ��https://blog.csdn.net/lyq_12/article/details/80755261
/********************************************************************************************************
	* @brief ��ÿ����ͨ���ϵĸ������Cho(x/y)����Sort_by(0/1)����
	* @param: inputContoursΪ��������
	* @param: ChoΪ����Ķ���; (x/X/0)��ʾ��x��������; (y/Y/1)��ʾ��y��������
	* @param: Sort_by����ķ�ʽ; (0/descend)��ʾ�Ӵ�С����,������; (1/ascend)��ʾ��С��������,������;
	* @return: outputContoursΪ�������
	*/
vector<vector<Point>>  SortContourPoint(vector<vector<Point>> inputContours)
{
	vector<Point> tempContoursPoint;
	vector<vector<Point>> outputContours;
	for (int i = 0; i < inputContours.size(); i++)
	{
		tempContoursPoint.clear(); //ÿ��ѭ��ע�����
		// ��2����Ϊ����ȡ���������ظ����������㣬��Ҫɾȥһ��
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
		//����
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
* ����ʽ���
*/

Mat polyfit(const vector<cv::Point> p, const int k = 4)
{
	int n = p.size();
	//�����ɾ�����Ϊ��[1  �� x  ,  x^2  ,  x^3  ,  ...  ,  x^k],����������ϵ������
	//��ⳬ��������ʱ����Ҫ�ڵ�ʽ����ͬʱ���һ����������ϵ�������ת��
	//����$\phi ^T \phi A=\phi ^T Y$
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
* ����ʽ��ϱ��ʽ
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

//����ʽ���
vector<cv::Point> poly_predict(const cv::Mat A, const int n, const cv::Point first_point,
	const int x_begin,const int x_end, const bool flag = false)
{
	vector<cv::Point> fit_line;
	for (int ii = x_begin; ii <=x_end;ii++)
	{

	}
	return fit_line;
}





//�ֶζ���ʽ����е�������������Ҫ�ǰ�ĳ���εĴ���ϵĵ�Ū��һ���㼯
inline vector<cv::Point> subindex(const vector<cv::Point> v_now, const int i_piece, const int step)
{
	//i_piece :��i_piece�δ�����ӵ㼯
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

//�ֶζ���ʽ���
piecewise_poly_para PiecewisePoly_fit(const vector<cv::Point> v_now, const int n_piece)
{
	//v_now ��������ߵ㼯
	//n_piece �ֶ���ϵĶ������ֶ�˼·��
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

//�ֶζ���ʽ���Ԥ��
vector<cv::Point> PiecewisePoly_predict(const piecewise_poly_para A, const int n_piece, const int n, 
	                     const int step,const cv::Point first_point,const bool flag=false)
{
	//n ��ϵ���
	//first_point ɨ�������ĵ�һ���������
	//flag �Ƿ���Ҫʹ�öԵ�һ����֮ǰ�ĵ�������
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
* ���ֲ���
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
* ���ϵ��
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
//* ���ĳ���
//*/
//inline void rephoto()
//{
//
//}

/*
* ����һ����
* Ϊ�˷�ֹ�м��л���ʡ���ˣ����Է�����һ��������֮��Ҫÿ��5����֮���ȥ������û�а���
*/
vector<cv::Point> find_next_ring(const cv::Mat polar_thin,const vector<cv::Point> last_ring ,
	const int ring_index,const int gap9=50,const float search_rate=1.0)
{
	//һ������ȫ��
	vector<cv::Point> v_now;
	cv::Mat polar_thin_ = polar_thin.clone();
	//cv::Mat polar_thin_=
	//last_x����һ�����ĺ����꣬Ҳ���Ǳ���ɨ���x����ʼλ��
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
		//��ǰ��⵽������
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
				cv::floodFill(polar_thin_flood, cv::Point(xx, yy), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // ɾ����ǰ����
				cv::floodFill(polar_thin_, cv::Point(xx, yy), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // ɾ����ǰ����
				cv::bitwise_xor(polar_thin_flood, polar_thin.clone(), polar_thin_flood); // ����������ȡ�û�
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
			//yy��last_y����֮��last_y���³�ʼ��
			last_y = polar_thin.rows + 10;
		}
		else
		{
			break;
		}
	}
	return v_now;
}

//�������е�placido��
vector<vector<cv::Point>> find_all_ring(cv::Mat polar_thin)
{
	vector<vector<cv::Point>> ring_set;
	vector<cv::Point> last_ring, init_ring,v_now,v_now_;
	//last_ring ����һ��������Ҫ��������һ��������������������0������ʱ����ǵ���init_ring���������ring_set ��i-1��
	// init_ring ����ʼ��������Ҫ������
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

//���vector��ֵ
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

//���vector�ı�׼��
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

//ȡ������
inline bool notfunction(bool flag)
{
	if (flag==true)
		flag = false;
	else
		flag = true;
	return flag;
}

//���ɨ�軷��ʱ���ǲ���ɨ����ˣ����ɨ�赽����һ�������ͽ���ɾ��
vector<cv::Point> drop_next_ring(vector <cv::Point> v_now,vector<cv::Point> last_ring)
{
	int ii;
	bool flag = true;
	vector <cv::Point> v_now_;
	vector<int> v_xdelta,v_ydelta;
	vector<cv::Point> ::iterator vbeg = v_now.begin(), vend = v_now.end();
	//����ɨ�赽�ĵ㼯��ǰ�����һ�����x��y�Ĳ�ֵ
	//v_xdelta and v_ydelta �ǲ�ֵ�ļ���
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

// ������
int main()
{
	//********** ͼƬ·�� **********//
	string basic_path = "D:/placido/data/normal/"; // ����Ļ���·��
	//string basic_path = "F:/Arr/21.11.08/phfine/"; // ����Ļ���·��
	string folder = basic_path + "1/"; // �����ļ���·��
	string gauss = basic_path + "gauss/";//��˹�˲��������·��
	string adathr = basic_path + "adath/";//����Ӧ��ֵ�ָ�������·��
	string closer = basic_path + "close/";//����Ӧ�ָ����б����㣬������һ����ˮ���
	string localr = basic_path + "local/";
	string coutour_midr = basic_path + "cen/";//Ѱ�����������ڶ�λԲ�ĵ�����
	string scanr = basic_path + "scan/";//ɨ���
	string dealth = basic_path + "dealth/";
	string polarr = basic_path + "polar/";
	string imgpr = basic_path + "img_point/";
	string roir = basic_path + "roi/";
	string tr = basic_path + "thin/";
	string tr_c = basic_path + "thin_contours/";
	string bg_path = basic_path + "bg/";



	//********** ����ͼƬ **********//
	vector<cv::String> imagePathList; // �洢�����ļ�����·��
	cv::glob(folder, imagePathList); // �����ļ�
	cout << "there are " << imagePathList.size() << " files in this root path!!! " << endl;
	for (int index =16; index < imagePathList.size(); ++index)
	{
		/*
		* ͼƬԤ����ģ�飺
		* 1.�����ȡ����ͼƬ�Ƿ�Ϊ�գ���Ϊ��������һ�š�������ȡ������ƬתΪ�Ҷ�ͼ
		* 2.ֱ��ͼ���⻯�����5*5��sigma=0.9�ĸ�˹�˲�����������ͼ��gauss·����
		*/
		string file = imagePathList[index].substr(folder.length(), imagePathList[index].size());//ͼƬ�ļ���
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
			img_gray = to_gray(img); // �ҶȻ�
		}



		// ��˹�˲�
		cv::Mat img_gauss = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::GaussianBlur(img_gray, img_gauss, cv::Size(11, 11), 1.2); // ��˹�˲���
		//GammaTransform(img_gauss,img_gauss,1.5);
		cv::imwrite(gauss + file, img_gauss);



		// ����Ӧ��ֵ�ָ�
		cv::Mat img_adath = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::adaptiveThreshold(img_gauss, img_adath, 255,
			cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 7, -1.2); // ����Ӧ��ֵ�ָ��
		cv::medianBlur(img_adath, img_adath, 3); // ��ֵ�˲���
		cv::imwrite(adathr + file, img_adath);



		// ��ˮ��䷨ȡ���м�����ɸ���
		//�ȶԶ�ֵͼ���б����㣬Ϊ�˽����Ŀ��ܳ��ֲ���С�ϻ��պ�����
		int flood_dist = 10; // ��ˮ������߽�ľ����
		cv::Mat img_close = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat img_flood = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // ���������ˡ�
		int area_adpt = cv::countNonZero(img_adath); // ͳ�Ʒָ���������
		if (area_adpt > (img_adath.cols * img_adath.rows) * 0.15) // ���ָ�����������ͼƬ�ߴ�һ������ʱ,�򲻽��б������
		{
			img_close = img_adath.clone(); // ֱ��������Ӧͼ����������ͼ
		}
		else
		{
			cv::morphologyEx(img_adath, img_close, cv::MORPH_CLOSE, close_kernel); // ������,���Ӷ��Ѵ�
		}
		img_flood = img_close.clone(); // ��¡ͼƬ,���ں�������
		img_close = bwareaopen(img_close, 5); // ɾ������С��5�����ؿ��
		img_flood = floodFill_four_corners(img_flood, flood_dist, 255); // �ѱ������
		cv::medianBlur(img_flood, img_flood, 3); // ��ֵ�˲���
		int area_flood = cv::countNonZero(~img_flood); // ͳ�Ʒָ���������
		if (area_flood < 250) // �ж��Ƿ�û��Բ��
		{
			img_flood = img_close.clone(); // ��¡ͼƬ,���ں�������
			cv::morphologyEx(img_flood, img_flood, cv::MORPH_CLOSE, close_kernel); // ������,���Ӷ��Ѵ�
			img_flood = floodFill_four_corners(img_flood, flood_dist, 255); // �ѱ������
			img_flood = floodFill_four_corners(img_flood, flood_dist, 0); // �ѱ������
		}
		else if (area_flood < 2500) // �ж��Ƿ�ʣ��һ��СԲ��
		{
			img_flood = ~img_flood; // ȡ��,�����������
			cv::medianBlur(img_flood, img_flood, 3); // ��ֵ�˲�ȥ����
		}
		else
		{
			img_flood = floodFill_four_corners(img_flood, flood_dist, 0); // �ѱ������
		}
		cv::imwrite(closer + file, img_close);


		// ��ȡ��������
		vector<vector<cv::Point>> cou_cen;//��������
		cv::Point circle_mid;
		cv::Mat img_cou = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		cv::Mat e_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); // ����ˡ�
		cv::Mat imgoral = img.clone();
		cv::bitwise_not(img_flood, img_flood); // ������
		cv::findContours(img_flood, cou_cen, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // ������
		sort(cou_cen.begin(), cou_cen.end(), ascendSort); // ��������������
		int i, j = 0;
		for (i = 0; i < cou_cen.size(); ++i)
		{
			//ȥ��������̫�ٵĹ���������
			if (cou_cen[i].size() > 50)
			{
				//�ж������ĵ�һ�����Ƿ�����ԱȽ����ĵĵط�����ֹ�ɵ�һЩ����ֵֹĶ���
				if ((cou_cen[i][0].x > 50) && (cou_cen[i][0].x < img_gray.cols - 50)) // ��
				{
					if ((cou_cen[i][0].y > 50) && (cou_cen[i][0].y < img_gray.rows - 50)) // ��
					{
						cv::Mat img_cou = cv::Mat::zeros(img_gray.size(), CV_8UC1);
						cv::drawContours(img_cou, cou_cen, i, cv::Scalar(255), cv::FILLED);
						vector<cv::Vec3f> circles;
						cv::HoughCircles(img_cou, circles, cv::HOUGH_GRADIENT, 1, 200, 255, 11, 6, 43); // ��
						if (circles.size() != 0)
						{
							cv::Moments m = cv::moments(img_cou, true); // ��ȡ���ĵ���Ϣ
							circle_mid.x = m.m10 / m.m00; // �洢���ĵ�x����
							circle_mid.y = m.m01 / m.m00; // �洢���ĵ�y����
							cv::circle(imgoral, cv::Point(circle_mid), 2, cv::Scalar(0, 0, 255), -1); // ���Բ�ġ�
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
			printf("Unable to locate the center of Placido's disk\n"); // ���ִ���:�޷���λPlacido���̵�����
			//return 903; // ���ش���ֵ
			continue;
		}
		else if ((circle_mid.x == 0) || (circle_mid.y == 0))
		{
			printf("Central positioning error of Placido's disk\n"); // ���ִ���:Placido���̵����Ķ�λ����
			//return 904; // ���ش���ֵ
			continue;
		}



		/*
		* ȥ������
		* ԭ��
		* 1.�Զ�ֵͼ���в���������������̼���https://github.com/psurya1994/polar-scanning-algorithm
		* 2.ͨ�������ܹ���������λ��ֹͣ������ͼ�����껯���������㣬�����в�����ȫ��ճ������ȡ��mask
		* 3.����ͼ���ֵͼand�������õ�roi
		*/

		// �������ҳ�
		double pi = 3.1415926;
		cv::Mat img_B = img_close.clone(); // ��ȡ�߽�ͼ(B:get boundary)
		cv::Mat img_point = cv::Mat::zeros(img_gray.size(), CV_8UC1);
		int Samp_num = 300; // ���ϵĲ����������(Number of ring sampling points)��
		int Samp_rmax = (img_B.cols <= img_B.rows ? img_B.cols : img_B.rows); // �������뾶(Maximum sampling radius)��
		vector<cv::Point> B_point; // ��Ե��(B:boundary)
		vector<float> B_dist; // �߽�㵽���ĵľ���(B:boundary;dist:distance)
		close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::morphologyEx(img_B, img_B, cv::MORPH_CLOSE, close_kernel);
		cv::medianBlur(img_B, img_B, 3); // ��ֵ�˲�
		for (double theta = 0.0; theta < 360.0; (theta += 360 / Samp_num)) // �����Ƕ�
		{
			//nn����¼�����������������nn���ڵ���(nn_max = 16)��ʱ������ԸýǶȵĲ���
			int Samp_r = 4, nn = 0, nn_max = 16; // �����뾶(sampling radius)��
			double theta_rad = (theta * pi) / 180; // ת��Ϊ������
			cv::Point search_point = circle_mid + cv::Point(floor(Samp_r * cos(theta_rad)), floor(Samp_r * sin(theta_rad))); // �����������
			bool prev_spvalue = false; // ��һ����Ѱ���ֵ(previous search point value)
			bool inside_bounds = false; // �ж��������Ƿ���ͼ��(if a search given point is inside the bounds of the image.)
			do
			{
				if (nn >= nn_max)
					break;
				if (bool(img_B.at<uchar>(search_point)) != prev_spvalue) // �ж��Ƿ�����һ��������ͬ
				{
					B_point.push_back(search_point); // �����Ե��
					float dist = sqrt(pow((search_point.x - circle_mid.x), 2) 
						+ pow((search_point.y - circle_mid.y), 2)); // ����߽�㵽���ĵľ���
					B_dist.push_back(round(dist)); // ����߽�㵽���ĵľ���
					prev_spvalue = (!prev_spvalue); // ��ת��һ����Ѱ���ֵ
					Samp_r += 1; // ��������ٶ�,�����һ�����ؿ�(����ѡ��ɾ��)
					cv::circle(imgoral, cv::Point(search_point), 0, cv::Scalar(0, 255, 255), -1);
					cv::circle(img_point, cv::Point(search_point), 0, 255, -1);
					nn = 0;
				}
				else
				{
					++nn;
				}
				Samp_r += 1; // �뾶������
				search_point = circle_mid + cv::Point(floor(Samp_r * cos(theta_rad)), floor(Samp_r * sin(theta_rad))); // ��ȡ��һ��������
				inside_bounds = inside_bounds = ((search_point.x < img_B.cols) && (search_point.x >= 0)
					&& (search_point.y < img_B.rows) && (search_point.y >= 0)); // �����ж���һ�����ص��Ƿ���ͼ��

			} while (inside_bounds); // �����ýǶȵĴ����ĵ��߽������
		}
		cv::Mat cl_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); // ��
		for (i = 0; i < 10; ++i) // ��
			cv::morphologyEx(img_point, img_point, cv::MORPH_CLOSE, cl_kernel);
		cv::medianBlur(img_point, img_point, 3); // ��
		cv::Mat img_roi;
		cv::bitwise_and(img_point, img_close, img_roi);
		img_roi = bwareaopen(img_roi, 20); // ��
		cv::imwrite(scanr + file, imgoral);
		cv::imwrite(imgpr + file, img_point);
		cv::imwrite(roir + file, img_roi);



		// ������任
		cv::Mat polar, invpolar, polar_point, toph, polar_gradx, polar_gradxi, polar_gradxe;
		cv::warpPolar(img_roi, polar, cv::Size(img_gray.cols * 2, img_gray.rows * 2), circle_mid,
			circle_mid.x * 0.85, cv::INTER_LINEAR + cv::WARP_POLAR_LINEAR); // ������ת���뾶����̫��̫��Ὣ��ԵҲŪ���� ��
		cv::medianBlur(polar, polar, 3); // ��
		cv::threshold(polar, polar, 206, 255, cv::THRESH_BINARY); // ȥ��������Բ��
		cv::imwrite(polarr + file, polar);



		// ������
		cv::Mat polar_thin, polar_thin_contours;
		thin_getPoints(polar.clone(), polar_thin);
		cv::imwrite(basic_path +"thino/o" + file, polar_thin);



		// ɾ��������Ϣ��Ե
		cv::Matx33d la_kernel(1,0,-1,2,0,-2,1,0,-1); // ��
		//cv::Matx12d la_kernel(-1,1);
		cv::Mat polar_thin_=polar_thin.clone(), la_kernel_t,polar_thin__= polar_thin.clone();
		//cv::transpose(la_kernel, la_kernel_t);
		//cv::Matx33d la_kernel(-1,1,0,0,0,0,0,0,0);
		cv::filter2D(polar_thin.clone(), polar_thin, -1, la_kernel);
		//polar_thin__ = polar_thin.clone();
		//cv::filter2D(polar_thin, polar_thin, -1, la_kernel);
		//polar_thin = bwareaopen(polar_thin, 35);// ��
		polar_thin = bwareaopen(polar_thin, 20);
		polar_thin__ = polar_thin.clone();
		cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
		cv::morphologyEx(polar_thin,polar_thin,cv::MORPH_CLOSE,kernel_open);
		//cv::subtract(polar_thin,polar_thin_,polar_thin_);
		cv::imwrite(tr + file , polar_thin);



		// ���������
		vector<vector<cv::Point>> ring_set;
		vector<Point> v_now; // ���浱ǰ��������
		vector<vector<int>> ring_xdelta;
		vector<int> v_xdelta;
		int x = 5, y = 0; // x,y����
		int x_dist_max = 90; // �һ������Χ
		int x_before = 0; // ǰһ��ֱ�ߵ���������
		int dist_max = 20; // �ߵ�����֮���������̷�Χ.
		int polar_thin_rows = polar_thin.rows; // ��ȡͼƬ���
		int polar_thin_cols = polar_thin.cols; // ��ȡͼƬ���
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
					cv::floodFill(polar_flood, cv::Point(x, y), cv::Scalar(0), 0, cv::Scalar(), cv::Scalar(), 8); // ɾ����ǰ����
					cv::bitwise_xor(polar_flood, polar_thin.clone(), polar_flood); // ����������ȡ�û�
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
		//����
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
			//��ɨ�������
			v_now_ = drop_next_ring(v_now,ring_set[i-1]);
			v_now_.swap(v_now);
			rescan:
			cout << "ring "<< i <<"  :  " << v_now.size() << endl;
			//�����ڵ�8����֮�ڵĲ���ɨ������ĵ���������502*0.65=326�����ʱ����Ϊ����ͼ�յ�Ӱ��̫�󣬽�������
			//ע�� ����۽�ëӰ�쵽̫����Ļ�����ô����Ļ�һ��������ܵ���Ӱ���������Խ���ȥ����
			if (i < 9 && v_now.size() < int(0.65 * polar_thin_rows))
			{
				cout << "the count of this ring: " << i << " is too small" << endl;
				/*
				* ���ĳ���ӿ�
				*/
				break;
			}
			//����̫���ˣ������ܲ����������502*0.3����
			else if (i>=9 && v_now.size() < int(0.3* polar_thin_rows))
			{
				int num = 0, ii=0;
				vector<cv::Point> ::iterator v_now_beg = v_now.begin(), v_now_end = v_now.end();
				//ͳ�Ƶ�����̫�ٵ�ʱ�򣬿���ǰ300����֮ǰ��û����ԱȽ���������
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
					//�������ϰ�ε�������180���㣬���зֶζ���ʽ���
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
			//�������������ֹ��ٵ����
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
					//����400ֱ���ö���ʽ���
					//��ɨ������ĵ�һ���������������Ǵ���25��ʱ�򣬾��ÿ�ͷ�˵�x�����õ�һ��x������ֱ�ӽ������
					cv::Mat AA = polyfit(p, 8);
					for (int ii = 0; ii < polar_thin.rows; ii++)
					{
						int vv;
						//��ɨ��ĵ�0�����y���ڵ���25���Ҵ���ϵĵ��yС��25
						if ((v_now[0].y >= 25) && (ii<=25))
							vv = v_now[0].x;
						
						else
							vv = int(polycaulate(double(ii), AA));
						fit_line.push_back(cv::Point(vv, ii));
					}
				}
				else if (v_now.size() < 400)
				{
					//��ǰ�����һ����֮���y��ֵ
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
					//�������˳���ļ����˵��������ܵ�����������Ӱ�죬
					//���Ӱ�����ص�С�ڵ���150��ʱ����ΪӰ����ԱȽ��٣�����ֱ�Ӷ����������ֶ����
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
						//last_�����֮��ʣ�¶��ٸ���
						//���ʣ�µĵ����Ǵ��ڵ���50����Ե�ǰ�����ϰ�εĵ���°��ʣ��ĵ�ֱ������
						//���ʣ�µĵ�С��50������ôֻ���ϰ�εĵ������
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
								//�������ʣ��ĵ�����һ��ʣ������502���̫̫Զ��ǿ����ϻ���������������Զ����������ֻ��ϵ�
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
				//���������v_now����502����ô˵��ɨ�赽����һ����
				else if (v_now.size() >=502)
				{
					int max_delta = 0;
					//�����ɨ�赽�ĵ�ĵ���Ϊ502��ʱ�򣬿��ܳ�������������һ���������ߣ�����ɨ�赽�˲�����һ����
					if (v_now.size() == 502)
					{
						//max_delta��ָǰ���������y����Ĳ�ֵ�����ֵ������������������ģ���ô������϶��������1
						for (int ii = 0; ii < (v_now.size() - 2); ii++)
						{
							if ((v_now[ii + 1].y - v_now[ii].y) > max_delta)
							{
								max_delta = (v_now[ii + 1].y - v_now[ii].y);
								//�������������Ǵ��ڵ���2�ģ���˵��������
								if (max_delta >= 2)
								{
									break;
								}
							}
						}
						//���ɨ�赽ǰ������������������Ǵ���1����˵����502
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
				//����
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
