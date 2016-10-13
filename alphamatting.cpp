#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <deque>  //"double-ended queue- It is dynamic array that is implemented so that it can grow in both the directions"
#include <cmath>


using namespace cv;
using namespace std;


Mat boxfilter(Mat imSrc, int r)
{	
	int hei=imSrc.rows;
	int wid=imSrc.cols;
	cout<<hei<<endl<<wid<<endl;;
	
	
	Mat imDst = Mat::zeros(imSrc.size(),imSrc.type());

	//cumulative sum over Y axis

	Mat imCum1=Mat::zeros(imSrc.size(),imSrc.type());
	
	for (int i=0;i<hei;i++)
	{

		imCum1.at<double>(Point(i,0)) = imSrc.at<double>(Point(i,0));

	}
	
	for (int j=1;j<wid;j++)
	{
		for (int i=0;i<hei;i++)
		{
			
			imCum1.at<double>(Point(i,j)) = imSrc.at<double>(Point(i,j)) + imCum1.at<double>(Point(i-1,j));

		}
						
	}

	//difference over Y axis
	for (int i=0;i<r+1;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<double>(i,j)=imCum1.at<double>(i+r,j);
		}
	}

	for (int i=r+1;i<hei-r;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<double>(i,j)=imCum1.at<double>(i+r,j)-imCum1.at<double>(i-r-1,j);
		}
	}

	for (int i=hei-r;i<hei;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<double>(i,j)=imCum1.at<double>(hei-1,j)-imCum1.at<double>(i-r-1,j);
		}
	}

	//cumulative sum over X axis
	Mat imCum2=Mat::zeros(imSrc.size(),imSrc.type());
	for (int i=0;i<imSrc.rows;i++)
	{
		imCum2.at<double>(i,0)=imDst.at<double>(i,0);
	}
	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=1;j<imSrc.cols;j++)
		{
			imCum2.at<double>(i,j)=imDst.at<double>(i,j)+imCum2.at<double>(i,j-1);
		}
	}
	
	//difference over Y axis
	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=0;j<r+1;j++)
		{
			imDst.at<double>(i,j)=imCum2.at<double>(i,j+r);
		}
	}

	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=r+1;j<wid-r;j++)
		{
			imDst.at<double>(i,j)=imCum2.at<double>(i,j+r)-imCum2.at<double>(i,j-r-1);
		}
	}

	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=wid-r;j<wid;j++)
		{
			imDst.at<double>(i,j)=imCum2.at<double>(i,wid-1)-imCum2.at<double>(i,j-r-1);
		}
	}
	
	return imDst;
}


Mat getLaplacian(Mat I, int r)
{
	double eps = 0.0000001f;
	int h=I.rows;
	int w=I.cols;

	int c = Mat::channels(I);
	int wr;
	Mat channel[3];
	Mat meanI_r, meanI_g, meanI_b, chan1box, chan2box, chan3box;
	Mat rr, rg, rb, gg, gb, bb, RR, RG, RB, GG, GB, BB, varI_rr, varI_rg, varI_rb, varI_gg, varI_gb, varI_bb, A1, A2;	
		
	Mat N = boxfilter(Mat::ones(I.size(),I.type()), r);
	spilt(I, channel);
	
	chan1box = boxfilter(channel[2], r);
	divide(chan1box, N, 1, meanI_r);
	chan2box = boxfilter(channel[1], r);
	divide(chan2box, N, 1, meanI_g);
	chan3box = boxfilter(channel[0], r);
	divide(chan3box, N, 1, meanI_b);

	//Variance of I in each local patch
	//		rr, rg, rb
	//		rg, gg, gb
	//		rb, gb, bb

	RR = boxfilter((multiply(channel[2], channel[2], rr)), r);
	RG = boxfilter((multiply(channel[2], channel[1], rg)), r);	
	RB = boxfilter((multiply(channel[2], channel[0], rb)), r);
	GG = boxfilter((multiply(channel[1], channel[1], gg)), r);
	GB = boxfilter((multiply(channel[1], channel[0], gb)), r);
	BB = boxfilter((multiply(channel[0], channel[0], bb)), r);

	subtract(divide(RR, N, 1, A1),subtract(meanI_r, meanI_r, A2), varI_rr);
	subtract(divide(RG, N, 1, A1),subtract(meanI_r, meanI_g, A2), varI_rg);
	subtract(divide(RB, N, 1, A1),subtract(meanI_r, meanI_b, A2), varI_rb);
	subtract(divide(GG, N, 1, A1),subtract(meanI_g, meanI_g, A2), varI_gg);
	subtract(divide(GB, N, 1, A1),subtract(meanI_g, meanI_b, A2), varI_gb);
	subtract(divide(BB, N, 1, A1),subtract(meanI_b, meanI_b, A2), varI_bb);
	
	signed int tlen;
	wr = (2*r+1) * (2*r+1);
	Mat cc = Mat::zeros(I.rows,I.cols);
	Mat cc_t = Mat::zeros((I.rows+r),(I.cols-r));
	Mat Identity = Mat::eye((I.rows+r), (I.cols-r), CV_8uc1);	
	subtract(Identity, cc_t, temp_cc)	 
	tlen = sum(temp_cc) * wr * wr;

	Mat M_idx = Mat::Mat(h, w, CV_8UC1); // this is in parallel to MATLAB code 'M_idx = reshape([1:h*w], h, w)'
	Mat vals = Mat::zeros(tlen,1);
	int len = 0;
	Mat sigma, meanI;
	Mat Identity = Mat::eye(3, 3, CV_8uc1);
	
	for (j = 1+r; j<h-r; j++)	
	{
		for (i = 1+r; i<w-r; i++)
		{
			//image.at<Vec3b>(Point(1,2))
			sigma = (Mat_<double>(3,3) << varI_rr.at<double>(j,i), varI_rg.at<double>(j,i), varI_rb.at<double>(j,i), varI_rg.at<double>(j,i), varI_gg.at<double>(j,i), varI_gb.at<double>(j,i), varI_rb.at<double>(j,i), varI_gb.at<double>(j,i), varI_bb.at<double>(j,i));

			meanI = (Mat_<double>(1,3) << meanI_r.at<double>(j,i), meanI_g.at<double>(j,i), meanI_b.at<double>(j,i));
			scaleAdd(Identity, eps, sigma, sigma);
			
			Mat win_idx = Mat::Mat(2*r,2*r);
			for (p = y-r; p<y+r; p++)
			{
				for (q = x-r; q<x+r; q++)
				{
					wind_idx.at<double>(p,q) = M_idx.at<double>(p,q);							
				}			
			}
			
			Mat wind_idx = wind_idx.reshape(1, p*q);
			Mat winI;
			
			for (p = y-r; p<y+r; p++)
			{
				for (q = x-r; q<x-r; q++)
				{
					for (ch = 0; ch<c; ch++ )	
						{				
							winI.at<double>(p,q,ch) = I.at<double>(p,q,ch);	
						}													
				}

			}
			
			Mat winI = winI.reshape(1, wr);
			meani = meanI
			subtract(winI,meanI, winI)


			

			 
			
			
			for (p = (1+len), q = 1;p<(wr*wr+len), q<tvals.rows;p++,q++)
			{
				vals.at<double>(p) = tvals.at<double>(q);
			}

			len = len + wr * wr;			
		}	
	}
	
	

	
}

int main(void)
{
	Mat image, sparse_map;	
	int r = 1;
	//image = imread("image.jpg", CV_LOAD_IMAGE_COLOR);	
    sparse_map = imread("sparse_map.png", CV_LOAD_IMAGE_UNCHANGED);
	resize(sparse_map, image, Size(), 0.2, 0.2);
	Mat afterbox = boxfilter(image, r);	
	//imwrite("afterbox.png", afterbox);

	return 0;
}
