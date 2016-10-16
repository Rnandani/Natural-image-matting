#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat boxfilter(Mat imSrc, int r)
{
	int hei=imSrc.rows; //i
	int wid=imSrc.cols; //j
	Mat imDst=Mat::zeros(imSrc.size(),imSrc.type());
	
	//cumulative sum over Y axis
	Mat imCum1=Mat::zeros(imSrc.size(),imSrc.type());
	for (int j=0;j<wid;j++)
	{
		imCum1.at<uchar>(0,j)=imSrc.at<uchar>(0,j);
	
	}
	
	for (int i=1;i<hei;i++)
	{
		for (int j=0;j<wid;j++)
		{
			imCum1.at<uchar>(i,j)=imSrc.at<uchar>(i,j)+imCum1.at<uchar>(i-1,j);
			
		}
	}
		
	//difference over Y axis
	for (int i=0;i<r+1;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<uchar>(i,j)=imCum1.at<uchar>(i+r,j);
		}
	}

	for (int i=r+1;i<hei-r;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<uchar>(i,j)=imCum1.at<uchar>(i+r,j)-imCum1.at<uchar>(i-r-1,j);
		}
	}

	for (int i=hei-r;i<hei;i++)
	{
		for (int j=0;j<imSrc.cols;j++)
		{
			imDst.at<uchar>(i,j)=imCum1.at<uchar>(hei-1,j)-imCum1.at<uchar>(i-r-1,j);
		}
	}
	

	//cumulative sum over X axis
	Mat imCum2=Mat::zeros(imSrc.size(),imSrc.type());
	for (int i=0;i<imSrc.rows;i++)
	{
		imCum2.at<uchar>(i,0)=imDst.at<uchar>(i,0);
	}
	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=1;j<imSrc.cols;j++)
		{
			imCum2.at<uchar>(i,j)=imDst.at<uchar>(i,j)+imCum2.at<uchar>(i,j-1);
		}
	}

	//difference over X axis
	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=0;j<r+1;j++)
		{
			imDst.at<uchar>(i,j)=imCum2.at<uchar>(i,j+r);
		}
	}

	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=r+1;j<wid-r;j++)
		{
			imDst.at<uchar>(i,j)=imCum2.at<uchar>(i,j+r)-imCum2.at<uchar>(i,j-r-1);
		}
	}

	for (int i=0;i<imSrc.rows;i++)
	{
		for (int j=wid-r;j<wid;j++)
		{
			imDst.at<uchar>(i,j)=imCum2.at<uchar>(i,wid-1)-imCum2.at<uchar>(i,j-r-1);
		}
	}
	return imDst;
	
}


Mat getLaplacian(Mat I, int r)
{
	double eps = 0.0000001f;
	int h=I.rows;
	int w=I.cols;
	int c =I.channels();
	int wr;
	Mat meanI_r, meanI_g, meanI_b, chan1box, chan2box, chan3box;
	Mat rr, rg, rb, gg, gb, bb, RR, RG, RB, GG, GB, BB, varI_rr, varI_rg, varI_rb, varI_gg, varI_gb, varI_bb, A1, A2;	
	
	Mat N = boxfilter(Mat::ones(I.size(),CV_8UC1), r);

	// Split the image into different channels
    vector<Mat> channel(3);
	split(I, channel);

	Mat ch1, ch2,ch3;
	ch1 = channel[0];  //Blue channel
	ch2 = channel[1];  //Green channel
	ch3 = channel[2];  //Red channel

    chan1box = boxfilter(ch3, r);
	divide(chan1box, N, meanI_r, 1, -1);
	chan2box = boxfilter(ch2, r);
	divide(chan2box, N, meanI_g, 1, -1);
	chan3box = boxfilter(ch1, r);
	divide(chan3box, N, meanI_b, 1, -1);

	//Variance of I in each local patch
	//		rr, rg, rb
	//		rg, gg, gb
	//		rb, gb, bb
	
	multiply(ch3, ch3, rr);
	RR = boxfilter(rr, r);
	multiply(ch3, ch2, rg);
	RG = boxfilter(rg, r);	
	multiply(ch3, ch1, rb);
	RB = boxfilter(rb, r);
	multiply(ch2, ch2, gg);
	GG = boxfilter(gg, r);
	multiply(ch2, ch1, gb);
	GB = boxfilter(gb, r);
	multiply(ch1, ch1, bb);
	BB = boxfilter(bb, r);

	divide(RR, N, A1, 1, -1);
	subtract(meanI_r, meanI_r, A2);
	subtract(A1, A2, varI_rr);

	divide(RG, N, A1, 1, -1);
	subtract(meanI_r, meanI_g, A2);
	subtract(A1, A2, varI_rg);

	divide(RB, N, A1, 1, -1);
	subtract(meanI_r, meanI_b, A2);
	subtract(A1, A2, varI_rb);

	divide(GG, N, A1, 1);
	subtract(meanI_g, meanI_g, A2);
	subtract(A1, A2, varI_gg);

	divide(GB, N, A1, 1, -1);
	subtract(meanI_g, meanI_b, A2);
	subtract(A1, A2, varI_gb);

	divide(BB, N, A1, 1, -1);
	subtract(meanI_b, meanI_b, A2);
	subtract(A1, A2, varI_bb);

	
	double tlen;
	wr = (2*r+1) * (2*r+1);
	Mat cc = Mat::zeros(I.size(), CV_8UC1);
	Mat cc_t = Mat::zeros((h+r),(w-r), CV_8UC1);
	Mat Ident = Mat::eye((I.rows+r), (I.cols-r), CV_8UC1);
	Mat temp_cc;
	subtract(Ident, cc_t, temp_cc);	
	double s = cv::sum(temp_cc )[0];
	tlen = s * wr * wr;

	Mat M_idx = Mat::Mat(h, w, CV_8UC1); // this is in parallel to MATLAB code 'M_idx = reshape([1:h*w], h, w)'
	Mat vals = Mat::zeros(tlen,1,CV_8UC1);
	int len = 0;
	Mat sigma, meanI;
	Mat Identity = Mat::eye(3, 3, CV_8UC1);

	//---code debugged and it is correct!---//

	/*
	for (int i=1+r; i<h-r; i++)	
	{
		for (int j=1+r; j<w-r; j++)
		{
			
			sigma = (Mat_<double>(3,3) << varI_rr.at<double>(i,j), varI_rg.at<double>(i,j), varI_rb.at<double>(i,j), varI_rg.at<double>(i,j), varI_gg.at<double>(i,j), varI_gb.at<double>(i,j), varI_rb.at<double>(i,j), varI_gb.at<double>(i,j), varI_bb.at<double>(i,j));
			meanI = (Mat_<double>(1,3) << meanI_r.at<double>(i,j), meanI_g.at<double>(i,j), meanI_b.at<double>(i,j));
			scaleAdd(Identity, eps, sigma, sigma);
			/*
			//win_idx = M_idx(y-r:y+r,x-r:x+r);
			Mat win_idx = Mat::Mat(2*r,2*r, CV_8UC1);
			for (int p = j-r; p<j+r; p++)
			{
				for (int q = i-r; q<i+r; q++)
				{
					win_idx.at<double>(p,q) = M_idx.at<double>(p,q);							
				}			
			}
			
			Mat win_idx = win_idx.reshape(1, win_idx.rows*win_idx.cols);
			Mat winI = Mat::zeros(I.size(), CV_8UC3);
			
			//winI=I(y-r:y+r,x-r:x+r,:);
			for (int p = j-r; p<j+r; p++)
			{
				for (int q = i-r; q<i-r; q++)
				{
					winI.at<double>(p,q) = I.at<double>(p,q);													
				}
			}
			
			//winI=reshape(winI,wr,c);
			Mat winI = winI.reshape(wr, c);

			//winI=winI-meanI(ones(wr,1),:);
			
			Mat meani = Mat::ones(wr, meanI.cols, CV_8U);
			for(int j=0; j<meani.cols; j++)
			{
				for(int i=0; i<meani.rows; i++)
				{
					meani.at<double>(i,0) = meanI.at<double>(0,j);
				}
			}
			subtract(winI,meani, winI);
			/*
			//tvals=(1+winI*inv(Sigma)*winI')/wr;
			for (int p = (1+len),int q = 1;p<(wr*wr+len), q<tvals.rows;p++,q++)
			{
				vals.at<double>(p) = tvals.at<double>(q);
			}

			len = len + wr * wr;
			*/
			
		//}	
	//}
		
	return I;	
}



int main( int argc, const char** argv )
{
    Mat image, outputbox;
	int r = 1;
	image = imread("1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

	outputbox = getLaplacian(image, r);

	namedWindow( "Display output", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display output", outputbox );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
