#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <algorithm>
#include <opencv2/core/eigen.hpp>
#include <time.h>
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>

using namespace cv;
using namespace std;
using namespace Eigen;

void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);
    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }
    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<double>(i,j)<<"\t";
        }
        fout<<endl;
    }
    fout.close();
}

SparseMatrix<double> spdiags(const MatrixXd& B, const VectorXd& d, int m, int n)
{
	SparseMatrix<double> A(m,n);
	typedef Triplet<double> T;
    vector<T> triplets;
    triplets.reserve(std::min(m,n)*d.size());
    int l=0;
    for (int k = 0; k < d.size(); k++) 
	{
       int i_min = std::max(double(0), -1*d(k));
       int i_max = std::min(double((m - 1)), (n - d(k) - 1));
       int B_idx_start = m >= n ? d(k) : 0; 
       for (int i = i_min; i <= i_max; i++) 
	   { 
        triplets.push_back( T(i, i+k, B(B_idx_start + i, k)) );
       }
     }
     A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
   }

Mat boxfilter(Mat imSrc, int r)
{
	int hei=imSrc.rows; //i
	int wid=imSrc.cols; //j
	//imSrc.convertTo(imSrc, CV_64FC1);
	Mat imDst=Mat::zeros(imSrc.size(),imSrc.type());
	
	//cumulative sum over Y axis
	Mat imCum1=Mat::zeros(imSrc.size(),imSrc.type());
	for (int j=0;j<wid;j++)
	{
		imCum1.at<double>(0,j)=imSrc.at<double>(0,j);
	
	}
	
	for (int i=1;i<hei;i++)
	{
		for (int j=0;j<wid;j++)
		{
			imCum1.at<double>(i,j)=imSrc.at<double>(i,j)+imCum1.at<double>(i-1,j);
			
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

	//difference over X axis
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


SparseMatrix<double> getLaplacian(Mat I, int r)
{	
	double eps = 0.0000001f;
	int h=I.rows;
	int w=I.cols;
	int c =I.channels();
	int wr;
	double temp;
	Mat meanI_r, meanI_g, meanI_b, chan1box, chan2box, chan3box;
	Mat rr, rg, rb, gg, gb, bb, RR, RG, RB, GG, GB, BB, varI_rr, varI_rg, varI_rb, varI_gg, varI_gb, varI_bb, A1, A2;	
	
	Mat N = boxfilter(Mat::ones(I.size(),CV_64FC1), r);

	// Split the image into different channels
    vector<Mat> channel(3);
	split(I, channel);

	Mat ch1, ch2, ch3;
	ch1 = channel[0];  //Blue channel
	ch2 = channel[1];  //Green channel
	ch3 = channel[2];  //Red channel

	//meanI_r = boxfilter(I(:, :, 1), r) ./ N;
	//meanI_g = boxfilter(I(:, :, 2), r) ./ N;
	//meanI_b = boxfilter(I(:, :, 3), r) ./ N;
    chan1box = boxfilter(ch3, r);
	divide(chan1box, N, meanI_r, 1, -1);
	chan2box = boxfilter(ch2, r);
	divide(chan2box, N, meanI_g, 1, -1);
	chan3box = boxfilter(ch1, r);
	divide(chan3box, N, meanI_b, 1, -1);

	//Variance of I in each local patch
	//		  rr, rg, rb
	//Sigma = rg, gg, gb
	//		  rb, gb, bb
	
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
	multiply(meanI_r, meanI_r, A2);
	subtract(A1, A2, varI_rr);

	divide(RG, N, A1, 1, -1);
	multiply(meanI_r, meanI_g, A2);
	subtract(A1, A2, varI_rg);

	divide(RB, N, A1, 1, -1);
	multiply(meanI_r, meanI_b, A2);
	subtract(A1, A2, varI_rb);

	divide(GG, N, A1, 1);
	multiply(meanI_g, meanI_g, A2);
	subtract(A1, A2, varI_gg);

	divide(GB, N, A1, 1, -1);
	multiply(meanI_g, meanI_b, A2);
	subtract(A1, A2, varI_gb);

	divide(BB, N, A1, 1, -1);
	multiply(meanI_b, meanI_b, A2);
	subtract(A1, A2, varI_bb);

	
	double tlen;
	wr = (2*r+1) * (2*r+1);
	Mat cc = Mat::zeros(I.size(), CV_64FC1);

	//tlen=sum(sum(1-cc(r+1:end-r,r+1:end-r)))*(wr^2);
	Mat cc_t = Mat::zeros((h-2*r),(w-2*r), CV_64FC1);
	Mat Ident = Mat::ones((I.rows-2*r), (I.cols-2*r), CV_64FC1);
	Mat temp_cc;
	subtract(Ident, cc_t, temp_cc);	
	double s = sum(temp_cc)[0];
	tlen = s * wr * wr;

	//M_idx = reshape([1:h*w], h, w)
	Mat M_idx = Mat::zeros(h, w, CV_64FC1);
	int count =1;
	for (int gc=0; gc<M_idx.cols; gc++)
		for (int gr=0; gr<M_idx.rows; gr++)
		{
			M_idx.at<double>(gr,gc) = count;
			count= count+1;
		}
	 M_idx = M_idx.reshape(1, h);
	
	//vals=zeros(tlen,1);
	Mat vals = Mat::zeros(tlen,1,CV_64FC1);
	int len = 0;

	Mat sigma, meanI;
	Mat Identity = Mat::eye(3, 3, CV_64FC1);
	int i =0; int j =0; 
	int u, v, p, q;
	Mat Ident2 = Mat::ones(9, 9, CV_64FC1);
    Mat invsigma(3, 3, CV_64FC1);
	Mat row_idx = Mat::zeros(tlen,1, CV_64FC1);	
	Mat col_idx = Mat::zeros(tlen, 1, CV_64FC1);
	

	for (i=0+r; i<h-r; i++)	
	{
		for (j=0+r; j<w-r; j++)
		{
			sigma = (Mat_<double>(3,3) << varI_rr.at<double>(i,j), varI_rg.at<double>(i,j), varI_rb.at<double>(i,j), varI_rg.at<double>(i,j), varI_gg.at<double>(i,j), varI_gb.at<double>(i,j), varI_rb.at<double>(i,j), varI_gb.at<double>(i,j), varI_bb.at<double>(i,j));			
			meanI = (Mat_<double>(1,3) << meanI_r.at<double>(i,j), meanI_g.at<double>(i,j), meanI_b.at<double>(i,j));
			//Sigma = Sigma + eps * eye(3);
			scaleAdd(Identity, eps, sigma, sigma);
			
			//win_idx = M_idx(y-r:y+r,x-r:x+r);
			Mat win_idx = Mat::zeros(3,3, CV_64FC1);
						
			//int u, v, p, q;
			for (u=0, p=i-r; p<=(i+r); u++, p++)
			{
				for (v=0, q=j-r; q<=(j+r); v++, q++)
				{
					win_idx.at<double>(u,v) = M_idx.at<double>(p,q);
				}			
			}

			//win_idx=win_idx(:);
			win_idx = win_idx.t();
			win_idx = win_idx.reshape(1, win_idx.rows*win_idx.cols);
			Mat winI = Mat::zeros(c, c, CV_64FC3);
			
			//winI=I(y-r:y+r,x-r:x+r,:);
			for (u=0, p=i-r; p<=(i+r); u++, p++)
			{
				for (v=0, q=j-r; q<=(j+r); v++, q++)
				{
					winI.at<Vec3d>(u,v) = I.at<Vec3d>(p,q);						
				}		
			}	
			winI = winI.t();

			//winI=reshape(winI,wr,c);
			winI = winI.reshape(1, wr);
			for(u =0; u<winI.rows; u++)
			{
				temp = winI.at<double>(u,0);
				winI.at<double>(u,0) = winI.at<double>(u,2);
				winI.at<double>(u,2) =temp;
			}

			//winI=winI-meanI(ones(wr,1),:);			
			Mat meani = Mat::ones(wr, meanI.cols, CV_64FC1);			
			for( v=0; v<meani.cols; v++)
			{
				for( u=0; u<meani.rows; u++)
				{
					meani.at<double>(u,v) = meanI.at<double>(0,v);
				}
			}
			subtract(winI, meani, winI);	

			//tvals=(1+winI*inv(Sigma)*winI')/wr;
			Mat tvals = Mat::zeros(winI.size(), CV_64FC1);
			sigma.convertTo(sigma, CV_64FC1);
			winI.convertTo(winI, CV_64FC1);
			invert(sigma, invsigma, DECOMP_SVD);
			Mat temp2 = winI*invsigma*winI.t();
			tvals=(Ident2+winI*invsigma*winI.t())/wr;
					
			//row_idx(1+len:wr^2+len)=reshape(win_idx(:,ones(wr,1)),wr^2,1);
			Mat  win_idxt= Mat::zeros(wr, wr, CV_64FC1);			
			for( v=0; v<win_idxt.cols; v++)
			{
				for( u=0; u<win_idxt.rows; u++)
				{
					win_idxt.at<double>(v,u) = win_idx.at<double>(v,0);
				}
			}	
			win_idxt = win_idxt.reshape(1, wr*wr);
			
			for (v=0+len, u=0; v < wr*wr+len; v++, u++)
			{
				row_idx.at<double>(v,0) = win_idxt.at<double>(u,0);
			}
			//t = win_idx';
			Mat trans = win_idx.t();		
			Mat  trans_t= Mat::zeros(wr, wr, CV_64FC1);	

			//col_idx(1+len:wr^2+len)=reshape(t(ones(wr,1),:),wr^2,1);
			for( v=0; v<trans_t.cols; v++)
			{
				for( u=0; u<trans_t.rows; u++)
				{
					trans_t.at<double>(u,v) = trans.at<double>(0,v);
				}
			}
			trans_t = trans_t.reshape(1, wr*wr);
			for (v=0+len, u=0; v < wr*wr+len; v++, u++)
			{
				col_idx.at<double>(v,0) = trans_t.at<double>(u,0);
			}

			//vals(1+len:wr^2+len)=tvals(:);			
			tvals = tvals.reshape(1, wr*wr);			
			for (p=(0+len),q=0; q<tvals.rows; p++,q++)
			{
				vals.at<double>(p,0) = tvals.at<double>(q,0);
			}	
			//len=len+wr^2;
			len = len + (wr * wr);			
		}		
	}

	vals = vals.t();

	//L=sparse(row_idx,col_idx,vals,h*w,h*w);
	//Create a vector of triplets
	typedef Triplet<double> Trip;	
	vector<Trip> trp;
	//Create the triplets
	for (int vectorsize = 0; vectorsize < row_idx.rows; vectorsize++)
	{
		trp.push_back(Trip(row_idx.at<double>(vectorsize,0),col_idx.at<double>(vectorsize,0),vals.at<double>(0,vectorsize)));
	}
	//Assign them to the sparse Eigen matrix
	SparseMatrix<double> L(h*w+1, h*w+1);
	L.setFromTriplets(trp.begin(), trp.end());
	
	//sumL=sum(L,2);
	SparseMatrix<double> sumL(h*w+1,1);
	sumL = L * VectorXd::Ones(L.cols());

	//L=spdiags(sumL(:),0,h*w,h*w)-L;
	VectorXd d(1);d<<0;
	SparseMatrix<double> Lap(h*w+1, h*w+1);
	SparseMatrix<double> Lf(h*w+1, h*w+1);
	Lap = spdiags(sumL, d, h*w+1, h*w+1);
	Lf = Lap - L;
	return Lf;
}

int main( int argc, const char** argv )
{
    Mat image, sparseDmap, outputbox;
	int r = 1;
	clock_t t1,t2;
	t1=clock();
	image = imread("3.png", CV_LOAD_IMAGE_UNCHANGED);   // Read original image	
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
	
	sparseDmap = imread("3_sparse_depth.png", CV_LOAD_IMAGE_UNCHANGED); // Read sparse map
	if(! sparseDmap.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

	int h=image.rows;
	int w=image.cols;
	int c =image.channels();

	image.convertTo(image, CV_64FC3, 1.0/255.0);
	//cout<<"image:"<<image.at<Vec3d>(0,0)<<endl;
	sparseDmap.convertTo(sparseDmap, CV_64FC1,1.0/255.0);
	double theta = 0.0001f;
	double lambda = 1.0000e-03f;

	//constsMap=sparseDMap>0.0001;
	MatrixXd constsMap(h, w);
	for( int i=0; i<sparseDmap.rows; i++)
	{
		for(int j=0; j<sparseDmap.cols; j++)
		{
			if(sparseDmap.at<double>(i,j) > theta)
			{
				constsMap(i,j) = 1.0;				
			}
			else
			{
				constsMap(i,j) = 0.0;
			}
		}
	}

	MatrixXd sparseDM(h, w);
	for( int i=0; i<sparseDmap.rows; i++)
	{
		for(int j=0; j<sparseDmap.cols; j++)
		{
			sparseDM(i,j) = sparseDmap.at<double>(i,j);
		}
	}

	//L=getLaplacian(I,1); 
	SparseMatrix<double> Laplace(h*w+1, h*w+1);
	Laplace = getLaplacian(image, r);

	//make a sparse diagonal matrix necessary for matting process
	//D=spdiags(constsMap(:),0,sizeI,sizeI);
	VectorXd d(1);d<<0;
	Map<MatrixXd> M1(constsMap.data(), h*w+1,1);
	SparseMatrix<double> D(h*w+1, h*w+1);
	D = spdiags(M1, d, h*w+1, h*w+1);

	//x=(L+lambda*D)\(lambda*D*sparseDMap(:));
	//x = A\B where A is L+lambda*D and B is (lambda*D*sparseDMap(:))
	VectorXd x;
	SparseMatrix<double> A(h*w+1, h*w+1);
	VectorXd B(h*w+1, 1);
	Map<MatrixXd> M2( sparseDM.data(), h*w+1,1);
	A = Laplace + lambda*D;
	B = lambda*(D*M2);
	BiCGSTAB<SparseMatrix<double>, Eigen::IncompleteLUT<double> >  BCGST;
	BCGST.preconditioner().setDroptol(.001);
	BCGST.compute(A);
	x = BCGST.solve(B);
	Map<MatrixXd> fullDmap(x.data(), w, h);
	Mat fullDMapCV(fullDmap.rows(), fullDmap.cols(), CV_64FC1, fullDmap.data());
	fullDMapCV = fullDMapCV.t();
	//fullDMap=reshape(x,h,w);
	//f_d=imadjust(fullDMap);
	
	imwrite( "result1.jpg", fullDMapCV);
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
	namedWindow( "Display", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display", fullDMapCV );                   // Show our image inside it.
	t2=clock();
	float diff ((float)t2-(float)t1);
    cout<<diff<<endl;
    system ("pause");
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
