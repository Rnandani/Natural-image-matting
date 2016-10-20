#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using namespace cv;
using namespace std;

Mat boxfilter(Mat imSrc, int r)
{
	int hei=imSrc.rows; //i
	int wid=imSrc.cols; //j
	imSrc.convertTo(imSrc, CV_64FC1);
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


Mat getLaplacian(Mat I, int r)
{
	
	double eps = 0.0000001f;
	int h=I.rows;
	int w=I.cols;
	int c =I.channels();
	int wr;
	Mat meanI_r, meanI_g, meanI_b, chan1box, chan2box, chan3box;
	Mat rr, rg, rb, gg, gb, bb, RR, RG, RB, GG, GB, BB, varI_rr, varI_rg, varI_rb, varI_gg, varI_gb, varI_bb, A1, A2;	
	
	Mat N = boxfilter(Mat::ones(I.size(),CV_64FC1), r);

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
	Mat cc = Mat::zeros(I.size(), CV_64FC1);

	//tlen=sum(sum(1-cc(r+1:end-r,r+1:end-r)))*(wr^2);
	Mat cc_t = Mat::zeros((h-2*r),(w-2*r), CV_64FC1);
	Mat Ident = Mat::ones((I.rows-2*r), (I.cols-2*r), CV_64FC1);
	Mat temp_cc;
	subtract(Ident, cc_t, temp_cc);	
	double s = sum(temp_cc )[0];
	tlen = s * wr * wr;

	//M_idx = reshape([1:h*w], h, w)
	Mat M_idx = Mat::zeros(1, h*w, CV_64FC1);
	int count =1;
	for (int g=0; g<M_idx.cols; g++)
	{
		M_idx.at<double>(0,g) = count;
		count= count+1;
	}
	 M_idx = M_idx.reshape(1, h);

	Mat vals = Mat::zeros(tlen,1,CV_64FC1);
	int len = 0;
	Mat sigma, meanI;
	Mat Identity = Mat::eye(3, 3, CV_64FC1);
	int i =0; int j =0; 

	for (i=1+r; i<=h-r; i++)	
	{
		for (j=1+r; j<=w-r; j++)
		{
			cout<<"j:"<<j<<endl;
			sigma = (Mat_<double>(3,3) << varI_rr.at<double>(i,j), varI_rg.at<double>(i,j), varI_rb.at<double>(i,j), varI_rg.at<double>(i,j), varI_gg.at<double>(i,j), varI_gb.at<double>(i,j), varI_rb.at<double>(i,j), varI_gb.at<double>(i,j), varI_bb.at<double>(i,j));
			meanI = (Mat_<double>(1,3) << meanI_r.at<double>(i,j), meanI_g.at<double>(i,j), meanI_b.at<double>(i,j));
			scaleAdd(Identity, eps, sigma, sigma);
			
			
			//win_idx = M_idx(y-r:y+r,x-r:x+r);
			Mat win_idx = Mat::zeros(3,3, CV_64FC1);
			
			int u, v, p, q;
			for (u=0, p=i-r; p<(i+r); u++, p++)
			{
				for (v=0, q=j-r; q<(j+r); v++, q++)
				{
					win_idx.at<double>(u,v) = M_idx.at<double>(p,q);
				}			
			}

			//win_idx=win_idx(:);
			win_idx = win_idx.reshape(1, win_idx.rows*win_idx.cols);
			
			Mat winI = Mat::zeros(c, c, CV_64FC3);
			
			//winI=I(y-r:y+r,x-r:x+r,:);
			for (u=0, p=i-r; p<(i+r); u++, p++)
			{
				for (v=0, q=j-r; q<(j+r); v++, q++)
				{
					winI.at<double>(u,v) = I.at<double>(p,q);					
				}		
			}	
			
			//winI=reshape(winI,wr,c);
			winI = winI.reshape(1, wr);

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
			Mat Ident2 = Mat::eye(9, 9, CV_64FC1);
			Mat invsigma(3, 3, CV_64FC1);
			sigma.convertTo(sigma, CV_64FC1);
			winI.convertTo(winI, CV_64FC1);
			invert(sigma, invsigma, DECOMP_SVD);
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
			
			Mat row_idx = Mat::zeros(win_idxt.size(), CV_64FC1);

//---code debugged and it is correct!---//
			cout<<"found error";
			for (v=0+len; v < wr*wr+len; v++)
			{
				row_idx.at<double>(v,0) = win_idxt.at<double>(v,0);
			}

			//t = win_idx';
			Mat trans = win_idx.t();
			
			//col_idx(1+len:wr^2+len)=reshape(t(ones(wr,1),:),wr^2,1);
			Mat  trans_t= Mat::zeros(wr, wr, CV_64FC1);			
			for( v=0; v<trans_t.cols; v++)
			{
				for( u=0; u<trans_t.rows; u++)
				{
					trans_t.at<double>(u,v) = trans.at<double>(0,v);
				}
			}
			trans_t = trans_t.reshape(1, wr*wr);

			Mat col_idx = Mat::zeros(win_idxt.size(), CV_64FC1);

			for (v=0+len; v < wr*wr+len; v++)
			{
				col_idx.at<double>(v,0) = trans_t.at<double>(v,0);
			}

			//vals(1+len:wr^2+len)=tvals(:);			
			tvals = tvals.reshape(1, wr*wr);
			cout<<"vals size:"<<vals.size()<<"vals type"<<vals.type()<<endl;
			cout<<"tvals size:"<<tvals.size()<<"tvals type"<<tvals.type()<<endl;
			
			for (p=(0+len),q=0; q<tvals.rows; p++,q++)
			{
				vals.at<double>(p,0) = tvals.at<double>(q,0);
			}
		
			//len=len+wr^2;
			len = len + (wr * wr);
			cout<<len;					
		}	
		cout<<"i:"<<i<<"j:"<<j<<endl;

	}

	return I;	
}

/*
void spdiags(Eigen::SparseMatrix<double> A)
{

    //Extraction of the diagnols before the main diagonal
   vector<double> vec1; int flag=0;int l=0;
   int i=0; int j=0; vector<vector<double> > diagD;
   vector<vector<double> > diagG;  int z=0; int z1=0;


   for(int i=0;i<A.rows();i++)
   {l=i;
       do
       {
           if(A.coeff(l,j)!=0)
           flag=1;

           vec1.push_back(A.coeff(l,j));
           l++;j++;

       }while(l<A.rows() && j<A.cols());

       if(flag==1) {diagG.resize(diagG.size()+1);diagG[z]=vec1; z++; }
       vec1.clear(); l=0;j=0; flag=0; cout<<endl;
   }


   flag=0;z=0; vec1.clear();
  
    // Extraction of the diagonals after the main diagonal                                        
   for(int i=1;i<A.cols();i++)
   {l=i;
       do
       {
           if(A.coeff(j,l)!=0)
           flag=1; 

           vec1.push_back(A.coeff(j,l));
           l++;j++;

       }while(l<A.cols() && j<A.rows());

       if(flag==1) {diagD.resize(diagD.size()+1);diagD[z]=vec1; z++;  }
       vec1.clear(); l=0;j=0; flag=0; cout<<endl;
   }
// End extraction of the diagonals

Eigen::VectorXi d = Eigen::VectorXi::Zero(A.rows() + A.cols() - 1);

for (int k=0; k < A.outerSize(); ++k) 
{
  for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
   {
    d(it.col() - it.row() + A.rows() - 1) = 1;

   }
}


int num_diags = d.sum();
Eigen::MatrixXd B(std::min(A.cols(), A.rows()), num_diags);

// fill B with diagonals
Eigen::ArrayXd v;
int B_col_idx = 0;
int B_row_sign = A.rows() >= A.cols() ? 1 : -1;
int indG=diagG.size()-1; int indD=0;

for (int i = 1 - A.rows(); i <=A.cols() - 1; i++)
{
  if (d(i + A.rows() - 1))
  {
      if(i<1)
      {   v.resize(diagG[indG].size());
          for(int i=0;i<diagG[indG].size();i++)
          {
                  v(i)=diagG[indG][i];
          }

          int B_row_start = std::max(0, B_row_sign * i);
            B.block(B_row_start, B_col_idx, diagG[indG].size(), 1) = v;

          B_col_idx++;
          indG--;
      }
      else
      {
         v.resize(diagD[indD].size());

         for(int i=0;i<diagD[indD].size();i++)
         {
            v(i)=diagD[indD][i] ;
         }

          int B_row_start = std::max(0, B_row_sign * i);
          B.block(B_row_start, B_col_idx, diagD[indD].size(), 1) = v;

          B_col_idx++;
          indD++;
      }
  }

}
   cout<<B<<endl; //the result of the function
}//end of the function

*/

int main( int argc, const char** argv )
{
    Mat image, sparseDmap, outputbox;
	int r = 1;
	image = imread("1.jpg", CV_LOAD_IMAGE_UNCHANGED);   // Read original image	
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
	sparseDmap = imread("1_sparse_depth.png", CV_LOAD_IMAGE_UNCHANGED); // Read sparse map
	if(! sparseDmap.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
	image.convertTo(image, CV_64FC3);
	sparseDmap.convertTo(sparseDmap, CV_64FC1);
	double theta = 0.0001f;
	//constsMap=sparseDMap>0.0001;
	for( int i=0; i<sparseDmap.rows; i++)
	{
		for(int j=0; j<sparseDmap.cols; j++)
		{
			if(sparseDmap.at<double>(i,j) > theta)
			{
				sparseDmap.at<double>(i,j) =1;
			}
			else
			{
				sparseDmap.at<double>(i,j) =0;
			}
		}
	}
	//L=getLaplacian(I,1);
	const int dims = 2;
	int size[] = {image.rows*image.cols, image.rows*image.cols};
	SparseMat Laplace(dims, size, CV_64F);
	Laplace = getLaplacian(image, r);

	//make a sparse diagonal matrix necessary for matting process
	//D=spdiags(constsMap(:),0,sizeI,sizeI);
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

	//namedWindow( "Display output", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display output", outputbox );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
