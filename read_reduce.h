#include "projectHeader.h"

void readMotionFeat(int start, int end, string featname,string dirname,cv::Mat &allfeatures, int reduceMax);
void readAllMotionFeatures(cv::Mat &allfeatures,string featurename,string shotfilename,string dirname,int num_examples,int startvidn,int endvidn, int reduceMax);
void readallCombined1Features(cv::Mat &allfeatures,string featurename,string shotfilename,string dirname,int num_examples,int startvidn,int endvidn, int reduceMax);
void readCombined2CodeWords(cv::Mat &codes, cv::Mat &labels, string inputcodefile1,string inputcodefile2,string inputcodefile3,int num_codewords);
void read_reduce_baseline_desc(string fileName,cv::Mat &allfeatures,int maxFeat,string filedirloc);
void readExtFeat(string featfilename,cv::Mat &allfeatures, int reduceMax);

