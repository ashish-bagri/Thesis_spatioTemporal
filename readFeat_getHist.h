#include "projectHeader.h"
#include "projectFunctions.h"
#include "read_reduce.h"

using namespace std;

void readFeat_getHist_motion();
void readFeat_getHist_combined1();

void readDictionary(cv::Mat &dictionary, string dictName);

int featureToCodeword(cv::Mat& feature, cv::Mat &dictionary, cv::Mat& codeword);
void getHist_motion(string featurename,int featureDim,string shotfilename,string dirname,int num_examples,int label);
void readFeat_getHist_motion_leopard();
void readFeat_getHist_baseline_leopard();
void readfeat_getHist_youtubedata();
