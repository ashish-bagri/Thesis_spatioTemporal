#include "projectHeader.h"
using namespace std;
cv::Mat concatenateFeatures_reduce(cv::Mat &hog, cv::Mat& hof, cv::Mat &mbh,int maxFeatures);
int readInputShotsFile(string inputshotfilename,vector<int>& startframes,vector<int>& endframes,vector<int>& keyframes,vector<int>& sizes,int num_max);
cv::Mat reduceFeatures(cv::Mat& Feat, int maxFeatures);
void getImagenamelist(string shotfilename,string imagedir, int num_examples,vector<string> &imageList);

void extract_write_features();
void read_reduce_kmeans();
void readFeat_getHist();

void readHist_SVM();

void SVM_TEST();

void extract_write_baseline();

string getActualName(string filename);
void getImagenamelist(string shotfilename,string imagedir, int num_examples,vector<string> &imageList);


void readAllanalysis();


void combine_3();
void read_reduce_youtube();

void getallconfig(vector<string> &configurations);
void getparts(string fullname,vector<string> &elements);
void rewriteHist();
