#include "projectFunctions.h"
#include "projectHeader.h"

using namespace std;
void split_N_sets(vector<cv::Mat>& codewords,vector<cv::Mat>& labels, int N,cv::Mat &allcodewords, cv::Mat &allLabels, string trainshotfile,int num_examples);
void crossValidationSVM(string featurename, int num_positive_codeswords, int num_negative_codeswords, int N);
void readcodewords(string inputcodefile, cv::Mat& codes,cv::Mat& labels,int num_codewords);
void doSVM(cv::Mat &training, cv::Mat& validation,cv::Mat& trainingLabels, cv::Mat& validationLabels, string featurename, vector<float> Cvalues, vector<string> cindxguide, cv::Mat& newaccuracies, int n);
void svmCrossVal(cv::Mat &alltraining,vector<cv:: Mat>& codewordsP,vector<cv::Mat>& labelsP ,vector<cv::Mat> &codewordsN, vector<cv::Mat>& labelsN, int N, string featurename,vector<float>& accuracies, float &best_C_value);
void shuffleCode_Label(cv::Mat& codewords, cv::Mat& labels,cv::Mat& shuffledcodes, cv::Mat& shuffledLabels);
void createAndWriteSvm(cv::Mat& alltraining, cv::Mat& allLabels,string featurename, float Cvalue, bool wantmetoshuffle);
void getUniqueRandomNumbers(vector<int>& randomnumbers, int start, int end, int num_r);
void getCvalueVectors(vector<float>& Cvalues, vector<string>& cindxguide, cv::Mat &alltraining);
double initial_C_estimate(cv::Mat& training);

void readCombine2codewords_baseline(cv::Mat &codes, cv::Mat &labels, vector<string> &codesfilename,int num_codewords);


void readHist_svm_combined2_baseline_tiger();
void readHist_svm_combined2_baseline_leopard();


void readHist_SVM_tiger();

void readHist_SVM_leopard();

void readHist_svm_combined2();
void readHist_getSVM_youtube();
void SVM_youtube();
