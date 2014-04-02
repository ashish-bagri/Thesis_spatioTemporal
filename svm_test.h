#include "projectFunctions.h"
#include "projectHeader.h"
#include "readHist_SVM.h"

void svm_test_motion(string featurename);
void readTestCodeWords(cv::Mat &testcodewords, cv::Mat &testlabels, string testposfilename, string testnegfilename,int num_test_pos, int num_test_neg);
void doSVMTest(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels , vector<float>& accuracies,string featurename);

void readCode_DofusionPrediction_leopard(string featurename, int num_test_pos, int num_test_neg, vector<float>& margins);
void fusionPrediction(cv::Mat& testingcodes, cv::Mat& testingLabels,vector<float>& margins, string svmfilename);
void readCode_DofusionPrediction(string featurename, int num_test_pos, int num_test_neg, vector<float>& margins);

void svm_test_combine2_tiger();
void SVM_TEST_motion();
void SVM_TEST_combined1();
void svm_test_combine2_baseline_tiger();

void svm_test_combine2_leopard();
void svm_test_combine2_baseline_leopard();

void combine_3_leopard();
void combine_3_tiger();
void combine3_tiger_choose();
void combine3_motionOrBaseline();

void svm_motion_precision_recall(string);
void svmTest_threshold(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels,string featurename);

void svmtest_distancemargin(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels,string featurename);
void getSVMMargins(string featurename);
void SVM_test_youtube();
void SVM_test_youtubePR();
void SVM_test_youtube_mulCategory();
void svm_getMargins(string svmfilename, cv::Mat& testingcodes, vector<float>& margins);
void classificPrecisionRecall(vector<float>& groundTruth, vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<int>& ranking);
float crossValParam(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,string filename = "",string LabelMarginfile = "");
float svmTest_eachClass_PR(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,vector<pair<string, pair<int,int> > > cat_video_shot, string PRfilename, string LabelMarginfile);
float validationResult(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,vector<float> &results);


void SVM_test_singleClass_youtube();
