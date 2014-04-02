#include "projectHeader.h"
int showflow(string imagefilelist, string dirToGetShot);

string threeFormatNumber(int number);
string fourFormatNumber(int number);
void posCatNames(vector<string> &poscategory);
void negCatNames(vector<string> &negcategory);
void extract_write_cat_features(vector<string> category, string train_test_filename);
void read_reduce_youtube_features(string pos_neg, string train_test, string featurename, cv::Mat& features, int maxFeat);
cv::Mat readFeatOfAShot_youtube(string filename, string featurename);

void extract_write_cat_features(vector<string> category, string train_test_filename, string shotsfilename, int (*featureextractor)(string,string,string));
int read_youtube_imagelist_getbaseline(string imagefilelist, string shotsdir, string savefilename);

cv::Mat readFeatofAshot_gethist_youtube(string featurefilename,cv::Mat &dictionary,int featDimension,int numOfFeatures = -1);

//cv::Mat readFeatofAshot_gethist_baseline_youtube(string imagefilelist, string dirToGetShot, string dirToSaveFeat, int featdimension,cv::Mat &dictionary);

void getVideoIndx_youtubedata(vector<int> & videoindx, string category, string train_test);
void getVideoIndx_youtubedata_sorted(vector<int> & videoindx, string category, string train_test);

//void readingFeat_readVSFileName(vector<int> &V, vector<int> &S,vector<float>& numFeatures,string category, string train_test);

void writingFeat_readVSFileName(vector<int> &V, vector<int> &S,string category, string train_test);

string getShotsDir_youtube(int videoNumber, int shotNumber, string category);
string getFeatureFileName(string shotsdir, int video, int shot);

string getBaselineFeatLocation(string category);
string getBaselineImageLocation(string category);

void doextract_write_youtube_baseline(vector<string> categories);
string getCodewordfilename_youtube(string category,int video,string featurename);
string getCodewordfilename_youtube_eachshot(string category,string featurename,int video,int shot);

void do_readfeat_gethist_youtube(string pos_neg, string train_test, string featurename, string shotsfilename);

string getBaselineFeatureFileName(string category, int video, int shot);
string getBaselineImageFileName(string category, int video, int shot);

int youtubeBaselineFeatures(string imagefilelist, string dirToGetShot, string dirToSaveFeat);

int youtubeBaselineFeatures(string imagefilelist, string dirToGetShot, string dirToSaveFeat, string dirToSaveImage);

cv::Mat read_youtubeBaselineFeatures(string imagefilelist, string dirToGetShot, string dirToSaveFeat);

cv::Mat read_youtubeBaselineFeatures(string featurefilename,int featDimension,int numOfFeatures = -1);

void readImagefileList(string imagefilelist,string dirToGetShot,vector<string> &imagelist);
string getRootDir();

void readHistogram_youtube(vector<string> category,vector<string> featurename,cv::Mat& codewords,string train_test);

//void readNegativeCategoryHistogram(int N, vector<string> category,string featurename,vector<cv::Mat>& codewordsN);
//void readPositiveCategoryHistogram(int N, string category,string featurename,vector<cv::Mat>& codewordsP);

void readNegativeCategoryHistogram_mulFeatures(int N, vector<string> category,vector<string> featurename,vector<cv::Mat>& codewordsN);
void readPositiveCategoryHistogram_mulFeatures(int N, string category,vector<string> featurename,vector<cv::Mat>& codewordsP);

void readHist_video_youtube(cv::Mat& histogram, int video,int numShots,string category, vector<string> featurename,vector<int> shots);
void readHist_video_youtube(cv::Mat& histogram, int video,int numShots,string category, string featurename);
void readHist_video_youtube(cv::Mat& histogram, string category, vector<string> featurename,int video,vector<int> shots);

const int positiveLabelValue = 1;
const int negativeLabelValue = -1;
string getFeatureFileLocation(string category, int video, int shot);

string getMotionFeatureFileLocation(string category, int video, int shot);
string getMotionFeatLocation(string category);
string getMotionFeatureFileName(string category, int video, int shot,string featurename);
