#include "projectHeader.h"
#include "projectFunctions.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
extern "C" {
#include <vl/generic.h>
#include "dsift.h"
}
using namespace std;

void extract_write_motion_features(char* video);
void extract_write_combined1_features(char* video);
int doTracking(int start_frame_abs, int end_frame_abs,char* video, cv::Mat &hogFeat, cv::Mat &hofFeat, cv::Mat &mbhXYFeat,int keyframenum);
void write_feat_to_file_combined1(cv::Mat &feat, int start, int end, string featname, string dirname);
void write_feat_to_file_motion(cv::Mat &feat, int start, int end, string featname, string dirname);


void setFilterParams(VlDsiftFilter* filter_image);
void readImage_getDesc(string fileName, string dirname);
int getDSIFTdesc_writeToFile(IplImage* img, string descFilename,string keyPointFilename);

int doTracking_youtubeData_2(string imagefilelist, string dirToGetShot, string dirToSaveFeat);
int doTracking_youtubeData(string imagefilelist, string dirToGetShot, string dirToSaveFeat);
void extract_write_youtubeFeatures();
void extract_write_youtube_baseline();
