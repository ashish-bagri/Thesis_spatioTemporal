#include "projectHeader.h"
#include "projectFunctions.h"
#include "read_reduce.h"

void read_reduce_kmeans_motion(char* video, int maxFeat);
void read_reduce_kmeans_combined1(char* video, int maxFeat);


double* doKmeans(cv::Mat& features,cv::Mat& dictionary,int dictionarySize,cv::Mat& labels);
void writeDictionary(cv::Mat &dictionary, int dictionarySize,int flags,double avg_distance,string featurename);


void read_reduce_youtube();
