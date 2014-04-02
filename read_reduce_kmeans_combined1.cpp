#include "read_reduce_kmeans.h"
void read_reduce_kmeans_writedict_combined1( string featurename, string shotfilename,int num_examples, string dirname,int maxFeat);

void read_reduce_kmeans_combined1(char* video, int maxFeat)
{
    read_reduce_kmeans_writedict_combined1("combined1",train_pos_shotfilename,NUM_POS_TRAIN,"trainpos",maxFeat);
}

void read_reduce_kmeans_writedict_combined1( string featurename, string shotfilename,int num_examples, string dirname,int maxFeat)
{
    cv::Mat features = cvCreateMat(0,featureDimension_combined,CV_32FC1);

    readallCombined1Features(features,featurename,shotfilename,dirname,num_examples,0,num_examples,maxFeat);

    cout<<"Start k means "<<endl;
    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_combined,CV_32FC1);
    cv::Mat labels = cvCreateMat(features.rows,1,CV_32FC1);

    double* retkmeans =  doKmeans(features,dictionary,dictionarySize,labels);
    cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

    writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"combined1");
    cout<<"completed writing the dictionary"<<endl;
}
