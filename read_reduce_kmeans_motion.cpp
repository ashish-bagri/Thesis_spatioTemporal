#include "read_reduce_kmeans.h"

void read_reduce_kmeans_writedict_motion( string featurename, string shotfilename,int num_examples, string dirname,int maxFeat);
void read_reduce_kmeans_motion(char* video, int maxFeat)
{
//    read_reduce_kmeans_writedict_motion("hog",train_pos_shotfilename,NUM_POS_TRAIN,"trainpos",maxFeat);
 //   read_reduce_kmeans_writedict_motion("hof",train_pos_shotfilename,NUM_POS_TRAIN,"trainpos",maxFeat);
    read_reduce_kmeans_writedict_motion("mbh",train_pos_shotfilename,NUM_POS_TRAIN,"trainpos",maxFeat);
}

void read_reduce_kmeans_writedict_motion( string featurename, string shotfilename,int num_examples, string dirname,int maxFeat)
{
    cout<<"Max feat "<<maxFeat<<endl;
    cout<<"featurename "<<featurename<<endl;

    if(featurename == "hog")
    {
        // do for hog
        cv::Mat hogfeatures = cvCreateMat(0,featureDimension_hog,CV_32FC1);
        readAllMotionFeatures(hogfeatures,"hog",shotfilename,dirname,num_examples,0,num_examples,maxFeat);

        cout<<"Start k means "<<endl;
        cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_hog,CV_32FC1);
        cv::Mat labels = cvCreateMat(hogfeatures.rows,1,CV_32FC1);

        double* retkmeans =  doKmeans(hogfeatures,dictionary,dictionarySize,labels);
        cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

        writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"hog");
        cout<<"completed writing the dictionary"<<endl;
    }
    else if(featurename == "hof")
    {
        // do for hog
        cv::Mat hoffeatures = cvCreateMat(0,featureDimension_hof,CV_32FC1);
        readAllMotionFeatures(hoffeatures,"hof",shotfilename,dirname,num_examples,0,num_examples,maxFeat);

        cout<<"Start k means "<<endl;
        cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_hof,CV_32FC1);
        cv::Mat labels = cvCreateMat(hoffeatures.rows,1,CV_32FC1);

        double* retkmeans =  doKmeans(hoffeatures,dictionary,dictionarySize,labels);
        cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

        writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"hof");
        cout<<"completed writing the dictionary"<<endl;
    }
    else if (featurename == "mbh")
    {
        cv::Mat mbhfeatures = cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);
        readAllMotionFeatures(mbhfeatures,"mbh",shotfilename,dirname,num_examples,0,num_examples,maxFeat);
        cout<<"Start k means "<<endl;
        cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_mbhXY,CV_32FC1);
        cv::Mat labels = cvCreateMat(mbhfeatures.rows,1,CV_32FC1);

        double* retkmeans =  doKmeans(mbhfeatures,dictionary,dictionarySize,labels);
        cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

        writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"mbh");
        cout<<"completed writing the dictionary"<<endl;

    }
    else
    {
        cout<<"Un recognized feature "<<featurename<<endl;
    }
}
