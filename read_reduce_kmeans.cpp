#include "read_reduce_kmeans.h"
int retries_random = 2;
int retries_pp = 1;
void read_reduce_kmeans()
{

    char* video = "/home/ashish/data/tigerfullavi.avi";
    int maxFeat = 5000;

    //read_reduce_kmeans_motion(video,maxFeat);
    read_reduce_kmeans_combined1(video, maxFeat);


}
int copyMat_float(cv::Mat& input, cv::Mat& output)
{
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            output.at<float>(i,j) = input.at<float>(i,j);
        }
    }
    return 0;
}

int copyMat_int(cv::Mat& input, cv::Mat& output)
{
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            output.at<int>(i,j) = input.at<int>(i,j);
        }
    }
    return 0;
}

int copyMat_float(cv::Mat & input, cv::Mat & output,int startrow,int endrow)
{
    for(int i=startrow; i<min(endrow,input.rows); i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            output.at<float>(i-startrow,j) = input.at<float>(i,j);
        }
    }
    return 0;
}

double* doKmeans(cv::Mat& features,cv::Mat& dictionary,int dictionarySize,cv::Mat& labels)
{
    cout<<"Inside k means"<<endl;
    cv::TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.0001);
    /*  int flags = cv::KMEANS_RANDOM_CENTERS;
      cv::Mat labels_random = cvCreateMat(features.rows,1,CV_32FC1);
      cv::Mat cluster_center_random = cvCreateMat(dictionarySize,features.cols,CV_32FC1);
      double min_random = cv::kmeans(features,dictionarySize,labels_random,tc,retries_random,flags,cluster_center_random);
      cout<<"The minimum error for random kmeans is "<<min_random<<endl;
    */

    int  flags = cv::KMEANS_PP_CENTERS;
    cv::Mat labels_kpp = cvCreateMat(features.rows,1,CV_16U);
    for(int i=0; i<labels_kpp.rows; i++)
    {
        labels_kpp.at<int>(i,0) = 0;
    }
    cv::Mat cluster_center_kmeanspp = cvCreateMat(dictionarySize,features.cols,CV_32FC1);
    double min_kpp = cv::kmeans(features,dictionarySize,labels_kpp,tc,retries_pp,flags,cluster_center_kmeanspp);
    cout<<"The minimum error for kmeans plus plus is "<<min_kpp<<endl;

    /*    string dictionaryname = "youtube_baseline_2000_insidetest";
        writeDictionary(cluster_center_kmeanspp,dictionarySize,cv::KMEANS_PP_CENTERS,min_kpp,dictionaryname);
    */
    ofstream labelfile("kmeansLabels_inside",ios::out);

    for(int i=0; i<labels_kpp.rows; i++)
    {
        for(int j=0; j<labels_kpp.cols; j++)
        {
            labelfile<<labels_kpp.at<int>(i,j)<<" ";
        }
        labelfile<<endl;
    }
    labelfile.close();

    copyMat_float(cluster_center_kmeanspp,dictionary);
    copyMat_int(labels_kpp,labels);
    double * retval = new double[2];
    retval[0] = cv::KMEANS_PP_CENTERS;
    retval[1] = min_kpp;
    return retval;
    /*    if(min_random > min_kpp)
        {
            cout<<"Kmeans plus plus is better."<<endl;
            copyMat_float(cluster_center_kmeanspp,dictionary);

            copyMat_float(labels_kpp,labels);

            double * retval = new double[2];

            retval[0] = cv::KMEANS_PP_CENTERS;
            retval[1] = min_kpp;
            return retval;
        }
        else
        {
            cout<<"Kmeans random is better."<<endl;
            copyMat_float(cluster_center_random,dictionary);
            copyMat_float(labels_random,labels);
            double * retval = new double[2];
            retval[0] = cv::KMEANS_RANDOM_CENTERS;
            retval[1] = min_random;
            return retval;
        }*/
}

void writeDictionary(cv::Mat &dictionary, int dictionarySize,int flags,double avg_distance,string featurename)
{
    char sizec[10];
    char flagc[10];

    sprintf(sizec,"%d",dictionarySize);
    sprintf(flagc,"%d",flags);

    string dictname = featurename + "_dictionary";

    ofstream outdict(dictname.c_str(),ios::out);


    cout<<"Feature"<<"\t"<<featurename<<endl;
    cout<<"dictionarySize"<<"\t"<<dictionarySize<<endl;
    if(flags = cv::KMEANS_PP_CENTERS)
    {
        cout<<"flags"<<"\t"<<"KMEANS_PP_CENTERS"<<endl;
    }
    else
    {
        cout<<"flags"<<"\t"<<"KMEANS_RANDOM_CENTERS"<<endl;
    }

    cout<<"The average miinimum error for kmeans obtained was "<<avg_distance<<endl;
    cout<<"The number of kmeans performed was:: Random center initialization "<<retries_random <<" times."<<endl;
    cout<<" Kmeans plus plus center initialization : "<<retries_pp<<" times"<<endl;

    int i,j;

    for(i=0; i<dictionary.rows-1; i++)
    {
        for(j=0; j<dictionary.cols; j++)
        {
            outdict<<dictionary.at<float>(i,j)<<"\t";
        }
        outdict<<endl;
    }
    for(j=0; j<dictionary.cols-1; j++)
    {
        outdict<<dictionary.at<float>(i,j)<<"\t";
    }
    outdict<<dictionary.at<float>(i,j);

    outdict.close();
}
