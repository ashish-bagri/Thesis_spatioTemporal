#include "readFeat_getHist.h"
using namespace cv;
void readFeat_getHist()
{
    readfeat_getHist_youtubedata();
}

int featureToCodeword(cv::Mat& feature, cv::Mat &dictionary, cv::Mat& codeword)
{
    cout<<"inside feature to codewords"<<endl;
    cv::BruteForceMatcher<L2<float> > matcher;       // Use FlannBasedMatcher matcher. It is better
    vector<DMatch> matches;
    matcher.match(feature,dictionary, matches);

    cout<<"Created a temp code."<<endl;

    for(int j=0; j<codeword.cols; j++)
    {
        codeword.at<float>(0,j) = 0;
    }
    int numFeat = feature.rows;

    cout<<"Number of features for this video & size of matches is "<<numFeat<<" "<<matches.size()<<endl;

    double L2norm = 0;
    for(int i=0; i<matches.size(); i++)
    {
        int indx = matches[i].trainIdx ;
        if(indx > codeword.cols )
            cout<<"Problem ! matched index is beyond the dictionary size"<<endl;
        codeword.at<float>(0,indx) = codeword.at<float>(0,indx) + 1.0f;//numFeat;
    }
    for(int i=0;i<codeword.cols;i++)
    {
        L2norm = codeword.at<float>(0,i) * codeword.at<float>(0,i)  + L2norm;
    }
    L2norm = sqrt(L2norm);
    cout<<"L2norm is "<<L2norm<<endl;

    for(int i=0;i<codeword.cols;i++)
    {
        codeword.at<float>(0,i) = codeword.at<float>(0,i) / L2norm;
    }

/*    //checking normalization
    double check = 0;
    for(int i=0;i<codeword.cols;i++)
    {
        check = check + (codeword.at<float>(0,i))*(codeword.at<float>(0,i));
    }
    // normalize the codeword !
    cout<<"check is "<<check<<endl;
*/
    return 0;
}

void readDictionary(cv::Mat &dictionary, string dictname)
{
 //   string dictname = featurename + "_dictionary";

 //   cout<<"Inside reading dictionary. Reading file "<<dictname<<endl;

    ifstream inputdict(dictname.c_str(),ios::in);

    if(!inputdict.good())
    {
        cout<<"Cannot read file "<<dictname<<endl;
        exit(0);
    }

    int featureDimension = 0;
/*
    if(featurename == "hog")
        featureDimension = featureDimension_hog;
    if(featurename == "hof")
        featureDimension = featureDimension_hof;
    if(featurename == "mbhXY")
        featureDimension = featureDimension_mbhXY;
    if(featurename == "mbh")
        featureDimension = featureDimension_mbhXY;
    if(featurename == "combined1")
        featureDimension = featureDimension_combined;
    if(featurename == "baseline")
        featureDimension = featureDimension_baseline;
*/
    featureDimension = dictionary.cols;
    cout<<"Feature dimension is "<<featureDimension<<endl;
    if(featureDimension == 0)
    {
        cout<<"problem in reading in the dictionary"<<endl;
        exit(0);
    }

    for(int i=0; i<dictionarySize; i++)
    {
        for(int j=0; j<featureDimension; j++)
        {
            inputdict>>dictionary.at<float>(i,j);
        }
    }
    inputdict.close();
    cout<<"read the file"<<endl;
}
