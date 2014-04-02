#include "read_reduce.h"
#include "projectFunctions.h"

using namespace std;

void read_reduce_baseline_desc(string fileName,cv::Mat &allfeatures,int maxFeat,string filedirloc)
{
    string name = getActualName(fileName);
    cout<<"Actual Name is "<<name<<endl;
    string descFileName = filedirloc +  name + "_descriptor";
    cout<<"FILE TO READ IS "<<descFileName<<endl;

    string imagename;
    int num_keypoints;
    int desc_size;

    ifstream des(descFileName.c_str(),ios::in);
    if(!des.good())
    {
        cout<<"Cannot open file "<<descFileName<<endl;
        return;
    }
    des>>imagename>>num_keypoints>>desc_size;
    cout<<"Desc size read is "<<desc_size<<endl;

    cv::Mat descriptors = cvCreateMat(num_keypoints,featureDimension_baseline,CV_32FC1);
    if(desc_size != featureDimension_baseline)
    {
        cout<<"Descriptor size is "<<desc_size<<" while it should be "<<featureDimension_baseline<<endl;
        exit(0);

    }
    for(int i=0; i<num_keypoints; i++)
    {
        for(int j=0; j<desc_size; j++)
        {
            des>>descriptors.at<float>(i,j);
        }
    }
    des.close();
    cout<<"Number of features read is "<<descriptors.rows<<endl;


    if(maxFeat == -1)
    {
        allfeatures.push_back(descriptors);
    }
    else
    {
        // reduce features
        cv::Mat reducedFeat = reduceFeatures(descriptors, maxFeat);
        cout<<"Reduced features to "<<reducedFeat.rows<<endl;
        allfeatures.push_back(reducedFeat);
    }
}

void readAllMotionFeatures(cv::Mat &allfeatures,string featurename,string shotfilename,string dirname,int num_examples,int startvidn,
                           int endvidn, int reduceMax)
{
    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(shotfilename,startframes,endframes,keyframes,sizes,num_examples);

    cout<<"The total number of shots are "<<startframes.size()<<endl;
    int vidn = startvidn;
    for(; vidn<endvidn; vidn++)
    {
        cout<<"reading features for shot # "<<vidn<<endl;
        readMotionFeat(startframes[vidn], endframes[vidn], featurename, dirname,allfeatures,reduceMax);
    }
    cout<<"Total features read is "<<allfeatures.rows<<endl;
}

void readallCombined1Features(cv::Mat &allfeatures,string featurename,string shotfilename,string dirname,int num_examples,int startvidn,int endvidn, int reduceMax)
{
    cout<<"Starting to read the combined features from the file "<<endl;
    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(shotfilename,startframes,endframes,keyframes,sizes,num_examples);

    cout<<"The total number of shots are "<<startframes.size()<<endl;
    int vidn = startvidn;
    for(; vidn<endvidn; vidn++)
    {
        cout<<"reading combined features for shot # "<<vidn<<endl;
        cv::Mat hog = cvCreateMat(0,featureDimension_hog,CV_32FC1);
        cv::Mat hof = cvCreateMat(0,featureDimension_hof,CV_32FC1);
        cv::Mat mbh = cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);

        readMotionFeat(startframes[vidn], endframes[vidn], "hog", dirname,hog,-1);
        readMotionFeat(startframes[vidn], endframes[vidn], "hof", dirname,hof,-1);
        readMotionFeat(startframes[vidn], endframes[vidn], "mbh", dirname,mbh,-1);

        cv::Mat combined = concatenateFeatures_reduce(hog, hof, mbh,reduceMax);
        cout<<"Read "<<combined.rows<<"features with columns "<<combined.cols<<endl;
        allfeatures.push_back(combined);
        combined.release();
        hog.release();
        hof.release();
        mbh.release();
        //readMotionFeat(startframes[vidn],endframes[vidn],"combined1",dirname,allfeatures,reduceMax);
    }
    cout<<"Total features read is "<<allfeatures.rows<<endl;
}

void readMotionFeat(int start, int end, string featname,string dirname,cv::Mat &allfeatures, int reduceMax)
{
    stringstream infilename ;
    infilename<<dirname<<"/"<<featname<<"_"<<start<<"_"<<end;
    ifstream in(infilename.str().c_str(),ios::in);
    if(!in.good())
    {
        cout<<"Cannot open file "<<infilename.str()<<endl;
        exit(0);
    }
    int startf,endf,keyf;
    int num_desc, desc_dim;
    in>>startf>>endf>>num_desc>>desc_dim;

    cv::Mat descriptors = cvCreateMat(num_desc,desc_dim,CV_32FC1);

    for(int i=0; i<num_desc; i++)
    {
        for(int j=0; j<desc_dim; j++)
        {
            in>>descriptors.at<float>(i,j);
        }
    }
    in.close();
    cout<<"Number of features read is "<<descriptors.rows<<endl;
    if(reduceMax == -1)
    {
        // neednt reduce
        allfeatures.push_back(descriptors);
    }
    else
    {
        // reduce features
        cv::Mat reducedFeat = reduceFeatures(descriptors, reduceMax);
        cout<<"Reduced features to "<<reducedFeat.rows<<endl;
        allfeatures.push_back(reducedFeat);
    }
}

void readExtFeat(string featfilename,cv::Mat &allfeatures, int reduceMax)
{
    ifstream in(featfilename.c_str(),ios::in);
    if(!in.good())
    {
        cout<<"Cannot open file "<<featfilename<<endl;
        exit(0);
    }
    int startf,endf,keyf;
    int num_desc, desc_dim;
    in>>startf>>endf>>num_desc>>desc_dim;

    cv::Mat descriptors = cvCreateMat(num_desc,desc_dim,CV_32FC1);

    for(int i=0; i<num_desc; i++)
    {
        for(int j=0; j<desc_dim; j++)
        {
            in>>descriptors.at<float>(i,j);
        }
    }
    in.close();
    cout<<"Number of features read is "<<descriptors.rows<<endl;
    if(reduceMax == -1)
    {
        // neednt reduce
        allfeatures.push_back(descriptors);
    }
    else
    {
        // reduce features
        cv::Mat reducedFeat = reduceFeatures(descriptors, reduceMax);
        cout<<"Reduced features to "<<reducedFeat.rows<<endl;
        allfeatures.push_back(reducedFeat);
    }

}

