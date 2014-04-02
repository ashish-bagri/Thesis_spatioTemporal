#include "readFeat_getHist.h"
void getHist_combined1(cv::Mat &dictionary,string shotfilename,string dirname, int num_examples,int label);
void readFeat_getHist_combined1()
{
    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_combined,CV_32FC1);
    readDictionary(dictionary, "combined1_dictionary");

    string trainposdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/tiger_motionfeatures_infile/tiger-trainpos-motionfeat";
    getHist_combined1(dictionary, train_pos_shotfilename,trainposdir, NUM_POS_TRAIN,1);
    string trainnegdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/tiger_motionfeatures_infile/tiger-trainneg-motionfeat";
    getHist_combined1(dictionary, train_neg_shotfilename,trainnegdir, NUM_NEG_TRAIN,-1);

    string testpos = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/tiger_motionfeatures_infile/tiger-testpos-motionfeat";
    getHist_combined1(dictionary, test_pos_shotfilename,testpos, NUM_POS_TEST,1);


    string testneg = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/tiger_motionfeatures_infile/tiger-testneg-motionfeat";
    getHist_combined1(dictionary, test_neg_shotfilename,testneg, NUM_NEG_TEST,-1);

}

void getHist_combined1(cv::Mat &dictionary,string shotfilename,string dirname, int num_examples,int label)
{
    string codewordfilename = "code_combined1_testneg" ;

    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(shotfilename,startframes,endframes,keyframes,sizes,num_examples);
    cout<<"The total number of shots are "<<startframes.size()<<endl;



    ofstream codes(codewordfilename.c_str(),ios::app);

    int vidn = 200;
    for(; vidn< startframes.size() ; vidn++)
    {
        //  cv::Mat allfeatures = cvCreateMat(0,featureDimension_combined,CV_32FC1);
        cout<<"reading combined features for shot # "<<vidn<<endl;
        cv::Mat hog = cvCreateMat(0,featureDimension_hog,CV_32FC1);
        cv::Mat hof = cvCreateMat(0,featureDimension_hof,CV_32FC1);
        cv::Mat mbh = cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);

        readMotionFeat(startframes[vidn], endframes[vidn], "hog", dirname,hog,-1);
        readMotionFeat(startframes[vidn], endframes[vidn], "hof", dirname,hof,-1);
        readMotionFeat(startframes[vidn], endframes[vidn], "mbh", dirname,mbh,-1);

        cv::Mat combined = concatenateFeatures_reduce(hog, hof, mbh,-1);
        cout<<"Read "<<combined.rows<<"features with columns "<<combined.cols<<endl;

//       readMotionFeat(startframes[vidn], endframes[vidn], "combined1", dirname,allfeatures,-1);
        cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);
        featureToCodeword(combined,dictionary,codeword);

        codes<<label<<"\t";
        for(int x=0; x<codeword.cols; x++)
        {
            codes<<codeword.at<float>(0,x)<<"\t";
        }
        codes<<endl;
        hog.release();
        hof.release();
        mbh.release();
        combined.release();
        codeword.release();
    }
    codes.close();
}

