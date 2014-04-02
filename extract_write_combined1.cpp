#include "extract_write.h"

using namespace std;

int extract_write_Combined1(char* video, string vocabshotfile,string dirname,int num_examples);

void extract_write_combined1_features(char* video)
{
    extract_write_Combined1(video,train_pos_shotfilename,"trainpos",NUM_POS_TRAIN);
    extract_write_Combined1(video,train_neg_shotfilename,"trainneg",NUM_NEG_TRAIN);
    extract_write_Combined1(video,test_pos_shotfilename,"testpos",NUM_POS_TEST);
    extract_write_Combined1(video,test_neg_shotfilename,"testneg",NUM_NEG_TEST);
}

int extract_write_Combined1(char* video, string vocabshotfile,string dirname,int num_examples)
{
    cout<<"Start reading videos and extracting features"<<endl;

    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(vocabshotfile,startframes,endframes,keyframes,sizes,num_examples);
    cout<<"The total number of shots are "<<startframes.size()<<endl;

    int vidn = 0;
    for(; vidn<startframes.size(); vidn++)
    {
        int start_frame = startframes[vidn];
        int end_frame = endframes[vidn];
        int key_frame = keyframes[vidn];

        cv::Mat hog =cvCreateMat(0,featureDimension_hog,CV_32FC1);
        cv::Mat hof =cvCreateMat(0,featureDimension_hof,CV_32FC1);
        cv::Mat mbh =cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);

// TODO:: DO I NEED TO DO TRACKING AGAIN ?? I CAN READ FEATURES WHICH I HAVE WRITTED ALREADY

        cout<<"In the "<<vidn<<" video"<<" Start frame "<<start_frame<<" End Frame "<<end_frame<<endl;
        cout<<"Start the tracking";

        doTracking(start_frame + FRAME_OFFSET, end_frame+FRAME_OFFSET,video, hog, hof, mbh,key_frame+FRAME_OFFSET);

        // concatenate features !
        cv::Mat combinedTemp = concatenateFeatures_reduce(hog, hof, mbh, -1);
        write_feat_to_file_combined1(combinedTemp, startframes[vidn], endframes[vidn],"combined1",dirname);

        cout<<"Tracking done for video "<<vidn<<endl;
        hog.release();
        hof.release();
        mbh.release();
        combinedTemp.release();
    }
    cout<<"ALL  tracking done. Now run kmeans and get the dictionary words"<<endl;
    return 0;
}

