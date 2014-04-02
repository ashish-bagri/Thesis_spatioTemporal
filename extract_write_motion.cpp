/**
MOtion features .

**/
#include "extract_write.h"

int extract_write_MotionFeatures(char* video, string vocabshotfile,string dirname,int num_examples);
void write_feat_to_file_motion(cv::Mat &feat, int start, int end, string featname, string dirname);
void extract_write_motion_features(char* video)
{
  //  extract_write_MotionFeatures(video,"leopard-train-30-pos","leopard-trainpos-feat",461);
 //   extract_write_MotionFeatures(video,"leopard-train-30-neg","leopard-trainneg-motionfeat",296);

  //  extract_write_MotionFeatures(video,"leopard-test-30-pos","leopard-testpos-motionfeat",358);
   extract_write_MotionFeatures(video,"leopard-test-30-neg","leopard-testneg-motionfeat",301);

 //   extract_write_MotionFeatures(video,"leopard-test-30-neg","leopard-testneg-motionfeat",301);

 /*   extract_write_MotionFeatures(video,train_neg_shotfilename,"trainneg",NUM_NEG_TRAIN);
    extract_write_MotionFeatures(video,test_neg_shotfilename,"testneg",NUM_NEG_TEST);
    extract_write_MotionFeatures(video,test_pos_shotfilename,"testpos",NUM_POS_TEST); */
}

int extract_write_MotionFeatures(char* video, string vocabshotfile,string dirname,int num_examples)
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
        cv::Mat hogFeat = cvCreateMat(0,featureDimension_hog,CV_32FC1);
        cv::Mat hofFeat = cvCreateMat(0,featureDimension_hof,CV_32FC1);
        cv::Mat mbhXYFeat = cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);

        cout<<"In the "<<vidn<<" video"<<" Start frame "<<start_frame<<" End Frame "<<end_frame<<endl;

        doTracking(start_frame+FRAME_OFFSET,end_frame+FRAME_OFFSET,video,hogFeat,hofFeat,mbhXYFeat,key_frame+FRAME_OFFSET);
        write_feat_to_file_motion(hogFeat, start_frame, end_frame, "hog", dirname);
        write_feat_to_file_motion(hofFeat, start_frame, end_frame, "hof", dirname);
        write_feat_to_file_motion(mbhXYFeat, start_frame, end_frame, "mbh", dirname);

        cout<<"Tracking and writing done for shot "<<vidn<<endl;
        hogFeat.release();
        hofFeat.release();
        mbhXYFeat.release();
    }
    cout<<"All tracking and writing of features done for "<< dirname<<endl;
    return 0;
}


