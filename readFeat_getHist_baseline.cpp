#include "readFeat_getHist.h"
void readFeat_getHist_baseline_leopard_trainpos();
void readFeat_getHist_baseline_leopard_trainneg();
void readFeat_getHist_baseline_leopard_testneg();
void readFeat_getHist_baseline_leopard_testpos();


void readFeat_getHist_baseline_leopard()
{

 //  readFeat_getHist_baseline_leopard_trainpos();
 //  readFeat_getHist_baseline_leopard_trainneg();
//  readFeat_getHist_baseline_leopard_testneg();
 //   readFeat_getHist_baseline_leopard_testpos();

}
void readFeat_getHist_baseline_leopard_trainpos()
{

 //   string trainpopsdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/baseline_feat/leopard-baseline-trainpos/";

    string trainpopsdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/baseline_features_new/tiger-baseline-trainpos/";


    vector<string> trainposfiles;

    //getImagenamelist(leopard_train_pos_shotfilename,trainpopsdir, NUM_POS_TRAIN_LEOPARD,trainposfiles);

    getImagenamelist(train_pos_shotfilename,trainpopsdir, NUM_POS_TRAIN,trainposfiles);


    cout<<"Total number of examples is "<<trainposfiles.size()<<endl;

    string codewordfilename = "tiger_code_baseline_trainpos";
    int label = 1;
    long int numfeat = 0;
    ofstream codes(codewordfilename.c_str(),ios::app);

    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);

    readDictionary(dictionary, "tiger_baseline_dictionary");


    int imgnum = 0;
    for(; imgnum < trainposfiles.size(); imgnum++ )
    {
        cout<<"inside image # "<<imgnum<<endl;
        cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

        read_reduce_baseline_desc(trainposfiles[imgnum],allfeatures,-1,trainpopsdir);

        cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);

        featureToCodeword(allfeatures,dictionary,codeword);

        numfeat = numfeat + allfeatures.rows;
        codes<<label<<"\t";
        for(int x=0; x<codeword.cols; x++)
        {
            codes<<codeword.at<float>(0,x)<<"\t";
        }
        codes<<endl;
        allfeatures.release();
        codeword.release();
    }

    cout<<"Total number of features is "<<numfeat<<endl;
    cout<<"Average features is "<<numfeat / trainposfiles.size()<<endl;
    codes.close();
}

void readFeat_getHist_baseline_leopard_trainneg()
{

 //   string trainnegdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/baseline_feat/leopard-baseline-trainneg/";

    string trainnegdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/baseline_features_new/tiger-baseline-trainneg/";

    vector<string> trainnegfiles;

 //   getImagenamelist(leopard_train_neg_shotfilename,trainnegdir, NUM_NEG_TRAIN_LEOPARD,trainnegfiles);
    getImagenamelist(train_neg_shotfilename,trainnegdir, NUM_NEG_TRAIN,trainnegfiles);

    cout<<"Total number of examples is "<<trainnegfiles.size()<<endl;

    string codewordfilename = "tiger_code_baseline_trainneg";
    int label = -1;
    long int numfeat = 0;
    ofstream codes(codewordfilename.c_str(),ios::app);

    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
    readDictionary(dictionary, "tiger_baseline_dictionary");

    int imgnum = 0;
    for(; imgnum < trainnegfiles.size(); imgnum++ )
    {
        cout<<"inside image # "<<imgnum<<endl;
        cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

        read_reduce_baseline_desc(trainnegfiles[imgnum],allfeatures,-1,trainnegdir);

        cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);

        featureToCodeword(allfeatures,dictionary,codeword);

        numfeat = numfeat + allfeatures.rows;
        codes<<label<<"\t";
        for(int x=0; x<codeword.cols; x++)
        {
            codes<<codeword.at<float>(0,x)<<"\t";
        }
        codes<<endl;
        allfeatures.release();
        codeword.release();
    }

    cout<<"Total number of features is "<<numfeat<<endl;
    cout<<"Average features is "<<numfeat / trainnegfiles.size()<<endl;
    codes.close();
}


void readFeat_getHist_baseline_leopard_testneg()
{

 //   string testnegdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/baseline_feat/leopard-baseline-testneg/";

    string testnegdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/baseline_features_new/tiger-baseline-testneg/";

    vector<string> testnegfiles;

    //getImagenamelist(leopard_test_neg_shotfilename,testnegdir, NUM_NEG_TEST_LEOPARD,testnegfiles);
    getImagenamelist(test_neg_shotfilename,testnegdir, NUM_NEG_TEST,testnegfiles);

    cout<<"Total number of examples is "<<testnegfiles.size()<<endl;

    string codewordfilename = "tiger_code_baseline_testneg";
    int label = -1;
    long int numfeat = 0;
    ofstream codes(codewordfilename.c_str(),ios::app);

    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
    readDictionary(dictionary, "tiger_baseline_dictionary");

    int imgnum = 0;
    for(; imgnum < testnegfiles.size(); imgnum++ )
    {
        cout<<"inside image # "<<imgnum<<endl;
        cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

        read_reduce_baseline_desc(testnegfiles[imgnum],allfeatures,-1,testnegdir);

        cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);

        featureToCodeword(allfeatures,dictionary,codeword);

        numfeat = numfeat + allfeatures.rows;
        codes<<label<<"\t";
        for(int x=0; x<codeword.cols; x++)
        {
            codes<<codeword.at<float>(0,x)<<"\t";
        }
        codes<<endl;
        allfeatures.release();
        codeword.release();
    }

    cout<<"Total number of features is "<<numfeat<<endl;
    cout<<"Average features is "<<numfeat / testnegfiles.size()<<endl;
    codes.close();
}


void readFeat_getHist_baseline_leopard_testpos()
{

 //   string testpposdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/baseline_feat/leopard-baseline-testpos/";

    string testpposdir = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/baseline_features_new/tiger-baseline-testpos/";

    vector<string> testposfiles;

    //getImagenamelist(leopard_test_pos_shotfilename,testpposdir, NUM_POS_TEST_LEOPARD,testposfiles);
     getImagenamelist(test_pos_shotfilename,testpposdir, NUM_POS_TEST,testposfiles);

    cout<<"Total number of examples is "<<testposfiles.size()<<endl;

    string codewordfilename = "tiger_code_baseline_testpos";
    int label = 1;
    long int numfeat = 0;
    ofstream codes(codewordfilename.c_str(),ios::app);

    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
    readDictionary(dictionary, "tiger_baseline_dictionary");

    int imgnum = 0;
    for(; imgnum < testposfiles.size(); imgnum++ )
    {
        cout<<"inside image # "<<imgnum<<endl;
        cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

        read_reduce_baseline_desc(testposfiles[imgnum],allfeatures,-1,testpposdir);

        cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);

        featureToCodeword(allfeatures,dictionary,codeword);

        numfeat = numfeat + allfeatures.rows;
        codes<<label<<"\t";
        for(int x=0; x<codeword.cols; x++)
        {
            codes<<codeword.at<float>(0,x)<<"\t";
        }
        codes<<endl;
        allfeatures.release();
        codeword.release();
    }

    cout<<"Total number of features is "<<numfeat<<endl;
    cout<<"Average features is "<<numfeat / testposfiles.size()<<endl;
    codes.close();
}
