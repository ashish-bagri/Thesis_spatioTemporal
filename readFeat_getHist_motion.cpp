#include "readFeat_getHist.h"

void readFeat_getHist_motion()
{
    getHist_motion("hog",featureDimension_hog,train_pos_shotfilename,"trainpos",NUM_POS_TRAIN,1);
    getHist_motion("hog",featureDimension_hog,train_neg_shotfilename,"trainneg",NUM_NEG_TRAIN,-1);
    getHist_motion("hog",featureDimension_hog,test_pos_shotfilename,"testpos",NUM_POS_TEST,1);
    getHist_motion("hog",featureDimension_hog,test_neg_shotfilename,"testneg",NUM_NEG_TEST,-1);

    getHist_motion("hof",featureDimension_hof,train_pos_shotfilename,"trainpos",NUM_POS_TRAIN,1);
    getHist_motion("hof",featureDimension_hof,train_neg_shotfilename,"trainneg",NUM_NEG_TRAIN,-1);
    getHist_motion("hof",featureDimension_hof,test_pos_shotfilename,"testpos",NUM_POS_TEST,1);
    getHist_motion("hof",featureDimension_hof,test_neg_shotfilename,"testneg",NUM_NEG_TEST,-1);

    getHist_motion("mbh",featureDimension_mbhXY,train_pos_shotfilename,"trainpos",NUM_POS_TRAIN,1);
    getHist_motion("mbh",featureDimension_mbhXY,train_neg_shotfilename,"trainneg",NUM_NEG_TRAIN,-1);
    getHist_motion("mbh",featureDimension_mbhXY,test_pos_shotfilename,"testpos",NUM_POS_TEST,1);
    getHist_motion("mbh",featureDimension_mbhXY,test_neg_shotfilename,"testneg",NUM_NEG_TEST,-1);

}
void readFeat_getHist_motion_leopard()
{
    string trainpopsdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/motion_feat/leopard-trainpos-motionfeat";
    getHist_motion("hog",featureDimension_hog,leopard_train_pos_shotfilename,trainpopsdir,NUM_POS_TRAIN_LEOPARD,1);
    getHist_motion("hof",featureDimension_hof,leopard_train_pos_shotfilename,trainpopsdir,NUM_POS_TRAIN_LEOPARD,1);
    getHist_motion("mbh",featureDimension_mbhXY,leopard_train_pos_shotfilename,trainpopsdir,NUM_POS_TRAIN_LEOPARD,1);

    string trainnegdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/motion_feat/leopard-trainneg-motionfeat";
    getHist_motion("hog",featureDimension_hog,leopard_train_neg_shotfilename,trainnegdir,NUM_NEG_TRAIN_LEOPARD,-1);
    getHist_motion("hof",featureDimension_hof,leopard_train_neg_shotfilename,trainnegdir,NUM_NEG_TRAIN_LEOPARD,-1);
    getHist_motion("mbh",featureDimension_mbhXY,leopard_train_neg_shotfilename,trainnegdir,NUM_NEG_TRAIN_LEOPARD,-1);


    string testposdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/motion_feat/leopard-testpos-motionfeat";
    getHist_motion("hog",featureDimension_hog,leopard_test_pos_shotfilename,testposdir,NUM_POS_TEST_LEOPARD,1);
    getHist_motion("hof",featureDimension_hof,leopard_test_pos_shotfilename,testposdir,NUM_POS_TEST_LEOPARD,1);
    getHist_motion("mbh",featureDimension_mbhXY,leopard_test_pos_shotfilename,testposdir,NUM_POS_TEST_LEOPARD,1);

    string testnegdir = "/media/FreeAgent\ Drive/thesis/LEOPARD_OUTPUT/motion_feat/leopard-testneg-motionfeat";
    getHist_motion("hog",featureDimension_hog,leopard_test_neg_shotfilename,testnegdir,NUM_NEG_TEST_LEOPARD,-1);
    getHist_motion("hof",featureDimension_hof,leopard_test_neg_shotfilename,testnegdir,NUM_NEG_TEST_LEOPARD,-1);
    getHist_motion("mbh",featureDimension_mbhXY,leopard_test_neg_shotfilename,testnegdir,NUM_NEG_TEST_LEOPARD,-1);

}


void getHist_motion(string featurename,int featureDim,string shotfilename,string dirname,int num_examples,int label)
{
    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDim,CV_32FC1);
    string dictname = featurename + "_dictionary";
    readDictionary(dictionary,dictname);

    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(shotfilename,startframes,endframes,keyframes,sizes,num_examples);

    cout<<"The total number of shots are "<<startframes.size()<<endl;

// get code wordsin batched of 50
    string codewordfilename = "leopard_code_" + featurename + "_" + "testneg";
    ofstream codes(codewordfilename.c_str(),ios::app);
    long double numfeat = 0;
    int vidn = 0;
    for(; vidn<startframes.size(); vidn++)
    {
        cv::Mat allfeatures = cvCreateMat(0,featureDim,CV_32FC1);
        cout<<"reading features for shot # "<<vidn<<endl;
        readMotionFeat(startframes[vidn], endframes[vidn], featurename, dirname,allfeatures,-1);
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
    cout<<"For "<<dirname<<endl;
    cout<<"Total number of features is "<<numfeat<<endl;
    cout<<"Average features is "<<numfeat / startframes.size()<<endl;
    codes.close();

}

void readExtFeat_getHist(string featurename,int featureDim,string dirname,int label)
{
// need coeword file where it has to be appended !
// need feature file name
// need dictionary name

    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDim,CV_32FC1);
    string dictname = featurename + "_dictionary";
    readDictionary(dictionary,dictname);

    string codewordfilename = "leopard_code_" + featurename + "_" + "testneg";
    ofstream codes(codewordfilename.c_str(),ios::app);

    cv::Mat allfeatures = cvCreateMat(0,featureDim,CV_32FC1);
    cv::Mat codeword  = cvCreateMat(1,dictionarySize,CV_32FC1);

    readExtFeat(featurename,allfeatures,-1);

    featureToCodeword(allfeatures,dictionary,codeword);
    codes<<label<<"\t";
    for(int x=0; x<codeword.cols; x++)
    {
        codes<<codeword.at<float>(0,x)<<"\t";
    }
    codes<<endl;
    allfeatures.release();
    codeword.release();
    codes.close();
}











