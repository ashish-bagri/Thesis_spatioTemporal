#include "readHist_SVM.h"
#include "read_reduce.h"
using namespace std;
using namespace cv;

void readHist_svm_combined2()
{
    readHist_svm_combined2_baseline_leopard();
}

void readHist_svm_combined2_baseline_tiger()
{
    cout<<"Inside feature combination method #2. "<<endl;

    int num_pos_codes = NUM_POS_TRAIN;

    // string hogpos = "code_hog_trainpos";
    //  string hofpos = "code_hof_trainpos";
    //  string mbhpos = "code_mbh_trainpos";
    string baseline_pos = "tiger_code_baseline_trainpos";

    vector<string> features_to_combine;

    // features_to_combine.push_back(hogpos);
    //  features_to_combine.push_back(hofpos);
    //features_to_combine.push_back(mbhpos);
    features_to_combine.push_back(baseline_pos);

    cv::Mat PosCodes = cvCreateMat(num_pos_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(num_pos_codes,1,CV_32FC1);

    cout<<"Reading combined positive codewords"<<endl;

    readCombine2codewords_baseline(PosCodes,PosLabels,features_to_combine,num_pos_codes);

    int N = 5;
    vector<cv::Mat> codewordsP(N);
    vector<cv::Mat> labelsP(N);
    split_N_sets(codewordsP,labelsP, N,PosCodes, PosLabels, train_pos_shotfilename,num_pos_codes);

//
    // string hogneg = "code_hog_trainneg";
    // string hofneg = "code_hof_trainneg";
    // string mbhneg = "code_mbh_trainneg";
    string baseline_neg = "tiger_code_baseline_trainneg";

    vector<string> features_to_combine_neg;
//   features_to_combine_neg.push_back(hogneg);
    //features_to_combine_neg.push_back(hofneg);
    // features_to_combine_neg.push_back(mbhneg);
    features_to_combine_neg.push_back(baseline_neg);

    int num_neg_codes = NUM_NEG_TRAIN;

    cv::Mat NegCodes = cvCreateMat(num_neg_codes,dictionarySize*features_to_combine_neg.size(),CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(num_neg_codes,1,CV_32FC1);

    readCombine2codewords_baseline(NegCodes,NegLabels,features_to_combine_neg,num_neg_codes);



    vector<cv::Mat> codewordsN(N);
    vector<cv::Mat> labelsN(N);
    split_N_sets(codewordsN,labelsN, N,NegCodes, NegLabels, train_neg_shotfilename,num_neg_codes);

//
    cv::Mat alltraining = cvCreateMat(0,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat alllabels = cvCreateMat(0,1,CV_32FC1);

    for(int i=0; i<N; i++)
    {
        alltraining.push_back(codewordsP[i]);
        alllabels.push_back(labelsP[i]);
        alltraining.push_back(codewordsN[i]);
        alllabels.push_back(labelsN[i]);
    }

    cout<<"Total number of training examples are "<<alltraining.rows<<endl;
    vector<float> accuracies;
    float best_C_value;

    svmCrossVal(alltraining,codewordsP,labelsP,codewordsN, labelsN, N,  "baseline0",accuracies, best_C_value);
    createAndWriteSvm(alltraining,alllabels,"tiger_baseline0",best_C_value, true);
}

void readHist_svm_combined2_baseline_leopard()
{
    cout<<"Inside feature combination method #2. "<<endl;

    int num_pos_codes = NUM_POS_TRAIN_LEOPARD;

    string hogpos = "leopard_code_hog_trainpos";
    string hofpos = "leopard_code_hof_trainpos";
    string mbhpos = "leopard_code_mbh_trainpos";
    string baseline_pos = "leopard_code_baseline_trainpos";

    vector<string> features_to_combine;

    //  features_to_combine.push_back(hogpos);
    //  features_to_combine.push_back(hofpos);
    features_to_combine.push_back(mbhpos);
    features_to_combine.push_back(baseline_pos);

    cv::Mat PosCodes = cvCreateMat(num_pos_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(num_pos_codes,1,CV_32FC1);

    cout<<"Reading combined positive codewords"<<endl;
    readCombine2codewords_baseline(PosCodes,PosLabels,features_to_combine,num_pos_codes);

    int N = 5;
    vector<cv::Mat> codewordsP(N);
    vector<cv::Mat> labelsP(N);
    split_N_sets(codewordsP,labelsP, N,PosCodes, PosLabels, leopard_train_pos_shotfilename,num_pos_codes);

    string hogneg = "leopard_code_hog_trainneg";
    string hofneg = "leopard_code_hof_trainneg";
    string mbhneg = "leopard_code_mbh_trainneg";
    string baseline_neg = "leopard_code_baseline_trainneg";

    vector<string> features_to_combine_neg;
    //  features_to_combine_neg.push_back(hogneg);
    //features_to_combine_neg.push_back(hofneg);
    features_to_combine_neg.push_back(mbhneg);
    features_to_combine_neg.push_back(baseline_neg);

    int num_neg_codes = NUM_NEG_TRAIN_LEOPARD;

    cv::Mat NegCodes = cvCreateMat(num_neg_codes,dictionarySize*features_to_combine_neg.size(),CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(num_neg_codes,1,CV_32FC1);

    readCombine2codewords_baseline(NegCodes,NegLabels,features_to_combine_neg,num_neg_codes);

    vector<cv::Mat> codewordsN(N);
    vector<cv::Mat> labelsN(N);
    split_N_sets(codewordsN,labelsN, N,NegCodes, NegLabels, leopard_train_neg_shotfilename,num_neg_codes);

//
    cv::Mat alltraining = cvCreateMat(0,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat alllabels = cvCreateMat(0,1,CV_32FC1);

    for(int i=0; i<N; i++)
    {
        alltraining.push_back(codewordsP[i]);
        alllabels.push_back(labelsP[i]);
        alltraining.push_back(codewordsN[i]);
        alllabels.push_back(labelsN[i]);
    }

    cout<<"Total number of training examples are "<<alltraining.rows<<endl;
    vector<float> accuracies;
    float best_C_value;

    svmCrossVal(alltraining,codewordsP,labelsP,codewordsN, labelsN, N,  "mbh_base",accuracies, best_C_value);
    createAndWriteSvm(alltraining,alllabels,"mbh_base",best_C_value, true);
}

void readCombine2codewords_baseline(cv::Mat &codes, cv::Mat &labels, vector<string> &codesfilename,int num_codewords)
{
// go through each file  one by one and fill in the codes and labels matrix !

    float norm_factor = codesfilename.size();
    for(int i=0; i<codesfilename.size(); i++)
    {
        cout<<"Reading file "<<codesfilename[i]<<endl;
        ifstream inputcodes1(codesfilename[i].c_str(),ios::in);

        if(!inputcodes1.good())
        {
            cout<<"cannot open file "<<codesfilename[i]<<endl;
            exit(0);
        }
        for(int n=0; n<num_codewords; n++)
        {
            float tempvalue;
            float label1;
            inputcodes1>>label1;
            labels.at<float>(n,0) = label1;
            for(int j = i*dictionarySize; j<(i+1)*dictionarySize; j++)
            {
                inputcodes1>>tempvalue;
                codes.at<float>(n,j) = tempvalue / norm_factor;
                //     debugginginfo = debugginginfo + tempcode.at<float>(0,i);
            }
        }
        inputcodes1.close();
    }
}
