#include "svm_test.h"
#include "read_reduce.h"
/*
void svm_test_combine2_leopard()
{
    cv::Mat PosCodes = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(0,1,CV_32FC1);

    string hogpos = "leopard_code_hog_testpos";
    string hofpos = "leopard_code_hof_testpos";
    string mbhpos = "leopard_code_mbh_testpos";


    cout<<"Reading combined positive codewords"<<endl;
    readCombined2CodeWords(PosCodes,PosLabels,hogpos,hofpos,mbhpos,NUM_POS_TEST_LEOPARD);
    cout<<"Read codewords #"<<PosCodes.rows<<endl;

    string hogneg = "leopard_code_hog_testneg";
    string hofneg = "leopard_code_hof_testneg";
    string mbhneg = "leopard_code_mbh_testneg";


    cout<<"Reading negative codewords "<<endl;
    cv::Mat NegCodes = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(0,1,CV_32FC1);

    readCombined2CodeWords(NegCodes,NegLabels,hogneg,hofneg,mbhneg,NUM_NEG_TEST_LEOPARD);

    cout<<"Read codewords #"<<NegCodes.rows<<endl;
    cv::Mat alltestcodewords = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat alltestlabels = cvCreateMat(0,1,CV_32FC1);
    alltestcodewords.push_back(PosCodes);
    alltestlabels.push_back(PosLabels);

    alltestcodewords.push_back(NegCodes);
    alltestlabels.push_back(NegLabels);

    cout<<"ALl test data is "<<alltestcodewords.rows<<endl;
    cout<<"To read svm and get the test results "<<endl;

    string svmfilename = "leopard_combined2_svm";
    vector<float> accuracies(3);
    cout<<"Using svm file name :"<<svmfilename;
    doSVMTest(svmfilename, alltestcodewords, alltestlabels , accuracies,"leopard_combined2");

}
*/
/*
void svm_test_combine2_tiger()
{
    cv::Mat PosCodes = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(0,1,CV_32FC1);

    string hogpos = "code_hog_testpos";
    string hofpos = "code_hof_testpos";
    string mbhpos = "code_mbh_testpos";


    cout<<"Reading combined positive codewords"<<endl;
    readCombined2CodeWords(PosCodes,PosLabels,hogpos,hofpos,mbhpos,NUM_POS_TEST);
    cout<<"Read codewords #"<<PosCodes.rows<<endl;

    string hogneg = "code_hog_testneg";
    string hofneg = "code_hof_testneg";
    string mbhneg = "code_mbh_testneg";


    cout<<"Reading negative codewords "<<endl;
    cv::Mat NegCodes = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(0,1,CV_32FC1);

    readCombined2CodeWords(NegCodes,NegLabels,hogneg,hofneg,mbhneg,NUM_NEG_TEST);

    cout<<"Read codewords #"<<NegCodes.rows<<endl;
    cv::Mat alltestcodewords = cvCreateMat(0,CombinedDictionarySize,CV_32FC1);
    cv::Mat alltestlabels = cvCreateMat(0,1,CV_32FC1);
    alltestcodewords.push_back(PosCodes);
    alltestlabels.push_back(PosLabels);

    alltestcodewords.push_back(NegCodes);
    alltestlabels.push_back(NegLabels);

    cout<<"ALl test data is "<<alltestcodewords.rows<<endl;
    cout<<"To read svm and get the test results "<<endl;

    string svmfilename = "combined2_svm";
    vector<float> accuracies(3);
    cout<<"Using svm file name :"<<svmfilename;
    doSVMTest(svmfilename, alltestcodewords, alltestlabels , accuracies,"combined2");
}
*/
void svm_test_combine2_baseline_tiger()
{
    string hogpos = "code_hog_testpos";
    string hofpos = "code_hof_testpos";
    string mbhpos = "code_mbh_testpos";
    string baseline_pos = "tiger_code_baseline_testpos";

    vector<string> features_to_combine;

   // features_to_combine.push_back(hogpos);
    // features_to_combine.push_back(hofpos);
    features_to_combine.push_back(mbhpos);
    features_to_combine.push_back(baseline_pos);

    int num_pos_codes = NUM_POS_TEST;

    cv::Mat PosCodes = cvCreateMat(num_pos_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(num_pos_codes,1,CV_32FC1);

    readCombine2codewords_baseline(PosCodes,PosLabels,features_to_combine,num_pos_codes);


    string hogneg = "code_hog_testneg";
    string hofneg = "code_hof_testneg";
    string mbhneg = "code_mbh_testneg";
    string baseline_neg = "tiger_code_baseline_testneg";

    vector<string> features_to_combine_neg;
   // features_to_combine_neg.push_back(hogneg);
  //  features_to_combine_neg.push_back(hofneg);
    features_to_combine_neg.push_back(mbhneg);
    features_to_combine_neg.push_back(baseline_neg);

    int num_neg_codes = NUM_NEG_TEST;

    cout<<"Reading negative codewords "<<endl;
    cv::Mat NegCodes = cvCreateMat(num_neg_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(num_neg_codes,1,CV_32FC1);

    readCombine2codewords_baseline(NegCodes,NegLabels,features_to_combine_neg,num_neg_codes);
    cout<<"Read codewords #"<<NegCodes.rows<<" having columns "<<NegCodes.cols<<endl;


    cv::Mat alltestcodewords = cvCreateMat(0,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat alltestlabels = cvCreateMat(0,1,CV_32FC1);

    alltestcodewords.push_back(PosCodes);
    alltestlabels.push_back(PosLabels);

    alltestcodewords.push_back(NegCodes);
    alltestlabels.push_back(NegLabels);

    cout<<"ALl test data is "<<alltestcodewords.rows<<endl;

    cout<<"To read svm and get the test results "<<endl;

    string svmfilename = "tiger_combined2_mbh_baseline_svm";

    vector<float> accuracies(3);
    cout<<"Using svm file name :"<<svmfilename;
    doSVMTest(svmfilename, alltestcodewords, alltestlabels , accuracies,"mbh_baseline");
    //  svmTest_threshold(svmfilename, alltestcodewords, alltestlabels, "combine2-motion-baseline0");
    //svmtest_distancemargin(svmfilename, alltestcodewords, alltestlabels, "hog_mbh_baseline");
}

void svm_test_combine2_baseline_leopard()
{
    string hogpos = "leopard_code_hog_testpos";
    string hofpos = "leopard_code_hof_testpos";
    string mbhpos = "leopard_code_mbh_testpos";
    string baseline_pos = "leopard_code_baseline_testpos";

    vector<string> features_to_combine;
  //features_to_combine.push_back(hogpos);
  features_to_combine.push_back(hofpos);
  // features_to_combine.push_back(mbhpos);
  // features_to_combine.push_back(baseline_pos);

    int num_pos_codes = NUM_POS_TEST_LEOPARD ;

    cv::Mat PosCodes = cvCreateMat(num_pos_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(num_pos_codes,1,CV_32FC1);

    readCombine2codewords_baseline(PosCodes,PosLabels,features_to_combine,num_pos_codes);


    string hogneg = "leopard_code_hog_testneg";
    string hofneg = "leopard_code_hof_testneg";
    string mbhneg = "leopard_code_mbh_testneg";
    string baseline_neg = "leopard_code_baseline_testneg";

    vector<string> features_to_combine_neg;
//   features_to_combine_neg.push_back(hogneg);
    features_to_combine_neg.push_back(hofneg);
 //   features_to_combine_neg.push_back(mbhneg);
  // features_to_combine_neg.push_back(baseline_neg);

    int num_neg_codes = NUM_NEG_TEST_LEOPARD;

    cout<<"Reading negative codewords "<<endl;
    cv::Mat NegCodes = cvCreateMat(num_neg_codes,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(num_neg_codes,1,CV_32FC1);

    readCombine2codewords_baseline(NegCodes,NegLabels,features_to_combine_neg,num_neg_codes);

    cout<<"Read codewords #"<<NegCodes.rows<<" having columns "<<NegCodes.cols<<endl;


    cv::Mat alltestcodewords = cvCreateMat(0,dictionarySize*features_to_combine.size(),CV_32FC1);
    cv::Mat alltestlabels = cvCreateMat(0,1,CV_32FC1);

    alltestcodewords.push_back(PosCodes);
    alltestlabels.push_back(PosLabels);

    alltestcodewords.push_back(NegCodes);
    alltestlabels.push_back(NegLabels);

    cout<<"ALl test data is "<<alltestcodewords.rows<<endl;

    cout<<"To read svm and get the test results "<<endl;

    string svmfilename = "hog_svm";

    vector<float> accuracies(3);
    cout<<"Using svm file name :"<<svmfilename;
     CvSVM svm ;
    cout<<"Using the file "<<svmfilename<<" as the svm input file "<<endl;
    svm.load(svmfilename.c_str());

  //  crossValParam(svm, alltestcodewords,alltestlabels,"leopard_hog","leopard_hog");
    doSVMTest(svmfilename, alltestcodewords, alltestlabels , accuracies,"hof2");
    //  svmTest_threshold(svmfilename, alltestcodewords, alltestlabels, "mbh_base");
    // svmtest_distancemargin(svmfilename, alltestcodewords, alltestlabels, "mbh_base");
}
