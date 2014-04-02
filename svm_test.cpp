#include "svm_test.h"
#include "read_reduce.h"
using namespace std;

void svm_test_motion_leopard(string featurename);
void SVM_TEST()
{
    svm_test_combine2_baseline_leopard();
   //SVM_test_singleClass_youtube();
//   SVM_test_youtubePR();
 //  SVM_test_youtube_mulCategory();
}

void svm_test_motion_leopard(string featurename)
{
    string testposcodename = "leopard_code_" + featurename + "_testpos";
    string testnegcodename = "leopard_code_" + featurename + "_testneg";

    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);

    readTestCodeWords(testcodewords,testlabels,testposcodename,testnegcodename,NUM_POS_TEST_LEOPARD,NUM_NEG_TEST_LEOPARD);

    cout<<"Total number of test codewords are "<<testcodewords.rows<<endl;
    // SVM test
    string svmfilename = "leopard_" + featurename + "_svm";
    vector<float> accuracies(3);
    doSVMTest(svmfilename,testcodewords,testlabels,accuracies,featurename);
}

void svm_test_motion(string featurename)
{
    string testposcodename = "tiger_code_" + featurename + "_testpos";
    string testnegcodename = "tiger_code_" + featurename + "_testneg";

    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);

    readTestCodeWords(testcodewords,testlabels,testposcodename,testnegcodename,NUM_POS_TEST,NUM_NEG_TEST);

    cout<<"Total number of test codewords are "<<testcodewords.rows<<endl;
    // SVM test
    string svmfilename =  "tiger_" + featurename + "_svm";
    vector<float> accuracies(3);



    doSVMTest(svmfilename,testcodewords,testlabels,accuracies,featurename);
}


void doSVMTest(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels , vector<float>& accuracies,string featurename)
{
    cout<<" Do SVM Testing "<<endl;
    CvSVM svm ;
    cout<<"Using the file "<<svmfilename<<" as the svm input file "<<endl;
    svm.load(svmfilename.c_str());

    // string prcurvefile = featurename + "_PRcurve";
    float averageap = crossValParam(svm, testingcodes, testingLabels,featurename);
    ofstream avgpr("avg_pr",ios::app);
    avgpr<<featurename<<"\t"<<averageap<<endl;

    string svmanalysisfile = "detailResult_" + featurename ;

    ofstream svmanalysis(svmanalysisfile.c_str(),ios::out);

    int right_pos = 0;
    int right_neg = 0;
    int total_pos = 0;
    int total_neg = 0;
    int num_correct_test = 0;
    int total_test = testingcodes.rows;

    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    float min_margin = 0;
    float max_margin = 0;
    for(int i=0; i<total_test; i++)
    {
        float pred = svm.predict(testingcodes.row(i));
        float pred_margin = svm.predict(testingcodes.row(i),true);
        if(pred_margin < min_margin)
            min_margin = pred_margin;
        if(pred_margin > max_margin)
            max_margin = pred_margin;

        cout<<i<<"\t"<<pred_margin<<"\t"<<pred<<endl;
        if(testingLabels.at<float>(i,0) == 1)
        {
            total_pos ++;
            if(pred == 1)
            {
                TP ++;
                right_pos++;
                num_correct_test ++ ;
                svmanalysis<<"TruePositive"<<endl;
            }
            else
            {
                FN ++;
                svmanalysis<<"FalseNegative"<<endl;
            }
        }
        else
        {
            // actually not a tiger
            total_neg++;
            if(pred == -1)
            {
                right_neg++;
                num_correct_test ++ ;
                TN++;
                svmanalysis<<"TrueNegative"<<endl;
            }
            else
            {
                FP ++;
                svmanalysis<<"FalsePositive"<<endl;
            }
        }
    }

    cout<<"Min margin == "<<min_margin<<endl;
    cout<<"Max margin == "<<max_margin<<endl;
    svmanalysis<<"Total analysis"<<endl;
    svmanalysis<<"True positive "<<TP<<endl;
    svmanalysis<<"True Negative "<<TN<<endl;
    svmanalysis<<"False Positive "<<FP<<endl;
    svmanalysis<<"False Negative "<<FN<<endl;
    svmanalysis<<"Precision : "<<TP / float((TP + FP))<<endl;
    svmanalysis<<"Recall : "<<TP / float((TP + FN))<<endl;


    cout<<"The total number of correct prediction is "<<num_correct_test<<" (out of "<<total_test<<"). Accuracy: "<<num_correct_test*100.0f/total_test<<endl;
    accuracies[0] = num_correct_test*100.0f/total_test;

    cout<<"The total positive correct prediction is = "<<right_pos<<"(out of "<<total_pos<<"). Accuracy: "<<right_pos*100.0f/total_pos<<endl;
    accuracies[1] = right_pos*100.0f/total_pos;

    cout<<"The total negative correct prediction is = "<<right_neg<<"(out of "<<total_neg<<"). Accuracy: "<<right_neg*100.0f/total_neg<<endl;
    accuracies[2] = right_neg*100.0f/total_neg;

    cout<<"Precision: "<<TP / float((TP + FP))<<endl;
    cout<<"Recall : "<<TP / float((TP + FN))<<endl;

    svmanalysis<<"Accuracy : "<<num_correct_test*100.0f/total_test<<endl;
    svmanalysis.close();

}

void readTestCodeWords(cv::Mat &testcodewords, cv::Mat &testlabels, string testposfilename, string testnegfilename,int num_test_pos, int num_test_neg)
{
    cv::Mat testpositive = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testnegative = cvCreateMat(0,dictionarySize,CV_32FC1);

    cv::Mat testposLabels = cvCreateMat(0,1,CV_32FC1);
    cv::Mat testnegLabels = cvCreateMat(0,1,CV_32FC1);

    cout<<"reading test pos codewords "<<endl;
    readcodewords(testposfilename,testpositive,testposLabels,num_test_pos);
    cout<<"Read "<<testpositive.rows<<" test positive codewords"<<endl;

    cout<<"reading test neg codewords "<<endl;
    readcodewords(testnegfilename,testnegative,testnegLabels,num_test_neg);
    cout<<"Read "<<testnegative.rows<<" test negative codewords"<<endl;

    cout<<"Join test pos and neg"<<endl;
    testcodewords.push_back(testpositive);
    testcodewords.push_back(testnegative);
    cout<<"Total test code words is "<<testcodewords.rows<<endl;
    testlabels.push_back(testposLabels);
    testlabels.push_back(testnegLabels);
}


