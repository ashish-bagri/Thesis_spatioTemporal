#include "svm_test.h"
#include "read_reduce.h"
using namespace std;

void svm_motion_precision_recall(string featurename)
{
    string testposcodename = "tiger_code_" + featurename + "_testpos";
    string testnegcodename = "tiger_code_" + featurename + "_testneg";

    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);

    readTestCodeWords(testcodewords,testlabels,testposcodename,testnegcodename,NUM_POS_TEST,NUM_NEG_TEST);

    cout<<"Total number of test codewords are "<<testcodewords.rows<<endl;
    // SVM test
    string svmfilename = "tiger_" + featurename + "_svm";
    svmTest_threshold(svmfilename,testcodewords,testlabels,featurename);
}
void getSVMMargins(string featurename)
{
    string testposcodename = "code_" + featurename + "_testpos";
    string testnegcodename = "code_" + featurename + "_testneg";

    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);

    readTestCodeWords(testcodewords,testlabels,testposcodename,testnegcodename,NUM_POS_TEST,NUM_NEG_TEST);

    cout<<"Total number of test codewords are "<<testcodewords.rows<<endl;
    // SVM test
    string svmfilename = featurename + "_svm";
    svmtest_distancemargin(svmfilename,testcodewords,testlabels,featurename);

}
void svm_getMargins(string svmfilename, cv::Mat& testingcodes, vector<float>& margins)
{
    CvSVM svm ;
    cout<<"Using the file "<<svmfilename<<" as the svm input file "<<endl;
    svm.load(svmfilename.c_str());
    int total_test = testingcodes.rows;

    for(int i=0; i<total_test; i++)
    {
        float pred_margin = svm.predict(testingcodes.row(i),true);
        margins[i] = pred_margin;
    }
}


// the function writes the actual margins for each of the testing examples. It is useful to visualize the hyperplane sepeartion
void svmtest_distancemargin(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels,string featurename)
{
//   srand((unsigned)time(0));
    CvSVM svm ;
    cout<<"Using the file "<<svmfilename<<" as the svm input file "<<endl;
    svm.load(svmfilename.c_str());

    string posIndxMargins   = featurename + "pos";
    string negIndxMargins = featurename + "neg";
    ofstream posmargin(posIndxMargins.c_str(),ios::out);
    ofstream negmargin(negIndxMargins.c_str(),ios::out);


    int total_test = testingcodes.rows;

    for(int i=0; i<total_test; i++)
    {
        float pred_margin = svm.predict(testingcodes.row(i),true);


        if(testingLabels.at<float>(i,0) == 1)
        {
            posmargin<<pred_margin<<"\t"<<i<<endl;
        }
        else
        {
            negmargin<<pred_margin<<"\t"<<i<<endl;
        }
    }
    //  posmargin<<"];"<<endl;
    //  negmargin<<"];"<<endl;

    //  posmargin<<"plot("<<featurename<<"_pos(:,1),"<<featurename<<"_pos(:,2),'go','MarkerSize',8);";
    //  negmargin<<"plot("<<featurename<<"_neg(:,1),"<<featurename<<"_neg(:,2),'ro','MarkerSize',8);";
    string scriptfilename = featurename + "margin.m";

    ofstream script(scriptfilename.c_str(),ios::out);

    script<<"load "<<posIndxMargins<<endl;
    script<<"load "<<negIndxMargins<<endl;
    script<<"figure"<<endl;
    script<<"plot("<<posIndxMargins<<"(:,1),"<<posIndxMargins<<"(:,2),'go','MarkerSize',8);"<<endl;
    script<<"hold on"<<endl;
    script<<"plot("<<negIndxMargins<<"(:,1),"<<negIndxMargins<<"(:,2),'ro','MarkerSize',8);"<<endl;
    script<<"title('Margin - "<<featurename<<"');"<<endl;
    script<<"xlabel('Prediction margins');"<<endl;

    script.close();
    posmargin.close();
    negmargin.close();
}

bool sortOrder_ascending(pair<int,float> i, pair<int,float> j)
{
    if(i.second < j.second)
        return true;
    else
        return false;
}

void sortOrder(vector<float>& scores, vector<int>& ranking)
{
    vector<pair<int,float> > order_pair(scores.size());
    for(int i=0; i<order_pair.size(); i++)
    {
        order_pair[i] = (pair<int,float>(i,scores[i]));
    }

    sort(order_pair.begin(),order_pair.end(),sortOrder_ascending);
    for(int i=0; i<order_pair.size(); i++)
    {
        ranking[i] = order_pair[i].first;
    }
}

void classifierAVgPR(vector<float>& groundTruth, vector<float>& scores)
{

}
/**
classificPrecisionRecall
input:
groundTruth - vector containing the labels of the ground truth
scores - vector containing the margings of svm, in synch with ground truth
output:
precision - precision values for every possible margin
recall - recall values for every possible margin
ap - average precision using the above results - simualtes the area under the curve
ranking - ranking of the ground truth using the scores in desceding order ! (more negative <positive examples> to less negative <negative examples>

**/
void classificPrecisionRecall(vector<float>& groundTruth, vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<int>& ranking)
{
    // rank the scores in a ranking..
   /* cout<<"Raw scores : "<<endl;
    ofstream raw("rawscores",ios::out);
    for(int i=0;i<scores.size();i++)
    {
        raw<<scores[i]<<endl;
    }
    raw.close();

    */
    sortOrder(scores,ranking);

    // calculate precision, recall and ap
    int retrieved_hits =0;
    int recall_norm = 0; // TP + FN = total num of positive points in the ground truth
    for(int i=0; i<groundTruth.size(); i++)
    {
        if(groundTruth[i] == 1)
            recall_norm++;
    }
 //   cout<<"Total recall norm, ie, positive examples is "<<recall_norm<<endl;

    ap=0;

    recall[0] = 0;
    for (int idx = 0; idx < groundTruth.size(); ++idx)
    {
        if (groundTruth[ranking[idx]] != -1) ++retrieved_hits;

        precision[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(idx+1);
        recall[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(recall_norm);

        if (idx == 0)
        {
            //add further point at 0 recall with the same precision value as the first computed point
            precision[idx] = precision[idx+1];
        }
        if (recall[idx+1] == 1.0)
        {
            //if recall = 1, then end early as all positive images have been found
            recall.resize(idx+2);
            precision.resize(idx+2);
            break;
        }
    }
    /* calculate ap */

    /* make precision monotonically decreasing for purposes of calculating ap */
    vector<float> precision_monot(precision.size());
    vector<float>::iterator prec_m_it = precision_monot.begin();
    for (vector<float>::iterator prec_it = precision.begin(); prec_it != precision.end(); ++prec_it, ++prec_m_it)
    {
        vector<float>::iterator max_elem;
        max_elem = std::max_element(prec_it,precision.end());
        (*prec_m_it) = (*max_elem);
    }
    /* calculate ap */
    for (size_t idx = 0; idx < (recall.size()-1); ++idx)
    {
        ap += (recall[idx+1] - recall[idx])*precision_monot[idx+1] +   //no need to take min of prec - is monotonically decreasing
              0.5f*(recall[idx+1] - recall[idx])*std::abs(precision_monot[idx+1] - precision_monot[idx]);
    }


}
// function to get precision recall values for different thresholds on the margin of the svm..
void svmTest_threshold(string svmfilename, cv::Mat& testingcodes, cv::Mat &testingLabels,string featurename)
{
    cout<<" Do SVM Testing "<<endl;
    CvSVM svm ;
    cout<<"Using the file "<<svmfilename<<" as the svm input file "<<endl;
    svm.load(svmfilename.c_str());

    string precision_recall_file =  featurename  + "PR";
    ofstream precision_recall(precision_recall_file.c_str(),ios::out);

    float min_margin = 0;
    float max_margin = 0;
    int total_test = testingcodes.rows;

    for(int i=0; i<total_test; i++)
    {
        float pred = svm.predict(testingcodes.row(i));
        float pred_margin = svm.predict(testingcodes.row(i),true);
        if(pred_margin < min_margin)
            min_margin = pred_margin;
        if(pred_margin > max_margin)
            max_margin = pred_margin;

    }
    cout<<"Min margin == "<<min_margin<<endl;
    cout<<"Max margin == "<<max_margin<<endl;

    float threshold = min_margin + 0.1;
    float step_threshold = 0.2;
    float prev_threshold = threshold;
    while(threshold <= max_margin)
    {
        if(threshold > 0 && prev_threshold < 0)
            threshold = 0;

        if(threshold == prev_threshold)
            threshold = threshold + step_threshold;

        cout<<"Threshold : "<<threshold<<endl;
        int right_pos = 0;
        int right_neg = 0;
        int total_pos = 0;
        int total_neg = 0;
        int num_correct_test = 0;


        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;
        for(int i=0; i<total_test; i++)
        {
            float pred_margin = svm.predict(testingcodes.row(i),true);
            float pred = -1;
            if(pred_margin < threshold)
            {
                pred = 1;
            }

            if(testingLabels.at<float>(i,0) == 1)
            {
                total_pos ++;
                if(pred == 1)
                {
                    TP ++;
                    right_pos++;
                    num_correct_test ++ ;
                }
                else
                {
                    FN ++;
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
                }
                else
                {
                    FP ++;
                }
            }
        }
        //      cout<<TP<<" "<<FP<<" "<<FN<<" "<<FP<<endl;
        prev_threshold = threshold;
        threshold = threshold + step_threshold;
        float accuracy = num_correct_test*100.0f/total_test;
        float precision = TP / float((TP + FP));
        float recall = TP / float((TP + FN));
        if(precision == 0 && recall == 0)
            continue;
        precision_recall<<prev_threshold<<"\t"<<precision<<"\t"<<recall<<"\t"<<accuracy<<endl;

    };


    precision_recall.close();
}

