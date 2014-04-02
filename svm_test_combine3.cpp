#include "svm_test.h"
#include "read_reduce.h"

void combine_3()
{
    combine3_motionOrBaseline();
}
void combine_3_leopard()
{
// late fusion
// multiple kernel learning !
// need to have the 3 classifier at hand.. 3 independently good classifier ? or what else is possible ?

//its late fusion ! so have to imagine independence at the earlier level
// read in the 3 test codewords !

    int num_test_pos = NUM_POS_TEST_LEOPARD;
    int num_test_neg = NUM_NEG_TEST_LEOPARD;
    int total_test = num_test_pos + num_test_neg;

    vector<float> margin_hog;
    vector<float> margin_hof;
    vector<float> margin_mbh;
    readCode_DofusionPrediction_leopard("hog",num_test_pos,num_test_neg, margin_hog);
    readCode_DofusionPrediction_leopard("hof",num_test_pos,num_test_neg, margin_hof);
    readCode_DofusionPrediction_leopard("mbh",num_test_pos,num_test_neg, margin_mbh);
    cout<<"size of margins is "<<margin_mbh.size();
    cout<<"Total test codes is "<<total_test<<endl;

// the first num_test_pos are positive !
// the next num_test_neg are negative
    vector<float> labels(num_test_neg+num_test_pos);
    cout<<"size of labels is "<<labels.size()<<endl;

    for(int i=0; i<num_test_pos; i++)
    {
        labels[i] = 1;
    }
    for(int i=num_test_pos; i<num_test_pos+num_test_neg; i++)
    {
        labels[i] = -1;
    }

    int right_pos = 0;
    int right_neg = 0;
    int num_correct_test = 0;

    for(int i=0; i<labels.size(); i++)
    {
        cout<<"Test shot # "<<i<<endl;

        float avg_margin = (margin_hog[i] + margin_hof[i] + margin_mbh[i] ) / 3.0f;
        cout<<margin_hog[i]<<"\t"<<margin_hof[i]<<"\t"<<margin_mbh[i]<<"\t"<<avg_margin<<"\t"<<labels[i]<<endl;
        // cout<<"Average margin is "<<avg_margin<<". Label is "<<labels[i]<<endl;
        float pred;
        if(avg_margin >  0)
        {
            pred = 1;
        }
        else
        {
            pred = -1;
        }

        if(pred == labels[i])
        {
            if(labels[i] == 1)
                right_pos++;
            else
                right_neg++;
            num_correct_test ++ ;
        }
    }

    cout<<"The total number of correct prediction is "<<num_correct_test<<" (out of "<<total_test<<"). Accuracy: "<<num_correct_test*100.0f/total_test<<endl;

    cout<<"The total positive correct prediction is = "<<right_pos<<"(out of "<<num_test_pos<<"). Accuracy: "<<right_pos*100.0f/num_test_pos<<endl;

    cout<<"The total negative correct prediction is = "<<right_neg<<"(out of "<<num_test_neg<<"). Accuracy: "<<right_neg*100.0f/num_test_neg<<endl;

}

void readCode_DofusionPrediction_leopard(string featurename, int num_test_pos, int num_test_neg, vector<float>& margins)
{
    cout<<"*************** FEATURE "<<featurename<<" ************************        "<<endl;
    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);

    stringstream testposs ;
    testposs<<"leopard_code_"<<featurename<<"_testpos";

    stringstream testnegs ;
    testnegs<<"leopard_code_"<<featurename<<"_testneg";

    readTestCodeWords(testcodewords,testlabels,testposs.str(),testnegs.str(),num_test_pos,num_test_neg);

    string svmfilename = "leopard_" + featurename + "_svm" ;
    fusionPrediction(testcodewords, testlabels, margins, svmfilename);

}


void combine_3_tiger()
{
    int num_test_pos = NUM_POS_TEST;
    int num_test_neg = NUM_NEG_TEST;
    int total_test = num_test_pos + num_test_neg;

    vector<float> margin_hog;
    vector<float> margin_hof;
    vector<float> margin_mbh;
    readCode_DofusionPrediction("hog",num_test_pos,num_test_neg, margin_hog);
    readCode_DofusionPrediction("hof",num_test_pos,num_test_neg, margin_hof);
    readCode_DofusionPrediction("baseline",num_test_pos,num_test_neg, margin_mbh);

    cout<<"size of margins is "<<margin_mbh.size();
    cout<<"Total test codes is "<<total_test<<endl;



// the first num_test_pos are positive !
// the next num_test_neg are negative
    vector<float> labels(num_test_neg+num_test_pos);
    cout<<"size of labels is "<<labels.size()<<endl;

    for(int i=0; i<num_test_pos; i++)
    {
        labels[i] = 1;
    }
    for(int i=num_test_pos; i<num_test_pos+num_test_neg; i++)
    {
        labels[i] = -1;
    }

    int right_pos = 0;
    int right_neg = 0;
    int num_correct_test = 0;

    for(int i=0; i<labels.size(); i++)
    {
        cout<<"Test shot # "<<i<<endl;

        float avg_margin = (margin_hog[i] + margin_hof[i] + margin_mbh[i] ) / 3.0f;
        cout<<margin_hog[i]<<"\t"<<margin_hof[i]<<"\t"<<margin_mbh[i]<<"\t"<<avg_margin<<"\t"<<labels[i]<<endl;
        // cout<<"Average margin is "<<avg_margin<<". Label is "<<labels[i]<<endl;
        float pred;
        if(avg_margin >  0)
        {
            pred = 1;
        }
        else
        {
            pred = -1;
        }

        if(pred == labels[i])
        {
            if(labels[i] == 1)
                right_pos++;
            else
                right_neg++;
            num_correct_test ++ ;
        }
    }

    cout<<"The total number of correct prediction is "<<num_correct_test<<" (out of "<<total_test<<"). Accuracy: "<<num_correct_test*100.0f/total_test<<endl;

    cout<<"The total positive correct prediction is = "<<right_pos<<"(out of "<<num_test_pos<<"). Accuracy: "<<right_pos*100.0f/num_test_pos<<endl;

    cout<<"The total negative correct prediction is = "<<right_neg<<"(out of "<<num_test_neg<<"). Accuracy: "<<right_neg*100.0f/num_test_neg<<endl;
    cout<<"Precision is "<<right_pos*100.0f / (right_pos + (num_test_neg - right_neg))<<endl;
    cout<<"Recall is "<<right_pos *100.0f/ (right_pos + (num_test_pos - right_pos))<<endl;

}

void readCode_DofusionPrediction(string featurename, int num_test_pos, int num_test_neg, vector<float>& margins)
{
    cout<<"*************** FEATURE "<<featurename<<" ************************        "<<endl;
    cv::Mat testcodewords = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat testlabels = cvCreateMat(0,1,CV_32FC1);
    stringstream testposs ;
    stringstream testnegs ;
    string svmfilename;

    if(featurename == "baseline")
    {
        testposs<<"tiger_";
        testnegs<<"tiger_";
        svmfilename = "tiger_" + featurename + "_svm" ;
    }
    else
    {
        svmfilename= featurename + "_svm";
    }
    testposs<<"code_"<<featurename<<"_testpos";
    testnegs<<"code_"<<featurename<<"_testneg";

    readTestCodeWords(testcodewords,testlabels,testposs.str(),testnegs.str(),num_test_pos,num_test_neg);


    fusionPrediction(testcodewords, testlabels, margins, svmfilename);
}

void fusionPrediction(cv::Mat& testingcodes, cv::Mat& testingLabels,vector<float>& margins, string svmfilename)
{
    CvSVM svm ;
    cout<<"using svm file name "<<svmfilename<<endl;
    svm.load(svmfilename.c_str());

    int total_test = testingcodes.rows;
    cout<<"Total test codewords here is "<<total_test<<endl;
    for(int i=0; i<total_test; i++)
    {
        float pred_margin = svm.predict(testingcodes.row(i),true);
        //      cout<<"Pred_margin is "<<pred_margin;
        float pred = svm.predict(testingcodes.row(i));
        //     cout<<"\t Prediction is "<<pred<<endl;

        margins.push_back(pred*abs(pred_margin));
    }
}

void combine3_tiger_choose()
{
    int num_test_pos = NUM_POS_TEST;
    int num_test_neg = NUM_NEG_TEST;
    int total_test = num_test_pos + num_test_neg;
    vector<string> features;
    features.push_back("hog");
    features.push_back("hof");
    features.push_back("mbh");
    features.push_back("baseline");

    int num_features = features.size();
    vector< vector<float> > margin(num_features);
    for(int i=0; i<num_features; i++)
    {
        cout<<"Using feature "<<features[i]<<endl;
        readCode_DofusionPrediction(features[i],num_test_pos,num_test_neg, margin[i]);
    }

    // the first num_test_pos are positive !
// the next num_test_neg are negative
    vector<float> labels(num_test_neg+num_test_pos);
    cout<<"size of labels is "<<labels.size()<<endl;

    for(int i=0; i<num_test_pos; i++)
    {
        labels[i] = 1;
    }
    for(int i=num_test_pos; i<num_test_pos+num_test_neg; i++)
    {
        labels[i] = -1;
    }

    int right_pos = 0;
    int right_neg = 0;
    int num_correct_test = 0;

    for(int i=0; i<labels.size(); i++)
    {
        cout<<"Test shot # "<<i<<endl;

        float avg_margin =0;
        for(int k=0; k<num_features; k++)
        {
            avg_margin += margin[k][i];
            cout<<margin[k][i]<<"\t";
        }
        cout<<labels[i]<<endl;
        avg_margin = avg_margin / num_features;

        float pred;
        if(avg_margin >  0)
        {
            pred = 1;
        }
        else
        {
            pred = -1;
        }

        if(pred == labels[i])
        {
            if(labels[i] == 1)
                right_pos++;
            else
                right_neg++;
            num_correct_test ++ ;
        }
    }

    cout<<"The total number of correct prediction is "<<num_correct_test<<" (out of "<<total_test<<"). Accuracy: "<<num_correct_test*100.0f/total_test<<endl;

    cout<<"The total positive correct prediction is = "<<right_pos<<"(out of "<<num_test_pos<<"). Accuracy: "<<right_pos*100.0f/num_test_pos<<endl;

    cout<<"The total negative correct prediction is = "<<right_neg<<"(out of "<<num_test_neg<<"). Accuracy: "<<right_neg*100.0f/num_test_neg<<endl;
    cout<<"Precision is "<<right_pos*100.0f / (right_pos + (num_test_neg - right_neg))<<endl;
    cout<<"Recall is "<<right_pos *100.0f/ (right_pos + (num_test_pos - right_pos))<<endl;
}




void combine3_motionOrBaseline()
{

    int num_test_pos = NUM_POS_TEST;
    int num_test_neg = NUM_NEG_TEST;
    int total_test = num_test_pos + num_test_neg;
    vector<string> features;
    features.push_back("hog");
    //  features.push_back("hof");
    features.push_back("mbh");


    int num_features = features.size();
    vector< vector<float> > margin(num_features);
    for(int i=0; i<num_features; i++)
    {
        cout<<"Using feature "<<features[i]<<endl;
        readCode_DofusionPrediction(features[i],num_test_pos,num_test_neg, margin[i]);
    }

    // the first num_test_pos are positive !
// the next num_test_neg are negative
    vector<float> labels(num_test_neg+num_test_pos);

    vector<float> avg_margins;

    cout<<"size of labels is "<<labels.size()<<endl;

    for(int i=0; i<num_test_pos; i++)
    {
        labels[i] = 1;
    }
    for(int i=num_test_pos; i<num_test_pos+num_test_neg; i++)
    {
        labels[i] = -1;
    }

    int right_pos = 0;
    int right_neg = 0;
    int num_correct_test = 0;

    for(int i=0; i<labels.size(); i++)
    {
        float avg_margin =0;
        for(int k=0; k<num_features; k++)
        {
            avg_margin += margin[k][i];
            //         cout<<margin[k][i]<<"\t";
        }
        //    cout<<labels[i]<<endl;
        avg_margin = avg_margin / num_features;
        avg_margins.push_back(avg_margin);
    }

//have average margin for motion
// get baseline margin
    vector<float> baselinemargin;
    readCode_DofusionPrediction("baseline",num_test_pos,num_test_neg,baselinemargin );

    cout<<"Test shot # "<<"\t"<<"BaselineMargin "<<"MotionMargin"<<" Prediction"<<" Actual"<<endl;
//    cout<<"Test shot # "<<"\t"<<"Avg MotionMargin "<<"BaselineMargin"<<"Average "<<"Prediction"<<" Actual"<<endl;
    for(int i=0; i<labels.size(); i++)
    {
        // take a decision based on the value of max seperation ??
        cout<<"Test shot # "<<i<<"\t";
        float basemargin = baselinemargin[i];
        float motionmargin = avg_margins[i];
        //  float bestmargin = (baselinemargin[i] + avg_margins[i]) / 2.0f;
        //   cout<<motionmargin<<"\t"<<basemargin<<"\t"<<bestmargin<<"\t";

        cout<<basemargin<<"\t"<<motionmargin<<"\t";
        float bestmargin = max(abs(basemargin),abs(motionmargin));
        if(bestmargin == abs(motionmargin))
        {
            bestmargin = motionmargin;
        }
        else
        {
            bestmargin = basemargin;
        }

        float pred;
        if(bestmargin >  0)
        {
            pred = 1;
        }
        else
        {
            pred = -1;
        }
        cout<<bestmargin<<"\t";
        cout<<pred<<"\t"<<labels[i]<<endl;
        if(pred == labels[i])
        {
            if(labels[i] == 1)
                right_pos++;
            else
                right_neg++;
            num_correct_test ++ ;
        }
    }


    cout<<"The total number of correct prediction is "<<num_correct_test<<" (out of "<<total_test<<"). Accuracy: "<<num_correct_test*100.0f/total_test<<endl;

    cout<<"The total positive correct prediction is = "<<right_pos<<"(out of "<<num_test_pos<<"). Accuracy: "<<right_pos*100.0f/num_test_pos<<endl;

    cout<<"The total negative correct prediction is = "<<right_neg<<"(out of "<<num_test_neg<<"). Accuracy: "<<right_neg*100.0f/num_test_neg<<endl;
    cout<<"Precision is "<<right_pos*100.0f / (right_pos + (num_test_neg - right_neg))<<endl;
    cout<<"Recall is "<<right_pos *100.0f/ (right_pos + (num_test_pos - right_pos))<<endl;
}
