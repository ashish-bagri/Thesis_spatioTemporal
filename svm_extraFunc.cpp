#include "readHist_SVM.h"
#include "youtubedata.h"
#include "read_reduce.h"
#include "svm_test.h"

using namespace std;
void getCvalueVectors_range(vector<float>& Cvalues);
void readHist_video_withLabels_youtube(cv::Mat& histogram,cv::Mat& labels,string category, vector<string> featurename,int video,vector<int> shots,int label);
//void readHist_video_withLabels_youtube(cv::Mat& histogram,cv::Mat& labels, int video,int numShots,string category, vector<string> featurename,vector<int> shots,int label);
void readPositiveCategoryHistogram_mulFeatures(int N, string category,vector<string> featurename,vector<cv::Mat>& codewordsP,vector<cv::Mat>& labelsP);
void readNegativeCategoryHistogram_mulFeatures(int N, vector<string> category,vector<string> featurename,vector<cv::Mat>& codewordsN,vector<cv::Mat>& labelsN);
void getPos_Neg_Weights(vector<float>& posWeights,vector<float>& negWeights,int posexample = -1,int negexample = -1);
void gridSearch(cv::Mat& training,cv::Mat& traininglabels,cv::Mat& validation,cv::Mat& validationlabels,vector<float> Cvalues,vector<float> posWeights,vector<float> negWeights,cv::Mat &metric,int nindex);
void NfoldValidation(vector<cv::Mat> codewordsP,vector<cv::Mat> labelsP, vector<cv::Mat> codewordsN,vector<cv::Mat> labelsN,string svmfilenameis);
void trainAuto(vector<cv::Mat> codewordsP,vector<cv::Mat> labelsP, vector<cv::Mat> codewordsN,vector<cv::Mat> labelsN,string svmfilenameis);
// this is the current version (when i did re run of all the test ) used
void SVM_youtube()
{
    srand((unsigned)time(0));
    int N = 5;
    vector<string> allconfigs;
    getallconfig(allconfigs);
    cout<<"size of all config is "<<allconfigs.size()<<endl;
    for(int i=0; i<allconfigs.size(); i++)
    {
        vector<string> allcat;
        posCatNames(allcat);

        for(int pcat = 0; pcat<allcat.size(); pcat++)
        {
            string positivecat = allcat[pcat];
            cout<<"FOR POSITIVE CATEGORY : "<<positivecat<<endl;
            vector<string> negcat;
            for(int tt=0; tt<allcat.size(); tt++)
            {
                if(tt!=pcat)
                {
                    negcat.push_back(allcat[tt]);
                }
            }
            vector<string> feature;
            getparts(allconfigs[i],feature);
            stringstream svmfilenameis;
            for(int f=0; f<feature.size(); f++)
            {
                svmfilenameis<<feature[f]<<"_";
            }
            svmfilenameis<<positivecat<<"Pos";

            vector<cv::Mat> codewordsP(N);
            vector<cv::Mat> labelsP(N);

            readPositiveCategoryHistogram_mulFeatures(N,positivecat,feature,codewordsP,labelsP);
            int totalP = 0;
            // fill the labels vector
            for(int xx=0; xx<N; xx++)
            {
                totalP = totalP + codewordsP[xx].rows;
                cout<<"In index "<<xx<<", The number of histograms are "<<codewordsP[xx].rows<<endl;
            }
            cout<<"Total positive codewords is "<<totalP<<endl;
            cout<<"The codeword dimesion is "<<codewordsP[0].cols<<endl;

            vector<cv::Mat> codewordsN(N);
            vector<cv::Mat> labelsN(N);

            readNegativeCategoryHistogram_mulFeatures(N, negcat,feature,codewordsN,labelsN);
            int totalN = 0;
            for(int tt=0; tt<N; tt++)
            {
                cout<<"In index "<<tt<<", The number of histograms are "<<codewordsN[tt].rows<<endl;
                totalN = totalN + codewordsN[tt].rows;
            }
            cout<<"The total negative codewords are "<<totalN<<endl;
            cout<<"The codeword dimesion is "<<codewordsN[0].cols<<endl;

            NfoldValidation(codewordsP,labelsP, codewordsN,labelsN,svmfilenameis.str());
            // trainAuto(codewordsP,labelsP, codewordsN,labelsN,svmfilenameis.str());
            cout<<"Completed for positive "<<positivecat<<" with config : "<<allconfigs[i]<<endl;
        }
        //     cout<<"Completed for config"<<allconfigs[i]<<endl;
    }
}
void trainAuto(vector<cv::Mat> codewordsP,vector<cv::Mat> labelsP, vector<cv::Mat> codewordsN,vector<cv::Mat> labelsN,string svmfilenameis)
{
    int cols = codewordsP[0].cols;
    int N = codewordsP.size();

    cv::Mat alltraining = cvCreateMat(0,cols,CV_32FC1);
    cv::Mat alllabels = cvCreateMat(0,1,CV_32FC1);

    for(int tt=0; tt<N; tt++)
    {
        alltraining.push_back(codewordsP[tt]);
        alllabels.push_back(labelsP[tt]);
        alltraining.push_back(codewordsN[tt]);
        alllabels.push_back(labelsN[tt]);
    }

    cout<<"Total number of training examples are "<<alltraining.rows<<endl;
    CvSVM svm ;
    CvSVMParams params;
    params.kernel_type=CvSVM::RBF;
    params.svm_type=CvSVM::C_SVC;
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,1000000,1e-6);
    int k_fold = 10;
    int gridNU = 0;
    CvMat* classweight = cvCreateMat(1,2,CV_32FC1);
    float *data = classweight->data.fl;

    data[0] = 0.3;
    data[1] =  0.9;

//C,GAMMA,P,NU,COEF,DEGREE,BALANCE
    //svm.train_auto(alltraining, alllabels, cv::Mat(), cv::Mat(), params, k_fold, CvSVM::get_default_grid(CvSVM::C), 0, 0, 0, 0, 0, false);
    svm.train_auto( alltraining, alllabels, cv::Mat(), cv::Mat(), params,k_fold);

    string svmoutfile = svmfilenameis + "_svm";
    cout<<"Selected SVM. Saving file "<<svmoutfile<<endl;

    cout<<"C value was "<<svm.get_params().C<<endl;
    CvMat * cw = svm.get_params().class_weights;
    float* cwdata = cw->data.fl;
    // cout<<" Neg weight was :"<<cwdata[0]<<endl;
    // cout<<" Neg weight was :"<<cwdata[1]<<endl;
    //cout<<"Negative weight was: "<<negWeights[k]<<endl;
    cout<<"svm count :"<<svm.get_support_vector_count()<<endl;
    svm.save(svmoutfile.c_str());
    //params.class_weights = classweight;
}

void NfoldValidation(vector<cv::Mat> codewordsP,vector<cv::Mat> labelsP, vector<cv::Mat> codewordsN,vector<cv::Mat> labelsN,string svmfilenameis)
{
    int numPositive = 0;
    int numNegative = 0;
    for(int i=0; i<codewordsP.size(); i++)
    {
        numPositive = numPositive + codewordsP[i].rows;
    }

    for(int i=0; i<codewordsN.size(); i++)
    {
        numNegative = numNegative + codewordsN[i].rows;
    }
    cout<<"Number of positive training examples : "<<numPositive<<endl;
    cout<<"Number of negative training examples : "<<numNegative<<endl;


    int cols = codewordsP[0].cols;
    int N = codewordsP.size();

    cv::Mat alltraining = cvCreateMat(0,cols,CV_32FC1);
    cv::Mat alllabels = cvCreateMat(0,1,CV_32FC1);

    for(int tt=0; tt<N; tt++)
    {
        alltraining.push_back(codewordsP[tt]);
        alllabels.push_back(labelsP[tt]);
        alltraining.push_back(codewordsN[tt]);
        alllabels.push_back(labelsN[tt]);
    }

    cout<<"Total number of training examples are "<<alltraining.rows<<endl;

    vector<string> cindxguide;
    vector<float> Cvalues;
    vector<float> posWeights;
    vector<float> negWeights;
    getCvalueVectors(Cvalues,cindxguide,alltraining);
    //  getCvalueVectors_range(Cvalues);
    // cout<<"Number of C values "<<Cvalues.size()<<endl;
    // cin.get();
    /*
            Cvalues.push_back(2000000);
            Cvalues.push_back(10);
            Cvalues.push_back(4);
            Cvalues.push_back(1);
            Cvalues.push_back(0.2);
            Cvalues.push_back(0.5);
    */
    getPos_Neg_Weights(posWeights,negWeights,numPositive,numNegative);


    int allgrid = Cvalues.size()*posWeights.size()*negWeights.size();


    cv::Mat combination_C_pos_neg = cvCreateMat(allgrid,3,CV_32FC1);
    int runcount = 0;
    for(int i=0; i<Cvalues.size(); i++)
    {
        for(int j=0; j<posWeights.size(); j++)
        {
            for(int k=0; k<negWeights.size(); k++)
            {
                combination_C_pos_neg.at<float>(runcount,0) = Cvalues[i];
                combination_C_pos_neg.at<float>(runcount,1) = posWeights[j];
                combination_C_pos_neg.at<float>(runcount,2) = negWeights[k];
                runcount++;
            }
        }
    }
    // grid search criteria
    cv::Mat gridCriteriaMeasure = cvCreateMat(allgrid,N,CV_32FC1);

    for(int i=0; i<N; i++)
    {
        // index i becomes the validation set, the rest added together is the training set
        cv::Mat trainset = cvCreateMat(0,cols,CV_32FC1);
        cv::Mat valset = cvCreateMat(0,cols,CV_32FC1);
        cv::Mat trainlab = cvCreateMat(0,1,CV_32FC1);
        cv::Mat vallab = cvCreateMat(0,1,CV_32FC1);
        for(int j=0; j<N; j++)
        {
            if(i == j)
            {
                //             cout<<"MAKING "<<j<<" AS THE VALIDATION SET"<<endl;
                valset.push_back(codewordsP[j]);
                vallab.push_back(labelsP[j]);
                valset.push_back(codewordsN[j]);
                vallab.push_back(labelsN[j]);
            }
            else
            {
                trainset.push_back(codewordsP[j]);
                trainlab.push_back(labelsP[j]);
                trainset.push_back(codewordsN[j]);
                trainlab.push_back(labelsN[j]);
            }
        }
        gridSearch(trainset,trainlab,valset,vallab,Cvalues,posWeights,negWeights,gridCriteriaMeasure,i);
    }
    float bestacc = 0;
    int bestindx = 0;

    for(int i=0; i<gridCriteriaMeasure.rows; i++)
    {
        cout<<"C= "<<combination_C_pos_neg.at<float>(i,0)<<" PosWeight = "<<combination_C_pos_neg.at<float>(i,1)<<" NegWeight = "<<combination_C_pos_neg.at<float>(i,2);
        float accatthisc = 0;
        cout<<"AP = ";
        for(int j=0; j<gridCriteriaMeasure.cols; j++)
        {
            cout<<gridCriteriaMeasure.at<float>(i,j)<<"+ ";
            accatthisc = gridCriteriaMeasure.at<float>(i,j) + accatthisc;
        }
        cout<<"="<<accatthisc<<endl;
        if(accatthisc > bestacc )
        {
            bestacc = accatthisc;
            bestindx = i;
        }
    }

    if(bestacc == 0)
    {
        cout<<"no point creating this svm as all the precision values were 0"<<endl;
        return;
    }

    cout<<"Best index selected is "<<bestindx<<" with GridCostFunc = "<<bestacc<<endl;
    cout<<"The best index config is: C= "<<combination_C_pos_neg.at<float>(bestindx,0)<<"PosWeight = "<<combination_C_pos_neg.at<float>(bestindx,1)<<" NegWeight = "<<combination_C_pos_neg.at<float>(bestindx,2)<<endl;
    // create the best svm
    CvMat* classweight = cvCreateMat(1,2,CV_32FC1);
    float *data = classweight->data.fl;
    data[0] = combination_C_pos_neg.at<float>(bestindx,2);
    data[1] =  combination_C_pos_neg.at<float>(bestindx,1);
    CvSVM svm ;
    CvSVMParams params;
    params.kernel_type=CvSVM::LINEAR;
    params.svm_type=CvSVM::C_SVC;
    params.C=combination_C_pos_neg.at<float>(bestindx,0);
    params.class_weights = classweight;
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,1000000,1e-6);
    bool res=svm.train(alltraining,alllabels,cv::Mat(),cv::Mat(),params);
    string svmoutfile = svmfilenameis + "_svm";
    svm.save(svmoutfile.c_str());
    string prbest = svmfilenameis + "_train";
    float f1 = crossValParam(svm,alltraining,alllabels,prbest,prbest);
    cout<<"AP measure on training set is"<<f1<<endl;
    cout<<"Number of support vectors are "<<svm.get_support_vector_count()<<endl;
}

void getPos_Neg_Weights(vector<float>& posWeights,vector<float>& negWeights,int posExample, int negExample)
{

    if(posExample == -1 || negExample == -1)
    {
        posWeights.push_back(1.0f);
        negWeights.push_back(1.0f);
        return ;
    }

    posWeights.push_back(1.0f);
    //posWeights.push_back(0.95f);
     posWeights.push_back(1.05f);

    float ratio = posExample / float(negExample);

    negWeights.push_back(ratio);
    negWeights.push_back(1.15f*ratio);
    //negWeights.push_back(0.85f*ratio);
    negWeights.push_back(1.10f*ratio);
    negWeights.push_back(0.90f*ratio);
}

void gridSearch(cv::Mat& training,cv::Mat& traininglabels,cv::Mat& validation,cv::Mat& validationlabels,vector<float> Cvalues,vector<float> posWeights,vector<float> negWeights,cv::Mat &metric,int nindex)
{
//   cout<<"Inside grid search"<<endl;
    int configdone = 0;
    for(int i=0; i<Cvalues.size(); i++)
    {
        for(int j=0; j<posWeights.size(); j++)
        {
            for(int k=0; k<negWeights.size(); k++)
            {
                CvMat* classweight = cvCreateMat(1,2,CV_32FC1);
                float *data = classweight->data.fl;

                data[0] =  negWeights[k];
                data[1] =  posWeights[j];;

                CvSVM svm ;
                CvSVMParams params;
                params.kernel_type=CvSVM::LINEAR;
                //params.kernel_type=CvSVM::RBF;
                params.svm_type=CvSVM::C_SVC;
                params.C=Cvalues[i];
                params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,1000000,1e-6);
                params.class_weights = classweight;

                bool res=svm.train(training,traininglabels,cv::Mat(),cv::Mat(),params);
                //        float crossValParamValue = validationResult(svm,validation,validationlabels);
                float crossValParamValue = crossValParam(svm,validation,validationlabels);
                //      cout<<svm.get_support_vector_count()<<endl;
                metric.at<float>(configdone,nindex) = crossValParamValue;
                configdone++;
            }
        }
    }
}

float svmTest_eachClass_PR(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,vector<pair<string, pair<int,int> > > cat_video_shot, string PRfilename, string LabelMarginfile)
{
    vector<float> margins(validation.rows);
    vector<float> groundTruth(validation.rows);

    for(int i=0; i<validation.rows; i++)
    {
        float predmargin = svm.predict(validation.row(i),true);
        margins[i] = predmargin;
        groundTruth[i] = validationlabels.at<float>(i,0);
    }
    float ap=0;
    vector<float> precision(groundTruth.size() + 1);
    vector<float> recall(groundTruth.size() + 1);
    vector<int> rankingorder(groundTruth.size());
    classificPrecisionRecall(groundTruth, margins, precision,recall, ap, rankingorder);
// lets plot the precision curves !
    if(PRfilename !="")
    {
        stringstream prfile;
        prfile<<"PR_"<<PRfilename;
        ofstream PR(prfile.str().c_str(),ios::out);
        for(int i=0; i<precision.size(); i++)
        {
            PR<<precision[i]<<"\t"<<recall[i]<<endl;
        }
        PR.close();
    }
    if(LabelMarginfile !="")
    {
        stringstream lmfile;
        lmfile<<"LabelMargin_"<<LabelMarginfile;
        ofstream LM(lmfile.str().c_str(),ios::out);
        for(int i=0; i<rankingorder.size(); i++)
        {
            LM<<groundTruth[rankingorder[i]]<<"\t"<<margins[rankingorder[i]]<<"\t";
            LM<<cat_video_shot[rankingorder[i]].first<<"\t"<<cat_video_shot[rankingorder[i]].second.first<<"\t"<<cat_video_shot[rankingorder[i]].second.second<<endl;
        }
        LM.close();
    }
//   cout<<"AP: "<<ap<<endl;
    return ap;
}

float crossValParam(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,string PRfilename, string LabelMarginfile)
{
    vector<float> margins(validation.rows);
    vector<float> groundTruth(validation.rows);

    for(int i=0; i<validation.rows; i++)
    {
        float predmargin = svm.predict(validation.row(i),true);
        margins[i] = predmargin;
        groundTruth[i] = validationlabels.at<float>(i,0);
    }
    float ap=0;
    vector<float> precision(groundTruth.size() + 1);
    vector<float> recall(groundTruth.size() + 1);
    vector<int> rankingorder(groundTruth.size());
    classificPrecisionRecall(groundTruth, margins, precision,recall, ap, rankingorder);
// lets plot the precision curves !
    if(PRfilename !="")
    {
        stringstream prfile;
        prfile<<"PR_"<<PRfilename;
        ofstream PR(prfile.str().c_str(),ios::out);
        for(int i=0; i<precision.size(); i++)
        {
            PR<<precision[i]<<"\t"<<recall[i]<<endl;
        }
        PR.close();
    }
    if(LabelMarginfile !="")
    {
        stringstream lmfile;
        lmfile<<"LabelMargin_"<<LabelMarginfile;
        ofstream LM(lmfile.str().c_str(),ios::out);


        for(int i=0; i<rankingorder.size(); i++)
        {
            LM<<groundTruth[rankingorder[i]]<<"\t"<<margins[rankingorder[i]]<<endl;
        }
        LM.close();
    }
//   cout<<"AP: "<<ap<<endl;
    return ap;
}

float validationResult(CvSVM &svm, cv::Mat& validation, cv::Mat& validationlabels,vector<float> &results)
{
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;

    int right_pos = 0;
    int right_neg = 0;
    int total_pos = 0;
    int total_neg = 0;
    int num_correct_val = 0;
    int num_validation = validation.rows;

    for(int i=0; i<num_validation; i++)
    {
        float pred = svm.predict(validation.row(i));
        //     cout<<"Prediction done"<<endl;
        if(validationlabels.at<float>(i,0) == positiveLabelValue)
        {
            total_pos ++;
            if(pred == positiveLabelValue)
            {
                TP ++;
                right_pos++;
                num_correct_val ++ ;
            }
            else
            {
                FN ++;
            }
        }
        else
        {
            total_neg++;
            if(pred == negativeLabelValue)
            {
                right_neg++;
                num_correct_val ++ ;
                TN++;
            }
            else
            {
                FP ++;
            }
        }
    }
    float precision = 0;
    float recall = 0;
    float f1 = 0;
    float berror = FP/float(total_neg) + FN/float(total_pos);

    // if(TP > ceil(0.2*total_pos) && TN > ceil(0.2*total_neg))
    // {
//    if(TP != 0 && TN != 0)
//   {
    precision =TP / float((TP + FP)) *100.0f ;
    recall = TP / float((TP + FN)) * 100.0f;
    f1 = 2.0*precision*recall / (precision + recall);
    cout<<"TP: "<<TP<<"\t TN: "<<TN<<"\t FP: "<<FP<<"\t FN: "<<FN;
    cout<<" F1: "<<f1<<" P= "<<precision<<" R= "<<recall<<" Berror= "<<berror<<endl;
    //}
    float accuracy = num_correct_val*100.0f/num_validation;

    results.push_back(precision);
    results.push_back(recall);
    results.push_back(accuracy);
    return precision;
}
int getNindx()
{
    // it gives numbers 0,1,2 in a pseudo random fashion

    int max2 = 3*1000;

    int r = (rand() % max2);
    //  cout<<r<<endl;
    if(r < 1000)
        return 0;
    else if(r < 2000)
        return 1;
    else
        return 2;
}

void readPositiveCategoryHistogram_mulFeatures(int N, string category,vector<string> featurename,vector<cv::Mat>& codewordsP,vector<cv::Mat>& labelsP)
{
    // arrange the shots in order of their num of shots
    vector<int> V;
    vector<int> S;
    // vector<float> numFeat;
    //readingFeat_readVSFileName(V,S,numFeat,category,"train");
    writingFeat_readVSFileName(V,S,category,"train");
    vector<int> videoindx;
    getVideoIndx_youtubedata_sorted(videoindx,category, "train");

    for(int i=0; i<N; i++)
    {
        codewordsP[i] = cvCreateMat(0,dictionarySize*featurename.size(),CV_32FC1);
        labelsP[i] = cvCreateMat(0,1,CV_32FC1);
    }
    vector<int> numInEachSplit(N);
    int numvideo = videoindx.size();
    int tv =0;
    for(tv=0; tv<numvideo; tv++)
    {
        //      cout<<category<<" "<<"video: "<<videoindx[tv]<<endl;
        int numShots = 0;
        vector<int> shots;

        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx[tv])
            {
                shots.push_back(S[vd]);
                numShots ++;
            }
        }

        int indxN = std::min_element(numInEachSplit.begin(),numInEachSplit.end()) - numInEachSplit.begin();
        numInEachSplit[indxN] = numInEachSplit[indxN] + numShots;
        readHist_video_withLabels_youtube(codewordsP[indxN],labelsP[indxN],category,featurename,videoindx[tv],shots,positiveLabelValue);
        //readHist_video_withLabels_youtube(codewordsP[indxN],labelsP[indxN],videoindx[tv],numShots,category,featurename,shots,positiveLabelValue);
    }
}


void readNegativeCategoryHistogram_mulFeatures(int N, vector<string> category,vector<string> featurename,vector<cv::Mat>& codewordsN,vector<cv::Mat>& labelsN)
{

    for(int i=0; i<N; i++)
    {
        codewordsN[i] = cvCreateMat(0,dictionarySize*featurename.size(),CV_32FC1);
        labelsN[i] = cvCreateMat(0,1,CV_32FC1);
    }
//   int globalvideocount = 0;

    for(int c=0; c<category.size(); c++)
    {
        cout<<"Reading negative codewords for "<<category[c]<<endl;
        int shotInCat = 0;
        //   cout<<"In category "<<category[c]<<endl;
        vector<int> videoindx;
        getVideoIndx_youtubedata_sorted(videoindx,category[c],"train");

        //  random_shuffle(videoindx.begin(),videoindx.end());
        vector<int> numInEachSplit(N);

        vector<int> V;
        vector<int> S;
        //vector<float> numFeat;
        writingFeat_readVSFileName(V,S,category[c],"train");

        for(int tv=0; tv<videoindx.size(); tv++)
        {
            //int indxN = globalvideocount % N ;
            //  globalvideocount ++;
            //      cout<<category[c]<<" "<<"video: "<<videoindx[tv]<<endl;
            int numShots = 0;
            vector<int> shots;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoindx[tv])
                {
                    shots.push_back(S[vd]);
                    numShots ++;
                }
            }
            shotInCat = shotInCat + numShots;
            int indxN = std::min_element(numInEachSplit.begin(),numInEachSplit.end()) - numInEachSplit.begin();
            numInEachSplit[indxN] = numInEachSplit[indxN] + numShots;
            readHist_video_withLabels_youtube(codewordsN[indxN],labelsN[indxN],category[c],featurename,videoindx[tv],shots,negativeLabelValue);
            //readHist_video_withLabels_youtube(codewordsN[indxN],labelsN[indxN],videoindx[tv],numShots,category[c],featurename,shots,negativeLabelValue);
        }
//        cout<<"Completed category "<<category[c]<<" with shots "<<shotInCat<<endl;
    }
}

// new function which reads histograms written individually for each shot
void readHist_video_withLabels_youtube(cv::Mat& histogram,cv::Mat& labels,string category, vector<string> featurename,int video,vector<int> shots,int label)
{
    int numShots = shots.size();
    cv::Mat temphistogram = cvCreateMat(numShots,dictionarySize*featurename.size(),CV_32FC1);
    cv::Mat templabels = cvCreateMat(numShots,1,CV_32FC1);
    for(int i=0; i<numShots; i++)
    {
        templabels.at<float>(i,0) = label;
    }
    for(int i=0; i<numShots; i++)
    {
        // for each shot ! read all the features !
        for(int f=0; f<featurename.size(); f++)
        {
            // read this featurename shots
            stringstream histfilename;
            histfilename<<getRootDir()<<category<<"/histogram/"<<dictionarySize<<"/hist_"<<category<<"_"<<featurename[f]<<"_"<<video<<"_"<<shots[i];

            ifstream hist(histfilename.str().c_str(),ios::in);
            if(!hist.good())
            {
                cout<<"Cannot read hist file "<<histfilename.str()<<endl;
                exit(0);
            }
            // add it to the big concatenated shot
            for(int j = f*dictionarySize; j<(f+1)*dictionarySize; j++)
            {
                hist>>temphistogram.at<float>(i,j) ;
            }
            hist.close();
        }
    }
    histogram.push_back(temphistogram);
    labels.push_back(templabels);

    // check if the read hist is the right one or not !!
    /*  ofstream testh("test_hist",ios::out);
      for(int i=0; i<numShots; i++)
      {
          for(int j=0; j<temphistogram.cols; j ++)
              testh<<temphistogram.at<float>(i,j)<<" ";
          testh<<endl;
      }
      testh.close();
      cin.get();
    */
}

// old function where all shots of the same video were in one file ! check  the other function for all shots seperately
void readHist_video_withLabels_youtube(cv::Mat& histogram,cv::Mat& labels, int video,int numShots,string category, vector<string> featurename,vector<int> shots,int label)
{
    cv::Mat temphistogram = cvCreateMat(numShots,dictionarySize*featurename.size(),CV_32FC1);
    cv::Mat templabels = cvCreateMat(numShots,1,CV_32FC1);
    for(int i=0; i<numShots; i++)
    {
        templabels.at<float>(i,0) = label;
    }

    float norm_factor = featurename.size();
    for(int i=0; i<featurename.size(); i++)
    {
        stringstream histfilename;
        histfilename<<getRootDir()<<category<<"/shothist_"<<category<<featurename[i]<<"_"<<video;
        ifstream hist(histfilename.str().c_str(),ios::in);
        //     cout<<"Reading file "<<histfilename.str()<<endl;

        if(!hist.good())
        {
            cout<<"Cannot read hist file "<<histfilename.str()<<endl;
            exit(0);
        }
        int n =0; // number of shots stored
        while(hist.good())
        {
            int shotnum;
            hist>>shotnum;
            //   cout<<"The shot number read is "<<shotnum<<endl;
            if(find(shots.begin(),shots.end(),shotnum) != shots.end())
            {
                float tempvalue;
                for(int j = i*dictionarySize; j<(i+1)*dictionarySize; j++)
                {
                    hist>>tempvalue;
                    temphistogram.at<float>(n,j) = tempvalue / norm_factor;
                }
                n++;
            }
            else
            {
                float notneeded;
                for(int kk=0; kk<dictionarySize; kk++)
                {
                    hist>>notneeded;
                }
            }
            if(n == numShots)
            {
                break;
            }
        }
    }
    histogram.push_back(temphistogram);
    labels.push_back(templabels);
    return ;
}
void getCvalueVectors_range(vector<float>& Cvalues)
{
    float i;
    for(i=0.1; i<400;)
    {
        Cvalues.push_back(i);
        i = i*2;
    }

}
