#include "readHist_SVM.h"

void readHist_SVM()
{
    SVM_youtube();
}


void readHist_SVM_tiger()
{
    // do it for all the 3 features. and train pos and train neg !
    crossValidationSVM("baseline",NUM_POS_TRAIN,NUM_NEG_TRAIN,5);
    /*
     crossValidationSVM("combined1",NUM_POS_TRAIN,NUM_NEG_TRAIN,5);
     crossValidationSVM("hog",NUM_POS_TRAIN,NUM_NEG_TRAIN,5);
     crossValidationSVM("hof",NUM_POS_TRAIN,NUM_NEG_TRAIN,5);
     crossValidationSVM("mbh",NUM_POS_TRAIN,NUM_NEG_TRAIN,5);
     */
}

void readHist_SVM_leopard()
{
    crossValidationSVM("baseline",NUM_POS_TRAIN_LEOPARD,NUM_NEG_TRAIN_LEOPARD,5);
//    crossValidationSVM("hog",NUM_POS_TRAIN_LEOPARD,NUM_NEG_TRAIN_LEOPARD,5);
//    crossValidationSVM("hof",NUM_POS_TRAIN_LEOPARD,NUM_NEG_TRAIN_LEOPARD,5);
//    crossValidationSVM("mbh",NUM_POS_TRAIN_LEOPARD,NUM_NEG_TRAIN_LEOPARD,5);
}

string code_pre = "leopard_code_";

void crossValidationSVM(string featurename, int num_positive_codeswords, int num_negative_codeswords, int N)
{
    cout<<"Splitting the training examples into "<<N<<" sets"<<endl;
    cv::Mat PosCodes = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat PosLabels = cvCreateMat(0,1,CV_32FC1);

    string poscodewordfile = code_pre + featurename + "_trainpos";

    readcodewords(poscodewordfile, PosCodes,PosLabels,num_positive_codeswords);

    cout<<"Completed reading "<< "pos"<<" code words. Number read is "<<PosCodes.rows <<" and labels :"<<PosLabels.rows<<endl;
    vector<cv::Mat> codewordsP(N);
    vector<cv::Mat> labelsP(N);

    split_N_sets(codewordsP,labelsP, N,PosCodes, PosLabels, train_pos_shotfilename,num_positive_codeswords);
//   split_N_sets(codewordsP,labelsP, N,PosCodes, PosLabels, leopard_train_pos_shotfilename,num_positive_codeswords);

    cv::Mat NegCodes = cvCreateMat(0,dictionarySize,CV_32FC1);
    cv::Mat NegLabels = cvCreateMat(0,1,CV_32FC1);

    string negcodewordfile = code_pre + featurename + "_trainneg";

    readcodewords(negcodewordfile, NegCodes,NegLabels,num_negative_codeswords);

    cout<<"Completed reading "<< "neg"<<" code words. Number read is "<<NegCodes.rows <<" and labels :"<<NegLabels.rows<<endl;
    vector<cv::Mat> codewordsN(N);
    vector<cv::Mat> labelsN(N);
    split_N_sets(codewordsN,labelsN, N,NegCodes, NegLabels, train_neg_shotfilename,num_negative_codeswords);

    //split_N_sets(codewordsN,labelsN, N,NegCodes, NegLabels, leopard_train_neg_shotfilename,num_negative_codeswords);

// all training set !
    cv::Mat alltraining = cvCreateMat(0,dictionarySize,CV_32FC1);
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

    svmCrossVal(alltraining,codewordsP,labelsP,codewordsN, labelsN,  N,  featurename,accuracies, best_C_value);
    createAndWriteSvm(alltraining,alllabels,featurename,best_C_value, true);
}

// just splitting all the training code words into 5 equal sets at the key frame level.
// TESTED ! WORKS GOOD !
void split_N_sets(vector<cv::Mat>& codewords,vector<cv::Mat>& labels, int N,cv::Mat &allcodewords, cv::Mat &allLabels, string trainshotfile,int num_examples)
{
    // read pos shot file
    cout<<"Inside split N sets"<<endl;

    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;
    readInputShotsFile(trainshotfile,startframes,endframes,keyframes,sizes,num_examples);
    cout<<"Total number of frames read is "<<keyframes.size()<<endl;
// now split the keyfranes into N parts
    vector<int> uniqueShuffledKeyFrames;
    {
        vector<int> shuffledKeyIndx;
        getUniqueRandomNumbers(shuffledKeyIndx,0,keyframes.size(),keyframes.size());
        for(int k=0; k<shuffledKeyIndx.size(); k++)
        {
            if(find(uniqueShuffledKeyFrames.begin(),uniqueShuffledKeyFrames.end(),keyframes[shuffledKeyIndx[k]]) == uniqueShuffledKeyFrames.end())
            {
// this key frames not in there
                uniqueShuffledKeyFrames.push_back(keyframes[shuffledKeyIndx[k]]);
            }
        }
    }
    cout<<"Size of unique key frames is  "<<uniqueShuffledKeyFrames.size()<<endl;
    vector<int> countframes(N);

    cout<<"The dictionary size here is "<<allcodewords.cols<<endl;
    for(int i=0; i<N; i++)
    {
        countframes[i] = 0;
        codewords[i] = cvCreateMat(0,allcodewords.cols,CV_32FC1);
        labels[i] = cvCreateMat(0,allLabels.cols,CV_32FC1);
    }
    // split key frames .. and add it to the respective codewords .. do no go through all the keys (keep some margin to adjust at the end)
    int i=0;
    int end_first_round = (uniqueShuffledKeyFrames.size() / N) * N - N;

//    cout<<"The ending of the first round is to be at "<<end_first_round<<endl;

    for(i=0; i<end_first_round;)
    {
        // the key frames goes into nth section (so everything belgoinging to this key frame goes there )
        for(int n=0; n<N; n++)
        {
            int k = uniqueShuffledKeyFrames[i];
            //       cout<<"The "<<i<<" unique key frame is "<<k<<endl;
            int temp = 0;
            for(int j=0; j<keyframes.size(); j++)
            {
                if(keyframes[j] == k)
                {
                    temp++;
                    codewords[n].push_back(allcodewords.row(j));
                    labels[n].push_back(allLabels.row(j));
                    countframes[n]++;
                }
            }
//           cout<<"The number of shots associated with this key frames were "<<temp<<endl;
            i++;
        }
    }
    for(i =end_first_round; i<uniqueShuffledKeyFrames.size(); i++)
    {
        // find the minimum size split and add this to that..

        int minindx = std::min_element(countframes.begin(),countframes.end()) - countframes.begin();
        int k = uniqueShuffledKeyFrames[i];
        for(int j=0; j<keyframes.size(); j++)
        {
            if(keyframes[j] == k)
            {
                codewords[minindx].push_back(allcodewords.row(j));
                labels[minindx].push_back(allLabels.row(j));
                countframes[minindx]++;
            }
        }
    }
// finally the sizes are::
    /*
       cout<<"The final split sizes are as follows "<<endl;
       for(int s=0; s<N; s++)
       {
           cout<<"Count of frames : "<<countframes[s]<<endl;
       }
       */
}
void svmCrossVal(cv::Mat &alltraining,vector<cv:: Mat>& codewordsP,vector<cv::Mat>& labelsP ,vector<cv::Mat> &codewordsN, vector<cv::Mat>& labelsN, int N, string featurename,vector<float>& accuracies, float &best_C_value)
{
    cout<<"Inside svmCrossVal"<<endl;
    vector<string> cindxguide;
    vector<float> Cvalues;
    getCvalueVectors(Cvalues,cindxguide,alltraining);
    cv::Mat newaccuracy = cvCreateMat(Cvalues.size(),N,CV_32FC1);

    accuracies.resize(Cvalues.size());
    for(int a=0; a<accuracies.size(); a++)
    {
        accuracies[a] = 0;
    }
    int cols = alltraining.cols;

    for(int i=0; i<N; i++)
    {
        // index i becomes the validation set, the rest added together is the training set
        cout<<"-----------------------------------INSIDE ITERATION "<<i<<"-------------------------------------"<<endl;
        cv::Mat trainset = cvCreateMat(0,cols,CV_32FC1);
        cv::Mat valset = cvCreateMat(0,cols,CV_32FC1);
        cv::Mat trainlab = cvCreateMat(0,1,CV_32FC1);
        cv::Mat vallab = cvCreateMat(0,1,CV_32FC1);
        for(int j=0; j<N; j++)
        {
            if(i == j)
            {
                cout<<"MAKING "<<j<<" AS THE VALIDATION SET"<<endl;
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
        doSVM(trainset,valset,trainlab, vallab, featurename, Cvalues, cindxguide, newaccuracy,i);
    }
// write the data here in a organised manner
    cout<<"The C value and precision are "<<endl;
    float bestacc = 0;
    int bestindx = 0;
    for(int i=0; i<newaccuracy.rows; i++)
    {
        float accatthisc = 0;
        cout<<Cvalues[i]<<"\t \t"<<cindxguide[i]<<" \t \t";
        for(int j=0; j<newaccuracy.cols; j++)
        {
            accatthisc = newaccuracy.at<float>(i,j) + accatthisc;
            cout<<newaccuracy.at<float>(i,j)<<" + \t";
        }
        cout<<" = "<<accatthisc<<endl;
        if(accatthisc > bestacc )
        {
            bestacc = accatthisc;
            bestindx = i;
        }
    }

    best_C_value = Cvalues[bestindx];
    if(bestacc == 0)
    {
        cout<<"no point creating this svm as all the precision values were 0"<<endl;
        exit(0);
    }
    cout<<endl<<"The best combined C value was "<<Cvalues[bestindx]<<"( "<<cindxguide[bestindx]<<")"<<endl;
    cout<<"-----------------------------------------------------------------------------"<<endl;
}

void readcodewords(string inputcodefile, cv::Mat& codes,cv::Mat& labels,int num_codewords)
{
    cout<<"Reading histogram from file :"<<inputcodefile<<endl;
    ifstream inputcodes(inputcodefile.c_str(),ios::in);
    if(!inputcodes.good())
    {
        cout<<"cannot open file "<<inputcodefile<<endl;
        exit(0);
    }
    float debugginginfo = 0;
    for(int n=0; n<num_codewords; n++)
    {
//   while(!inputcodes.eof())
        //  {
        debugginginfo = 0;
        float label;
        cv::Mat tempcode = cvCreateMat(1,dictionarySize,CV_32FC1);
        inputcodes>>label;
        for(int i=0; i<dictionarySize; i++)
        {
            inputcodes>>tempcode.at<float>(0,i);
            debugginginfo = debugginginfo + tempcode.at<float>(0,i);
        }
        /*    if(debugginginfo < 0.95)
            {
                cout<<"Reading codewords #"<<n<<endl;
                cout<<"ALEART !! The sum of current codeword is "<<debugginginfo<<endl;
                cout<<"NOT ADDING THIS CODEWORD "<<endl;
            }
            else
            {*/
        codes.push_back(tempcode);
        labels.push_back(label);
        // }
    }
    inputcodes.close();
}

void doSVM(cv::Mat &training, cv::Mat& validation,cv::Mat& trainingLabels, cv::Mat& validationLabels, string featurename, vector<float> Cvalues, vector<string> cindxguide, cv::Mat& newaccuracies, int n)
{
    cout<<"The number of training values & labels are "<<training.rows<<"\t"<<trainingLabels.rows<<endl;
    cout<<"The number of validation values & labels are "<<validation.rows<<"\t"<<validationLabels.rows<<endl;
//   stringstream grph;
    //  grph<<featurename<<"_C_ROC";
    //  ofstream graph(grph.str().c_str(),ios::app);
    //  graph<<"**** new iteration ****"<<endl;
    map<float,float> graphvalues;

    int num_pos = 0;
    int num_neg = 0;

    float num_validation = validation.rows;
    float num_training = training.rows;

    for(int i=0; i<validationLabels.rows; i++)
    {
        if(validationLabels.at<float>(i,0)== 1)
            num_pos++;
        if(validationLabels.at<float>(i,0) == -1)
            num_neg++;
    }
    cout<<"In validation Set, Total Positive examples is "<<num_pos<<". Num Negative examples is "<<num_neg<<endl;
    vector<int> num_correct_match_validation;
// b . start with an estimate of parameters
    cout<<"The SVM details are: "<<endl;
    cout<<"C SVM, LINEAR KERNEL,cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,100000,1e-6)"<<endl;

    for(int c=0; c<Cvalues.size(); c++)
    {
        cout<<endl<<"Starting svm for C value "<<Cvalues[c]<<"("<<cindxguide[c]<<")"<<endl;
        CvSVM svm ;
        CvSVMParams params;
        params.kernel_type=CvSVM::LINEAR;
        params.svm_type=CvSVM::C_SVC;
        params.C=Cvalues[c];
        params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,1000000,1e-6);
// c. create an svm for it
        bool res=svm.train(training,trainingLabels,cv::Mat(),cv::Mat(),params);
// d. check the svm performance on the validation set

        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        int right_pos = 0;
        int right_neg = 0;
        int total_pos = 0;
        int total_neg = 0;
        int num_correct_val = 0;

        for(int i=0; i<num_validation; i++)
        {
            float pred = svm.predict(validation.row(i));
            if(validationLabels.at<float>(i,0) == 1)
            {
                total_pos ++;
                if(pred == 1)
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
                if(pred == -1)
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
        cout<<"TP: "<<TP<<"\t TN: "<<TN<<"\t FP: "<<FP<<"\t FN: "<<FN<<endl;
        float precision = 0;
        float recall = 0;
        float f1 = 0;
        if(TP != 0)
        {
            precision =TP / float((TP + FP)) *100.0f ;
            recall = TP / float((TP + FN)) * 100.0f;
            f1 = 2.0*precision*recall / (precision + recall);
        }
        cout<<"Precision: "<<precision<<endl;
        cout<<"Recall : "<<recall<<endl;
        cout<<"F measure : "<<f1<<endl;

        float accuracy = num_correct_val*100.0f/num_validation;
        cout<<"Accuracy : "<<accuracy<<endl;
        cout<<"The number of support vectors is"<<svm.get_support_vector_count()<<endl;

        newaccuracies.at<float>(c,n) = f1;
    }
}


void createAndWriteSvm(cv::Mat& alltraining, cv::Mat& allLabels,string featurename, float Cvalue, bool wantmetoshuffle)
{
    cout<<"Creating the final SVM and writing it on to file"<<endl;
    cout<<"Number of training samples is "<<alltraining.rows<<endl;
    cout<<"C value used is "<<Cvalue<<endl;

    CvSVM svm ;
    CvSVMParams params;
    params.kernel_type=CvSVM::LINEAR;
    params.svm_type=CvSVM::C_SVC;
    params.C=Cvalue;
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,100000,1e-6);
    if(wantmetoshuffle == true)
    {
        cv::Mat train = cvCreateMat(0,alltraining.cols,CV_32FC1);
        cv::Mat label = cvCreateMat(0,1,CV_32FC1);
        shuffleCode_Label(alltraining,allLabels,train,label);
        bool res=svm.train(train,label,cv::Mat(),cv::Mat(),params);
    }
    else
    {
        bool res=svm.train(alltraining,allLabels,cv::Mat(),cv::Mat(),params);
    }

    string svmoutfile = featurename + "_svm";

    svm.save(svmoutfile.c_str());
    {
        /*
        checking the performance of the svm itself on the training set */
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        int right_pos = 0;
        int right_neg = 0;
        int total_pos = 0;
        int total_neg = 0;
        int num_correct_train = 0;

        for(int i=0; i<alltraining.rows; i++)
        {
            float pred = svm.predict(alltraining.row(i));
            if(allLabels.at<float>(i,0) == 1)
            {
                total_pos ++;
                if(pred == 1)
                {
                    TP ++;
                    right_pos++;
                    num_correct_train ++ ;
                }
                else
                {
                    FN ++;
                }
            }
            else
            {
                total_neg++;
                if(pred == -1)
                {
                    right_neg++;
                    num_correct_train ++ ;
                    TN++;
                }
                else
                {
                    FP ++;
                }
            }
        }

        /*    for(int i=0; i<alltraining.rows; i++)
            {
                float pred = svm.predict(alltraining.row(i));
                if(pred == allLabels.at<float>(i,0))
                {
                    num_correct ++;
                }
            }
          */
        cout<<"TP: "<<TP<<"\t TN: "<<TN<<"\t FP: "<<FP<<"\t FN: "<<FN<<endl;
        float precision = 0;
        float recall = 0;
        if(TP != 0)
        {
            precision =TP / float((TP + FP)) *100.0f ;
            recall = TP / float((TP + FN)) * 100.0f;
        }

        cout<<"Precision: "<<precision<<endl;
        cout<<"Recall : "<<recall<<endl;
        cout<<"F Measure "<<2*precision*recall/(precision+recall)<<endl;
        float accuracy = num_correct_train*100.0f/alltraining.rows;
        cout<<"Accuracy : "<<accuracy<<endl;
        cout<<"Total positive is "<<total_pos<<endl;
        cout<<"Total negative is "<<total_neg<<endl;
        cout<<"The number of support vectors are "<<svm.get_support_vector_count()<<endl;
    }
}

void shuffleCode_Label(cv::Mat& codewords, cv::Mat& labels,cv::Mat& shuffledcodes, cv::Mat& shuffledLabels)
{
    int num_to_shuffle = codewords.rows;
    vector<int> shuffledindx;
    getUniqueRandomNumbers(shuffledindx, 0, num_to_shuffle, num_to_shuffle);
    for(int i=0; i<shuffledindx.size(); i++)
    {
        shuffledcodes.push_back(codewords.row(shuffledindx[i]));
        shuffledLabels.push_back(labels.row(shuffledindx[i]));
    }
    if(shuffledLabels.rows != labels.rows)
    {
        cout<<"shuffled index do not have the smae size. something is wrong"<<endl;
        exit(0);
    }
    if(num_to_shuffle != shuffledcodes.rows)
    {
        cout<<"shuffled index do not have the smae size. something is wrong"<<endl;
        exit(0);
    }
}

void getUniqueRandomNumbers(vector<int>& randomnumbers, int start, int end, int num_r)
{
    srand((unsigned)time(0));

    int random_integer =0;


    for(int i=0; i<num_r; i++)
    {
        random_integer = (rand() % end) + start;

        if(find(randomnumbers.begin(),randomnumbers.end(),random_integer) == randomnumbers.end())
        {
            randomnumbers.push_back(random_integer);
        }
        else
        {
            i = i-1;
        }
    }
}


void getCvalueVectors(vector<float>& Cvalues, vector<string>& cindxguide, cv::Mat &alltraining)
{
    float initialC = initial_C_estimate(alltraining);
    cout<<"The initial C value is " <<initialC<<endl;

    Cvalues.push_back(initialC);
    cindxguide.push_back("C");

    Cvalues.push_back(1.0f/initialC);
    cindxguide.push_back("1/C");

    int num_mul_factors = 7;    // C 1/C  2C C/2 2/C   4C C/4 4/C was 7
    int mulfactor = 1;

    for(int x=1; x<=num_mul_factors; x++)
    {
        mulfactor = mulfactor * 2;
        Cvalues.push_back(initialC * mulfactor);
        stringstream s1;
        s1<<mulfactor<< "C";
        cindxguide.push_back(s1.str());

        Cvalues.push_back(initialC/mulfactor);
        stringstream s3;
        s3<<"C/"<<mulfactor;
        cindxguide.push_back(s3.str());

        Cvalues.push_back(1.0f*mulfactor/(initialC));
        stringstream s2;
        s2<<mulfactor<< "/C";
        cindxguide.push_back(s2.str());
    }
}

double initial_C_estimate(cv::Mat& training)
{
// average distance between two descriptors.. it forms a starting point for C estimate
    double total_distance =0;
    int num_train = training.rows;
    for(int i=0; i<num_train; i++)
    {
        for(int j=i+1; j<num_train; j++)
        {
            total_distance = total_distance + cv::norm(training.row(i),training.row(j),cv::NORM_L2);
        }
    }
    int total_den = num_train * (num_train- 1 ) / 2;
    return (total_distance/total_den);
}

