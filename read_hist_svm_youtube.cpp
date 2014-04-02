#include "youtubedata.h"
#include "readHist_SVM.h"
#include "read_reduce.h"
using namespace std;
/**
Adjust class weights to take into account the number of samples.
Cross validation for class weights

Also::
Splitting them randomly ! so read all training codewords together with the labels !


**/

// do 2 things::
// 1. combine all positibve and negative and then spilt them keeping their labels intact. ie, do not spilt positive and negative seperately
// 2. do N classification svm instead of 2 class classification

// can do it tonite ?? challenge !! - 4 hrs coding marathon
// managing class weights

// changing the number of train and test  !


// function which reads the histogram and creates svm of either individual or combined of 2 or more features;
//void readHist_svm_youtube()
//{
//    cout<<"Read histogram and combine features";
// for youtube data, i have histograms  video wise..
/**
read all codewords of positive category. -- >
need to split them into N sets > per video ? randomly in terms of shots ?
have 18 videos. split into 6 sets ? with 3 videos in each set ?
train on 5, val on 1.. for N times..
for negative examples..
cat 5
cow 6
bird  9
horse  6
into 6 sets >>
go to the first read video of bird --> select one number from 1-N.. assign it to that set
if all finish, start from begining ..
reading in histogram and dividing them into N sets simultaneously !
how about the combination codewords ?
first combine and then split them ! so a sperate function can be written to combine them as and when they are being assigned to a number of N !
**/

void readHist_getSVM_youtube()
{
    vector<string> poscat;
    posCatNames(poscat);
    int N = 3;
    vector<string> negcat;
    negCatNames(negcat);

    vector<string> allconfigs;
    getallconfig(allconfigs);
    cout<<"size of all config is "<<allconfigs.size()<<endl;
    for(int i=0; i<allconfigs.size(); i++)
    {
        vector<string> feature;
        getparts(allconfigs[i],feature);
        stringstream svmfilenameis;
        for(int f=0; f<feature.size(); f++)
        {
            svmfilenameis<<feature[f]<<"_";
        }
        svmfilenameis<<poscat[0]<<"Pos";

        vector<cv::Mat> codewordsP(N);
        vector<cv::Mat> labelsP(N);

        readPositiveCategoryHistogram_mulFeatures(N,poscat[0],feature,codewordsP);
        int totalP = 0;
        // fill the labels vector
        for(int i=0; i<N; i++)
        {
            int row = codewordsP[i].rows;
            cout<<"In index "<<i<<", The number of histograms are "<<row<<endl;
            totalP = totalP + row;
            labelsP[i] = cvCreateMat(row,1,CV_32FC1);
            for(int j=0; j<row; j++)
            {
                labelsP[i].at<float>(j,0) = 1;
            }
        }
        cout<<"Total positive codewords is "<<totalP<<endl;
        cout<<"The codeword dimesion is "<<codewordsP[0].cols<<endl;

        vector<cv::Mat> codewordsN(N);
        vector<cv::Mat> labelsN(N);

        readNegativeCategoryHistogram_mulFeatures(N, negcat,feature,codewordsN);


        int totalN = 0;
        for(int i=0; i<N; i++)
        {
            int row = codewordsN[i].rows;
            cout<<"In index "<<i<<", The number of histograms are "<<row<<endl;
            totalN = totalN + row;
            labelsN[i] = cvCreateMat(row,1,CV_32FC1);
            for(int j=0; j<row; j++)
            {
                labelsN[i].at<float>(j,0) = -1;
            }
        }


        cout<<"The total negative codewords are "<<totalN<<endl;
        cout<<"The codeword dimesion is "<<codewordsN[0].cols<<endl;

        cv::Mat alltraining = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
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
        svmCrossVal(alltraining,codewordsP,labelsP,codewordsN, labelsN, N,  svmfilenameis.str(),accuracies, best_C_value);
        createAndWriteSvm(alltraining,alllabels,svmfilenameis.str(),best_C_value, false);
    }
}
// new version, one shot per file
void readHist_video_youtube(cv::Mat& histogram, string category, vector<string> featurename,int video,vector<int> shots)
{
      int numShots = shots.size();
    cv::Mat temphistogram = cvCreateMat(numShots,dictionarySize*featurename.size(),CV_32FC1);

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
}


// old version
void readHist_video_youtube(cv::Mat& histogram, int video,int numShots,string category, vector<string> featurename,vector<int> shots)
{
    cv::Mat temphistogram = cvCreateMat(numShots,dictionarySize*featurename.size(),CV_32FC1);
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
                // this is one of the shots !
                //           cout<<"This is one of the shots needed !"<<endl;
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
                //         cout<<"This shot is not needed !"<<endl;
                // just read off the value !
                float notneeded;
                for(int kk=0; kk<dictionarySize; kk++)
                {
                    hist>>notneeded;
                }
            }
            if(n == numShots)
            {
                //              cout<<"Read all the shots needed !"<<endl;
                break;
            }
        }
    }
    histogram.push_back(temphistogram);
    return ;
}


void readNegativeCategoryHistogram_mulFeatures(int N, vector<string> category,vector<string> featurename,vector<cv::Mat>& codewordsN)
{
    cout<<"Reading negative codewords"<<endl;
    for(int i=0; i<N; i++)
    {
        codewordsN[i] = cvCreateMat(0,dictionarySize*featurename.size(),CV_32FC1);
    }
    int globalvideocount = 0;

    for(int c=0; c<category.size(); c++)
    {
        int shotInCat = 0;
        cout<<"In category "<<category[c]<<endl;
        vector<int> videoindx;
        getVideoIndx_youtubedata(videoindx,category[c],"train");
        vector<int> V;
        vector<int> S;
        //vector<float> numFeat;
        writingFeat_readVSFileName(V,S,category[c],"train");

        for(int tv=0; tv<videoindx.size(); tv++)
        {
            cout<<"Inside for video "<<videoindx[tv]<<endl;
            cout<<"This video goes to the index "<<globalvideocount % N<<endl;
            int indxN = globalvideocount % N ;
            globalvideocount ++;
            int numShots = 0;
            vector<int> shots;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoindx[tv])
                {
                    //                    int shotforthisvideo = S[vd];
                    //                    cout<<"Shot :  "<<shotforthisvideo<<" for Video:  "<<videoindx[tv]<<endl;
                    shots.push_back(S[vd]);
                    numShots ++;
                }                // get codewords for this video !
            }
            shotInCat = shotInCat + numShots;
            cout<<"Video: "<<videoindx[tv]<<". NumShots: "<<numShots<<endl;

            // read features of each of the features and get a combined codeword !
            readHist_video_youtube(codewordsN[indxN],videoindx[tv],numShots,category[c],featurename,shots);
        }
        cout<<"Completed category "<<category[c]<<" with shots "<<shotInCat<<endl;
    }
}
void readPositiveCategoryHistogram_mulFeatures(int N, string category,vector<string> featurename,vector<cv::Mat>& codewordsP)
{
    srand((unsigned)time(0));
// read the number of videos. .
    cout<<"Reading positive codewords"<<endl;
    for(int i=0; i<N; i++)
    {
        codewordsP[i] = cvCreateMat(0,dictionarySize*featurename.size(),CV_32FC1);
    }
    vector<int> videoindx;
    getVideoIndx_youtubedata(videoindx,category,"train");

    vector<int> V;
    vector<int> S;
    //vector<float> numFeat;
    writingFeat_readVSFileName(V,S,category,"train");

    for(int tv=0; tv<videoindx.size(); tv++)
    {
        cout<<"Inside for video "<<videoindx[tv]<<endl;
        //    int randomnum = rand() % N ;
        int indxN = tv % N;
        cout<<"This video goes to the index "<<indxN<<endl;
        int numShots = 0;
        vector<int> shots;

        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx[tv])
            {
                //            int shotforthisvideo = S[vd];
                //         cout<<"Shot :  "<<shotforthisvideo<<" for Video:  "<<videoindx[tv]<<endl;
                shots.push_back(S[vd]);
                numShots ++;
            }
            // get codewords for this video !
        }
        cout<<"Video: "<<videoindx[tv]<<". NumShots: "<<numShots<<endl;
        // read multipled feature codeword
        readHist_video_youtube(codewordsP[indxN],videoindx[tv],numShots,category,featurename,shots);
    }

}


