#include "svm_test.h"
#include "read_reduce.h"
#include "youtubedata.h"
void PRAFmeasure(cv::Mat &allClassification,vector<string> categories,cv::Mat &PRA);
using namespace std;
void readVideoShotOrder(string category,vector<pair<int,int> >& video_shot, string train_test);
void readHistogram_youtube(string category,vector<string> featurename,cv::Mat& codewords,string train_test);
void readHistogram_youtube_new(string category,vector<string> featurename,cv::Mat& codewords,string train_test);
void SVM_test_singleClass_youtube()
{
    vector<string> categories;
    posCatNames(categories);
    vector<string> allconfigs;
    getallconfig(allconfigs);
    cout<<"size of all config is "<<allconfigs.size()<<endl;

    vector<string> negcategories;
    negCatNames(negcategories);


    for(int i=0; i<allconfigs.size(); i++)
    {
        vector<string> feature;
        getparts(allconfigs[i],feature);
        stringstream thisconfigname;
        for(int f=0; f<feature.size(); f++)
        {
            thisconfigname<<feature[f]<<"_";
        }
        cout<<"Running for config "<<thisconfigname.str()<<endl;
        stringstream svmfilenameis;
        svmfilenameis<<thisconfigname.str()<<categories[0]<<"Pos"<<"_svm";

        stringstream resultfilename;
        resultfilename<<thisconfigname.str()<<categories[0]<<"Pos_results";

        ofstream results(resultfilename.str().c_str(),ios::out);

        CvSVM svm ;
        svm.load(svmfilenameis.str().c_str());
        results<<"Positive Category "<<categories[0]<<endl;
        int num_test_pos = 0;
        int num_test_neg = 0;
        // read pos code
        cv::Mat allcodes = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
        cv::Mat groundTruth = cvCreateMat(0,1,CV_32FC1);
        {
            cout<<"Read codewords for category "<<categories[0]<<endl;
            cv::Mat code = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
            readHistogram_youtube_new(categories[0],feature,code,"test");
            cv::Mat label = cvCreateMat(code.rows,1,CV_32FC1);
            for(int tlab = 0; tlab<code.rows; tlab++)
            {
                label.at<float>(tlab,0) = 1;

            }
            allcodes.push_back(code);
            groundTruth.push_back(label);
            num_test_pos =code.rows;
        }
        // read neg codewords
        {
            for(int c=0; c<negcategories.size(); c++)
            {
                results<<"Negative Category "<<negcategories[c]<<endl;

                cout<<"Read codewords for category "<<negcategories[c]<<endl;
                cv::Mat code = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
                readHistogram_youtube_new(negcategories[c],feature,code,"test");
                cv::Mat label = cvCreateMat(code.rows,1,CV_32FC1);

                for(int tlab = 0; tlab<code.rows; tlab++)
                {
                    label.at<float>(tlab,0) = -1;

                }
                allcodes.push_back(code);
                groundTruth.push_back(label);
            }
            num_test_neg = allcodes.rows - num_test_pos;
        }
        results<<"Number Support vectors "<<svm.get_support_vector_count()<<endl;
        results<<"C =  "<<svm.get_params().C<<endl;
        results<<"Number of Positive test examples: "<<num_test_pos<<endl;
        results<<"Number of Negative test examples: "<<num_test_neg<<endl;
        results<<"Total test examples: "<<num_test_pos + num_test_neg<<endl;
        stringstream prfilename;
        prfilename<<thisconfigname.str()<<categories[0]<<"Pos"<<"_test";
        float avgap = crossValParam(svm, allcodes, groundTruth,prfilename.str(),prfilename.str());
        vector<float> resultsvector;
        validationResult(svm,allcodes,groundTruth,resultsvector);
        results<<"Precision: "<<resultsvector[0]<<endl;
        results<<"Recall: "<<resultsvector[1]<<endl;
        results<<"Accuracy: "<<resultsvector[2]<<endl;
        results<<"Area under PR curve:  "<<avgap*100.0f<<endl;
    }
}

void SVM_test_youtubePR()
{
    vector<string> categories;
    posCatNames(categories);
    cout<<"Total number of categories are "<<categories.size();

    vector<string> allconfigs;
    getallconfig(allconfigs);
    cout<<"size of all config is "<<allconfigs.size()<<endl;

    for(int i=0; i<allconfigs.size(); i++)
    {
        vector<string> feature;
        getparts(allconfigs[i],feature);
        stringstream thisconfigname;
        for(int f=0; f<feature.size(); f++)
        {
            thisconfigname<<feature[f]<<"_";
        }
        cout<<"Running for config "<<thisconfigname.str()<<endl;
        vector<pair<string, pair<int,int> > > cat_video_shot;
        for(int s=0; s<categories.size(); s++)
        {
            // running for 1 classifier
            //if(categories[s]!="dog")
            //continue;
            stringstream svmfilenameis;
            svmfilenameis<<thisconfigname.str()<<categories[s]<<"Pos"<<"_svm";
            CvSVM svm ;
            svm.load(svmfilenameis.str().c_str());
            cout<<"Positive category is "<<categories[s]<<endl;
            // read all codewords with making s category as positive and rest negative
            cv::Mat allcodes = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
            cv::Mat groundTruth = cvCreateMat(0,1,CV_32FC1);
            for(int c=0; c<categories.size(); c++)
            {
                // read for category c[c] and get results for this category for all the svm's !
                vector<pair<int,int> > video_shot;
                cv::Mat code = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
                cout<<"Read codewords for category "<<categories[c]<<endl;
                readHistogram_youtube_new(categories[c],feature,code,"test");

                readVideoShotOrder(categories[c],video_shot,"test");
                // read code ! .. fill in labels as positive if it belongs to this category
                cv::Mat label = cvCreateMat(code.rows,1,CV_32FC1);

                for(int tlab = 0; tlab<code.rows; tlab++)
                {
                    if(c == s)
                    {
                        label.at<float>(tlab,0) = 1;
                    }
                    else
                    {
                        label.at<float>(tlab,0) = -1;
                    }
                    cat_video_shot.push_back(pair<string,pair<int,int> >(categories[c],pair<int,int>(video_shot[tlab].first,video_shot[tlab].second)));
                }
                allcodes.push_back(code);
                groundTruth.push_back(label);
            }
            // completed reading all codewords..
            // get PR curves for this classifier
            if(allcodes.rows != groundTruth.rows)
            {
                cout<<"Mismatch"<<endl;
                exit(0);
            }
            stringstream prfilename;
            prfilename<<thisconfigname.str()<<categories[s]<<"Pos"<<"_test";
          //  crossValParam(svm, allcodes, groundTruth,prfilename.str(),prfilename.str());
          svmTest_eachClass_PR(svm, allcodes,groundTruth,cat_video_shot,prfilename.str(),prfilename.str());
        }
    }
}

void SVM_test_youtube_mulCategory()
{
    vector<string> categories;
    posCatNames(categories);
    cout<<"Total number of categories are "<<categories.size();

    vector<string> allconfigs;
    getallconfig(allconfigs);
    cout<<"size of all config is "<<allconfigs.size()<<endl;

    for(int i=0; i<allconfigs.size(); i++)
    {
        vector<string> feature;
        getparts(allconfigs[i],feature);
        cv::Mat allClassification = cvCreateMat(categories.size(),categories.size(),CV_32FC1);

        for(int i=0; i<allClassification.rows; i++)
        {
            for(int j=0; j<allClassification.cols; j++)
            {
                allClassification.at<float>(i,j) = 0;
            }
        }
        stringstream thisconfigname;
        for(int f=0; f<feature.size(); f++)
        {
            thisconfigname<<feature[f]<<"_";
        }
        cout<<"Running for config "<<thisconfigname.str()<<endl;
        vector<vector<pair<string, pair<int,int> > > > inDifferentCategories(categories.size());
        vector<vector<float> > inDifferentCategoriesMargins(categories.size());

        for(int c=0; c<categories.size(); c++)
        {
            // read for category c[c] and get results for this category for all the svm's !

            cout<<"Read codewords for category "<<categories[c]<<endl;
            vector<pair<int,int> > video_shot;
            cv::Mat code = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);
            // readHistogram_youtube(categories[c],feature,code,"test");
            readHistogram_youtube_new(categories[c],feature,code,"test");

            readVideoShotOrder(categories[c],video_shot,"test");
            if(code.rows != video_shot.size())
            {
                cerr<<"Mismatch "<<endl;
                exit(0);
            }
            cv::Mat margins = cvCreateMat(code.rows,categories.size(), CV_32FC1);
            for(int s=0; s<categories.size(); s++)
            {
                vector<float> thisSVMMargin(code.rows);
                stringstream svmfilenameis;
                svmfilenameis<<thisconfigname.str()<<categories[s]<<"Pos"<<"_svm";
                svm_getMargins(svmfilenameis.str(),code,thisSVMMargin);
                // add the margin into the margins
                for(int m=0; m<thisSVMMargin.size(); m++)
                {
                    margins.at<float>(m,s) = thisSVMMargin[m];
                }
            }
            // all margins for this category read. make a decision on each of the codewords, and update the category matrix !
            stringstream marginfile;
            marginfile<<thisconfigname.str()<<categories[c]<<"Pos_Margins";
            ofstream marginsout(marginfile.str().c_str(),ios::out);
            marginsout.setf(ios::fixed, ios::floatfield);
            marginsout.precision(5);
            marginsout<<"Video \t Shot \t";
            for(int cwrite=0; cwrite<categories.size(); cwrite++)
            {
                marginsout<<categories[cwrite]<<"\t";
            }
            marginsout<<endl;
            cout<<"Write SVM margins in file "<<marginfile.str()<<endl;

            for(int mr=0; mr<margins.rows; mr++)
            {
                // write margins for this config into file !
                // a decision is to be taken for each row (ie, each codeword)
                marginsout<<video_shot[mr].first<<"\t";
                marginsout<<video_shot[mr].second<<"\t";
                float bestmargin = 10000;
                int bestindx = 0;

                for(int mc=0; mc<margins.cols; mc++)
                {
                    if(margins.at<float>(mr,mc) < bestmargin) //change here !
                    {
                        bestmargin = margins.at<float>(mr,mc);
                        bestindx = mc;
                    }
                    marginsout<<margins.at<float>(mr,mc)<<"\t";
                }
                marginsout<<"ActualCategory: "<<categories[c]<<"\t"<<"Selected:"<<categories[bestindx];
                inDifferentCategories[bestindx].push_back(pair<string,pair<int,int> >(categories[c],pair<int,int>(video_shot[mr].first,video_shot[mr].second)));
                inDifferentCategoriesMargins[bestindx].push_back(bestmargin);
                marginsout<<endl;
                // update the allClassification matrix..
                allClassification.at<float>(c,bestindx) = allClassification.at<float>(c,bestindx) + 1;
            }
            marginsout.close();
            cout<<"Completed category "<<categories[c]<<endl;
        }
        // write the allclassification
        stringstream confusionMat;
        confusionMat<<thisconfigname.str()<<"confusionMat";
        ofstream confusion(confusionMat.str().c_str(),ios::out);
        confusion<<"\t";
        for(int x=0; x<categories.size(); x++)
        {
            confusion<<categories[x]<<"\t";
        }
        confusion<<endl;
        for(int cl = 0; cl<allClassification.rows; cl++)
        {
            confusion<<categories[cl]<<"\t";
            for(int cc =0; cc<allClassification.cols; cc++)
            {
                confusion<<allClassification.at<float>(cl,cc)<<"\t";
            }
            confusion<<endl;
        }

        // writing precision recall of the confusion matrix
        cv::Mat PRA = cvCreateMat(categories.size(),4,CV_32FC1);
        PRAFmeasure(allClassification,categories,PRA);
        confusion<<"Category \t\t Precision \t Recall \t Accuracy \t Fmeasure"<<endl;
        for(int pra = 0; pra<PRA.rows; pra++)
        {
            confusion<<categories[pra]<<"\t\t"<<PRA.at<float>(pra,0)<<"\t"<<PRA.at<float>(pra,1)<<"\t"<<PRA.at<float>(pra,2)<<"\t"<<PRA.at<float>(pra,3)<<"\t"<<endl;
        }
        confusion.close();

        // writing confusion matrix in a different format
        /*    for(int xx=0; xx<categories.size(); xx++)
            {
                stringstream catfileconfname;
                catfileconfname<<categories[xx]<<"_confusionMat";
                ofstream catfile(catfileconfname.str().c_str(),ios::app);
                catfile<<thisconfigname.str()<<"\t";
                for(int yy=0; yy<categories.size(); yy++)
                {
                    catfile<<allClassification.at<float>(xx,yy)<<"\t";
                }
                catfile<<endl;
                catfile.close();
            }
            // writing results in a different format
            for(int xx=0; xx<categories.size(); xx++)
            {
                stringstream catfileconfname;
                catfileconfname<<categories[xx]<<"_Results";
                ofstream catfile(catfileconfname.str().c_str(),ios::app);
                catfile<<thisconfigname.str()<<"\t";
                for(int yy=0; yy<PRA.cols; yy++)
                {
                    catfile<<PRA.at<float>(xx,yy)<<"\t";
                }
                catfile<<endl;
                catfile.close();
            }
            */

        // writing the confusion between whom and whom with shot details
        for(int xx=0; xx<categories.size(); xx++)
        {
            string filename = "selectedAs" + categories[xx];
            ofstream f(filename.c_str(),ios::out);
            for(int lc=0; lc<inDifferentCategories[xx].size(); lc++)
            {
                f<<"Category: "<<inDifferentCategories[xx][lc].first<<" Video: "<<inDifferentCategories[xx][lc].second.first<<" Shot:"<<inDifferentCategories[xx][lc].second.second<<" Margin: "<<inDifferentCategoriesMargins[xx][lc]<<endl;

            }
            f.close();
        }

    }



}

void SVM_test_youtube()
{
    /*    vector<string> poscat;
        posCatNames(poscat);

        vector<string> negcat;
        negCatNames(negcat);
    */
    vector<string> allcat;
    posCatNames(allcat);
    for(int pcat = 0; pcat<1; pcat++)
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


        vector<string> allconfigs;
        getallconfig(allconfigs);


        cout<<"size of all config is "<<allconfigs.size()<<endl;
        for(int ac=0; ac<allconfigs.size(); ac++)
        {
            vector<string> feature;
            getparts(allconfigs[ac],feature);

            stringstream svmfilenameis;

            for(int f=0; f<feature.size(); f++)
            {
                svmfilenameis<<feature[f]<<"_";
            }
            svmfilenameis<<positivecat<<"Pos";

            cv::Mat codeword = cvCreateMat(0,dictionarySize*feature.size(),CV_32FC1);

            readHistogram_youtube(positivecat,feature,codeword,"test");

            int num_pos = codeword.rows;

            readHistogram_youtube(negcat,feature,codeword,"test");
            int num_neg = codeword.rows - num_pos;

            cout<<"Total positive test is "<<num_pos<<endl;
            cout<<"Total negative test is "<<num_neg<<endl;
            cout<<"Total test is "<<codeword.rows<<endl;
            cv::Mat labels = cvCreateMat(codeword.rows,1,CV_32FC1);
            for(int i=0; i<num_pos; i++)
            {
                labels.at<float>(i,0) = positiveLabelValue; // pos category
            }
            for(int i=num_pos; i<labels.rows; i++)
            {
                labels.at<float>(i,0) = negativeLabelValue; // neg category
            }

            // read all features !
            // do testing
            string svmfile = svmfilenameis.str() + "_svm";
            vector<float> accuracies(3);

            //  doSVMTest(svmfile, codeword, labels , accuracies,svmfilenameis);
            svmTest_threshold(svmfile, codeword, labels, svmfilenameis.str());
            //svmtest_distancemargin(svmfile, codeword, labels, svmfilenameis.str());
        }

    }
}
// new version with one shot per file
void readHistogram_youtube_new(string category,vector<string> featurename,cv::Mat& codewords,string train_test)
{
    //  cout<<"reading codewords for category "<<category<<endl;

    vector<int> videoindx;
    getVideoIndx_youtubedata(videoindx,category,train_test);
    vector<int> V;
    vector<int> S;

    writingFeat_readVSFileName(V,S,category,train_test);

    for(int tv=0; tv<videoindx.size(); tv++)
    {
        //    cout<<category<<" Video :"<<videoindx[tv]<<endl;
        vector<int> shots;
        int numShots = 0;
        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx[tv])
            {
                shots.push_back(S[vd]);
                numShots ++;
            }
        }
        readHist_video_youtube(codewords,category,featurename,videoindx[tv],shots);
    }
}
void readVideoShotOrder(string category,vector<pair<int,int> >& video_shot, string train_test)
{
    vector<int> videoindx;
    getVideoIndx_youtubedata(videoindx,category,train_test);
    vector<int> V;
    vector<int> S;

    writingFeat_readVSFileName(V,S,category,train_test);

    for(int tv=0; tv<videoindx.size(); tv++)
    {
        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx[tv])
            {
                video_shot.push_back(pair<int,int>(V[vd],S[vd]));
            }
        }
    }
}

void readHistogram_youtube(vector<string> category,vector<string> featurename,cv::Mat& codewords,string train_test)
{
    cout<<"Reading codewords"<<endl;

    for(int c=0; c<category.size(); c++)
    {
        cout<<"In category "<<category[c]<<endl;
        vector<int> videoindx;
        getVideoIndx_youtubedata(videoindx,category[c],train_test);
        vector<int> V;
        vector<int> S;
        vector<float> numFeat;
        //    readingFeat_readVSFileName(V,S,numFeat,category[c],train_test);

        for(int tv=0; tv<videoindx.size(); tv++)
        {
            vector<int> shots;
            int numShots = 0;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoindx[tv])
                {
                    shots.push_back(S[vd]);
                    numShots ++;
                }
            }
            readHist_video_youtube(codewords,videoindx[tv],numShots,category[c],featurename,shots);
        }
    }
    cout<<"Total codewords read is "<<codewords.rows<<" having dimension "<<codewords.cols<<endl;
}
void readHistogram_youtube(string category,vector<string> featurename,cv::Mat& codewords,string train_test)
{
    cout<<"reading codewords for category "<<category<<endl;

    vector<int> videoindx;
    getVideoIndx_youtubedata(videoindx,category,train_test);
    vector<int> V;
    vector<int> S;
    vector<float> numFeat;
    //  readingFeat_readVSFileName(V,S,numFeat,category,train_test);

    for(int tv=0; tv<videoindx.size(); tv++)
    {
//           cout<<"Inside for video "<<videoindx[tv]<<endl;
        vector<int> shots;
        int numShots = 0;
        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx[tv])
            {
                shots.push_back(S[vd]);
                numShots ++;
            }
        }
        readHist_video_youtube(codewords,videoindx[tv],numShots,category,featurename,shots);
    }
}
void readHist_video_youtube_rewrite(int video,int numShots,string category, vector<string> featurename,vector<int> shots);
void dorewrite(vector<string> category,vector<string> featurename, string train_test);
void rewriteHist()
{
    vector<string> feature;
    feature.push_back("hog");
    feature.push_back("hof");
    feature.push_back("mbh");
    feature.push_back("baseline");

    vector<string> poscat;
    posCatNames(poscat);

    vector<string> negcat;
    negCatNames(negcat);

    dorewrite(poscat,feature,"train");
    dorewrite(poscat,feature,"test");
}
void dorewrite(vector<string> category,vector<string> featurename, string train_test)
{
    for(int c=0; c<category.size(); c++)
    {
        cout<<"In category "<<category[c]<<endl;
        vector<int> videoindx;
        getVideoIndx_youtubedata(videoindx,category[c],train_test);
        vector<int> V;
        vector<int> S;
        vector<float> numFeat;
        //  readingFeat_readVSFileName(V,S,numFeat,category[c],train_test);

        for(int tv=0; tv<videoindx.size(); tv++)
        {
            cout<<"Inside for video "<<videoindx[tv]<<endl;
            vector<int> shots;
            int numShots = 0;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoindx[tv])
                {
                    shots.push_back(S[vd]);
                    numShots ++;
                }
            }
            cout<<"Video: "<<videoindx[tv]<<". NumShots: "<<numShots<<endl;
            readHist_video_youtube_rewrite(videoindx[tv],numShots,category[c],featurename,shots);
        }
    }
}

void readHist_video_youtube_rewrite(int video,int numShots,string category, vector<string> featurename,vector<int> shots)
{
    float norm_factor = featurename.size();
    for(int i=0; i<featurename.size(); i++)
    {
        stringstream histfilename;
        histfilename<<getRootDir()<<category<<"/hist_"<<category<<featurename[i]<<"_"<<video;
        ifstream hist(histfilename.str().c_str(),ios::in);
        //     cout<<"Reading file "<<histfilename.str()<<endl;

        if(!hist.good())
        {
            cout<<"Cannot read hist file "<<histfilename.str()<<endl;
            exit(0);
        }
        cv::Mat temphistogram = cvCreateMat(numShots,dictionarySize,CV_32FC1);
        for(int n=0; n<numShots; n++)
        {
            float tempvalue;
            for(int j = 0; j<dictionarySize; j++)
            {
                hist>>tempvalue;
                temphistogram.at<float>(n,j) = tempvalue;
            }
        }
        hist.close();
        cout<<"Creating new codewords"<<endl;
        stringstream histfilename2;
        histfilename2<<getRootDir()<<category<<"/shothist_"<<category<<featurename[i]<<"_"<<video;

        ofstream temppp(histfilename2.str().c_str(),ios::out);
        for(int x=0; x<numShots; x++)
        {
            temppp<<shots[x]<<"\t";
            for(int y=0; y<temphistogram.cols; y++)
            {
                temppp<<temphistogram.at<float>(x,y)<<"\t";
            }
            temppp<<endl;
        }
        temppp.close();
    }
    return ;
}

void PRAFmeasure(cv::Mat &allClassification,vector<string> categories,cv::Mat &PRA)
{

// get these values seperately for each of the categories !
    for(int i=0; i<categories.size(); i++)
    {
        float TP = 0;
        float FN = 0;
        float FP = 0;
        float allincat = 0;

        TP = allClassification.at<float>(i,i);
        for(int j=0; j<allClassification.cols; j++)
        {
            if(i!=j)
            {
                FN += allClassification.at<float>(i,j);
            }
        }

        for(int j=0; j<allClassification.rows; j++)
        {
            if(i!=j)
            {
                FP += allClassification.at<float>(j,i);
            }
        }
        allincat = FN + TP;

        PRA.at<float>(i,0) = TP / (TP+FP);
        PRA.at<float>(i,1) = TP / (TP+FN);
        PRA.at<float>(i,2) = TP / allincat;
        PRA.at<float>(i,3) = PRA.at<float>(i,0) * PRA.at<float>(i,1) * 2 / (PRA.at<float>(i,0) + PRA.at<float>(i,1)); // fmeasure
    }
}
