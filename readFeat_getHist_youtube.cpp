#include "readFeat_getHist.h"
#include "youtubedata.h"


// read features for a category level and video level.. one file for each video

void readfeat_getHist_youtubedata()
{
    // for the video 1 of the bird

 //   do_readfeat_gethist_youtube("positive","train","baseline","selected_shots2.list");
    do_readfeat_gethist_youtube("negative","test","baseline","selected_shots2.list");

}

void do_readfeat_gethist_youtube(string pos_neg, string train_test, string featurename, string shotsfilename)
{
    // this function reads all the features for train/test and pos/neg for all the categories that are there in the respective pos/neg !
    // it reads the type of feature - featurename
    //cv::Mat reduceFeatures(cv::Mat& Feat, int maxFeatures);
    cout<<"Read features for "<<pos_neg<<" examples for "<<train_test<<" for feature "<<featurename<<" and extract histogram"<<endl;
    cv::Mat dictionary;
    int featuredimension;
    if(featurename == "hog")
    {
        dictionary = cvCreateMat(dictionarySize,featureDimension_hog,CV_32FC1);
        featuredimension = featureDimension_hog;
    }
    else if(featurename == "hof")
    {
        dictionary = cvCreateMat(dictionarySize,featureDimension_hof,CV_32FC1);
        featuredimension = featureDimension_hof;
    }
    else if(featurename == "mbh")
    {
        dictionary = cvCreateMat(dictionarySize,featureDimension_mbhXY,CV_32FC1);
        featuredimension = featureDimension_mbhXY;
    }
    else if(featurename == "baseline")
    {
        dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
        featuredimension = featureDimension_baseline;
    }
    stringstream dictname ;
    dictname<<"youtube_"<<featurename<<"_1_dictionary";
    readDictionary(dictionary,dictname.str());

    cout<<"Reading features for "<<pos_neg<<" examples for "<<train_test<<endl;
    cout<<"The features read are "<<featurename<<endl;
    vector<string> category;
    if(pos_neg == "positive")
    {
        posCatNames(category);
    }
    else
    {
        negCatNames(category);
    }

    for(int i=0; i<category.size(); i++)
    {
        // read the training videos for this category !
        cout<<"Inside category "<<category[i]<<endl;
        vector<int> videoIndx;
        getVideoIndx_youtubedata(videoIndx,category[i],train_test);
        // read vst file for this category
        vector<int> V;
        vector<int> S;
        writingFeat_readVSFileName(V,S,category[i],train_test);
        for(int tv=0; tv<videoIndx.size(); tv++)
        {
            //cout<<"Inside for video "<<videoIndx[tv]<<endl;
            cv::Mat videoFeat;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoIndx[tv])
                {
                    int shotforthisvideo = S[vd];
                    cout<<"Video "<<videoIndx[tv]<<" Shot: "<<shotforthisvideo<<endl;
                    string codewordfilename = getCodewordfilename_youtube_eachshot(category[i],featurename,videoIndx[tv],shotforthisvideo);
                    ofstream codewords(codewordfilename.c_str(),ios::out);

                    string shotdirstr = getShotsDir_youtube(videoIndx[tv],shotforthisvideo,category[i]);
                    string imagefilelist = shotdirstr +  "imagelist";
                    cv::Mat code;
                    string savefile;
                    if(featurename == "baseline")
                    {

                        savefile = getBaselineFeatureFileName(category[i],videoIndx[tv],shotforthisvideo);
                        code = readFeatofAshot_gethist_youtube(savefile,dictionary,featuredimension,12699);
                    }
                    else
                    {
                        savefile = getMotionFeatureFileName(category[i],videoIndx[tv],shotforthisvideo,featurename);
                        code = readFeatofAshot_gethist_youtube(savefile,dictionary,featuredimension);
                    }

                    for(int j=0; j<dictionarySize; j++)
                    {
                        codewords<<code.at<float>(0,j)<<"\t";
                    }
                    codewords<<endl;
                    codewords.close();
                }
            }
        }
    }
}
cv::Mat readFeatofAshot_gethist_youtube(string featurefilename,cv::Mat &dictionary,int featDimension,int numOfFeatures)
{

    cv::Mat tempfeat = read_youtubeBaselineFeatures(featurefilename,featDimension,numOfFeatures);

    cv::Mat codeword = cvCreateMat(1,dictionarySize,CV_32FC1);
    featureToCodeword(tempfeat,dictionary,codeword);
    return codeword;
}
/*
cv::Mat readFeatofAshot_gethist_youtube(string filename, string featurename, int numfeat, int featdimension,cv::Mat &dictionary)
{
    stringstream featfile;
    featfile<<filename<<"_"<<featurename;
    ifstream in(featfile.str().c_str(),ios::in);

    if(!in.good())
    {
        cout<<"cannot read from the file "<<featfile.str()<<endl;
        exit(0);
    }
    int numrows, numcols;
    numrows = numfeat;
    numcols = featdimension;

    cv::Mat tempfeat = cvCreateMat(numrows,numcols,CV_32FC1);

    for(int i=0; i<numrows; i++)
    {
        for(int j=0; j<numcols; j++)
        {
            in>>tempfeat.at<float>(i,j);
        }
    }
    in.close();
    cv::Mat codeword = cvCreateMat(1,dictionarySize,CV_32FC1);
    featureToCodeword(tempfeat,dictionary,codeword);
    return codeword;
}
*/
/*
cv::Mat readFeatofAshot_gethist_baseline_youtube(string imagefilelist, string dirToGetShot, string dirToSaveFeat, int featdimension,cv::Mat &dictionary)
{
    vector<string> imagelist;
    readImagefileList(imagefilelist,dirToGetShot,imagelist);

    cout<<"Number of frames in this shot is "<<imagelist.size()<<endl;
    int middleframe = imagelist.size() / 2;

    string imgfullname = imagelist[middleframe];

    stringstream descFileName;
    descFileName<<dirToSaveFeat<<"_"<<middleframe<<"_siftDesc";
    cout<<"Read descriptor file : "<<descFileName.str()<<endl;

    ifstream des(descFileName.str().c_str(),ios::in);
    if(!des.good())
    {
        cout<<"Cannot open file "<<descFileName<<endl;
        exit(0);
    }
    cv::Mat descriptors = cvCreateMat(50325,featureDimension_baseline,CV_32FC1);
    for(int a=0; a<50325; a++)
    {
        for(int j=0; j<featureDimension_baseline; j++)
        {
            des>>descriptors.at<float>(a,j);
        }
    }
    des.close();
    cout<<"Number of features read is "<<descriptors.rows<<endl;


    cv::Mat codeword = cvCreateMat(1,dictionarySize,CV_32FC1);
    featureToCodeword(descriptors,dictionary,codeword);

    return codeword;
}

*/
