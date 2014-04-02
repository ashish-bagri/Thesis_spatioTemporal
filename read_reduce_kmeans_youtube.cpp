#include "read_reduce_kmeans.h"
#include "youtubedata.h"
#include <iterator>
#include <algorithm>
void do_read_reduce_youtube(string featurename,int featureDimension);
void read_reduce_youtube()
{
    //   do_read_reduce_youtube("hog",featureDimension_hog);
    do_read_reduce_youtube("baseline",featureDimension_baseline);
}

void do_read_reduce_youtube(string featurename, int featureDimension)
{
    cout<<"Inside read reduce kmeans youtube for feature "<<featurename<<endl;

    string essentialname = "essential_" + featurename;
    ofstream essential(essentialname.c_str(),ios::app);
    // do for each feature multiple times ! to get the sense of which is the best feature
    {
        cv::Mat features = cvCreateMat(0,featureDimension,CV_32FC1);

        read_reduce_youtube_features("positive", "train", featurename, features, 1000);
        cout<<"Total features read is "<<features.rows<<endl;

        read_reduce_youtube_features("negative", "train", featurename, features, 1000);
        cout<<"Total features read is "<<features.rows<<endl;

        cout<<"Start k means "<<endl;

        cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension,CV_32FC1);
        cv::Mat labels = cvCreateMat(features.rows,1,CV_16U);
        for(int i=0; i<labels.rows; i++)
        {
            labels.at<int>(i,0) = 0;
        }
        double* retkmeans =  doKmeans(features,dictionary,dictionarySize,labels);
        cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

        string dictionaryname = "youtube_" + featurename + "_1";
        writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],dictionaryname);

        ofstream labelfile("kmeansLabels_outside",ios::out);
        for(int i=0; i<labels.rows; i++)
        {
            labelfile<<labels.at<int>(i,0)<<endl;
        }
        labelfile.close();

        cout<<"Kmeans with "<<dictionarySize<< " cluster centers : The iteration 1 had an average distance of "<<retkmeans[1]<<endl;
        essential<<"The iteration 1 had an average distance of "<<retkmeans[1]<<endl;
        cout<<"completed writing the dictionary"<<endl;
    }
}

void read_reduce_youtube_features(string pos_neg, string train_test, string featurename, cv::Mat& features, int maxFeat)
{
    // this function reads all the features for train/test and pos/neg for all the categories that are there in the respective pos/neg !
    // it reads the type of feature - featurename
    int featDimension = features.cols;
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
//       cout<<"Inside category "<<category[i]<<endl;

        vector<int> videoIndx;
        getVideoIndx_youtubedata(videoIndx,category[i],train_test);

        // read vst file for this category
        vector<int> V;
        vector<int> S;
        writingFeat_readVSFileName(V,S,category[i],train_test);

        // for each video of this category, read the shots for it
        for(int tv=0; tv<videoIndx.size(); tv++)
        {
            // read through the v file.. if it has tv, then get the corresponding shot
            //    cout<<"Inside for video "<<videoIndx[tv]<<endl;
            int shotsdone = 0;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(shotsdone > 5)
                    continue;
                if(V[vd] == videoIndx[tv])
                {
                    shotsdone++;

                    cv::Mat videoFeat;
                    int shotforthisvideo = S[vd];
                    cout<<"Category: "<<category[i]<<" Shot: "<<shotforthisvideo<<" Video: "<<videoIndx[tv]<<endl;
                    if(featurename == "baseline")
                    {
                        string savefile = getBaselineFeatureFileName(category[i],videoIndx[tv],shotforthisvideo);
                        videoFeat = read_youtubeBaselineFeatures(savefile,featDimension,10699);
                    }
                    else
                    {
                        // have the shot number ! // have the video number !!
                        string savefilename = getMotionFeatureFileName(category[i],videoIndx[tv],shotforthisvideo,featurename);
                        videoFeat = read_youtubeBaselineFeatures(savefilename,featDimension);
                    }
                    //  string shotdirstr = getShotsDir_youtube(videoIndx[tv], shotforthisvideo, category[i]);
                    //   string imagefilelist = shotdirstr +  "imagelist";




                    //             string savefile=  getFeatureFileName(shotdirstr, videoIndx[tv], shotforthisvideo);
                    //   videoFeat.push_back(read_youtubeBaselineFeatures(imagefilelist,shotdirstr,savefile));
                    //videoFeat = read_youtubeBaselineFeatures(imagefilelist,shotdirstr,savefile);



                    cout<<"The number of features read is "<<videoFeat.rows<<endl;
                    if(maxFeat == -1)
                    {
                        features.push_back(videoFeat);
                    }
                    else if(videoFeat.rows > maxFeat)
                    {
                        cout<<"Reducing features to "<<maxFeat<<endl;
                        features.push_back(reduceFeatures(videoFeat,maxFeat));
                    }
                    else
                    {
                        features.push_back(videoFeat);
                    }
                    cout<<"Current count of features is "<<features.rows<<endl;
                }
            }
            //     cout<<"Completed for video "<<videoIndx[tv]<<endl;
        }
        //cout<<"Completed category "<<category[i]<<endl;
    }
}
cv::Mat readFeatOfAShot_youtube(string filename, string featurename)
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
    in>>numrows>>numcols;
    cv::Mat tempfeat = cvCreateMat(numrows,numcols,CV_32FC1);

    for(int i=0; i<numrows; i++)
    {
        for(int j=0; j<numcols; j++)
        {
            in>>tempfeat.at<float>(i,j);
        }
    }
    in.close();
    return tempfeat;
}
// new function where feature name is just the video and the shot number and is in a common location
cv::Mat read_youtubeBaselineFeatures(string featurefilename,int featDimension,int numOfFeatures)
{
    ifstream des(featurefilename.c_str(),ios::in);
    if(!des.good())
    {
        cout<<"Cannot open file "<<featurefilename<<endl;
        exit(0);
    }
    if(numOfFeatures !=-1)
    {
        cv::Mat descriptors = cvCreateMat(numOfFeatures,featDimension,CV_32FC1);
        for(int a=0; a<numOfFeatures; a++)
        {
            for(int j=0; j<featDimension; j++)
            {
                des>>descriptors.at<float>(a,j);
            }
        }
        des.close();
        cout<<"Number of features read is "<<descriptors.rows<<endl;
        return descriptors;
    }
    else
    {
        vector<float> allvalues;
        std::copy(
            std::istream_iterator<float>(des),
            std::istream_iterator<float>(),
            std::back_inserter(allvalues));
        int numRows = allvalues.size()/featDimension;
        cout<<"Number of features read is "<<numRows<<endl;
        int n = 0;
        cv::Mat descriptors = cvCreateMat(numRows,featDimension,CV_32FC1);
        for(int a=0; a<descriptors.rows; a++)
        {
            for(int j=0; j<descriptors.cols; j++)
            {
                descriptors.at<float>(a,j) = allvalues[n++];
            }
        }
        des.close();
        allvalues.clear();
        return descriptors;
    }
}
// old version where the feature name had the middle frame index
cv::Mat read_youtubeBaselineFeatures(string imagefilelist, string dirToGetShot, string dirToSaveFeat)
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

    return descriptors;
}
/*
int read_reduce_kmeans_baseline_youtube()
{
    vector<string> positivecat;
    vector<string> negativecat;
    posCatNames(positivecat);
    negCatNames(negativecat);

    cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

    for(int i=0; i<positivecat.size(); i++)
    {
        int maxFeat = 1200;
        string imagelocation = getBaselineLocation(positivecat[i]);
        string listname = imagelocation + "middleframes";
        ifstream middleframes(listname.c_str(),ios::in);
        while(middleframes.good())
        {
            string imagename;
            middleframes>>imagename;
            if(imagename == "" || imagename == " ")
                continue;
            string img = getActualName(imagename);

            string descFileName = imagelocation +  img + "_siftDesc";
            string keyFileName = imagelocation + img + "_siftKeypoint";

            ifstream des(descFileName.c_str(),ios::in);
            if(!des.good())
            {
                cout<<"Cannot open file "<<descFileName<<endl;
                return -1;
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
            if(maxFeat == -1)
            {
                allfeatures.push_back(descriptors);
            }
            else
            {
                // reduce features
                cv::Mat reducedFeat = reduceFeatures(descriptors, maxFeat);
                cout<<"Reduced features to "<<reducedFeat.rows<<endl;
                allfeatures.push_back(reducedFeat);
            }
        }

    }
    for(int i=0; i<negativecat.size(); i++)
    {
        int maxFeat = 400;
        string imagelocation = getBaselineLocation(negativecat[i]);
        string listname = imagelocation + "middleframes";
        ifstream middleframes(listname.c_str(),ios::in);
        while(middleframes.good())
        {
            string imagename;
            middleframes>>imagename;
            if(imagename == "" || imagename == " ")
                continue;
            string img = getActualName(imagename);

            string descFileName = imagelocation +  img + "_siftDesc";
            string keyFileName = imagelocation + img + "_siftKeypoint";

            ifstream des(descFileName.c_str(),ios::in);
            if(!des.good())
            {
                cout<<"Cannot open file "<<descFileName<<endl;
                return -1;
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
            if(maxFeat == -1)
            {
                allfeatures.push_back(descriptors);
            }
            else
            {
                // reduce features
                cv::Mat reducedFeat = reduceFeatures(descriptors, maxFeat);
                cout<<"Reduced features to "<<reducedFeat.rows<<endl;
                allfeatures.push_back(reducedFeat);
            }
        }
    }
    // all the features available. rn kmeans
    cout<<"Start k means "<<endl;
    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
    cv::Mat labels = cvCreateMat(allfeatures.rows,1,CV_32FC1);

    double* retkmeans =  doKmeans(allfeatures,dictionary,dictionarySize,labels);
    cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

    writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"baseline_youtube");
    cout<<"completed writing the dictionary"<<endl;
}
*/

