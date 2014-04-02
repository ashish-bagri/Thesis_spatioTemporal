#include "extract_write.h"
#include "youtubedata.h"
#include "time.h"

void extract_write_youtubeFeatures()
{

//   vector<string> positivecat;
    //   posCatNames(positivecat);

    // cout<<"The number of positive category is "<<positivecat.size()<<endl;
    vector<pair<int,int> > vshot;
    string cat = "cat";
    //vshot.push_back(pair<int,int>(24,16));
    //vshot.push_back(pair<int,int>(11,4));
    vshot.push_back(pair<int,int>(13,2));
  //  vshot.push_back(pair<int,int>(17,6));

    for(int i=0; i<vshot.size(); i++)
    {
        cout<<cat<<"\t V = "<<vshot[i].first<<"\t"<<vshot[i].second<<endl;
        string shotdirstr = getShotsDir_youtube(vshot[i].first,vshot[i].second,cat);
        string imagefilelist = shotdirstr +  "imagelist";
        string motionfileloc = getMotionFeatureFileLocation(cat,vshot[i].first,vshot[i].second);
        clock_t start = clock();
//        string basesaveimg = getBaselineImageFileName(category[i],videoindx[tv],shotforthisvideo);
        int numfeatures = youtubeBaselineFeatures(imagefilelist, shotdirstr, "","");
        cout<<"Time taken is"<<float(clock() - start )/ CLOCKS_PER_SEC<<endl;
        //int numfeatures = doTracking_youtubeData(imagefilelist, shotdirstr, motionfileloc);
    }



//    extract_write_cat_features(positivecat,"train");
//   extract_write_cat_features(positivecat,"test");
}
void extract_write_cat_features(vector<string> category, string train_test)
{
    cout<<"Extract youtube features"<<endl;
//   ofstream numFeat("numFeat",ios::app);
    ofstream finalist("finallist",ios::app);
    ofstream ignore("ignorelist",ios::app);
    int countshot = 0;

    for(int i=0; i<category.size(); i++)
    {
        // read in the train video numbers for this positive category
        //      cout<<"extracting features for category "<<category[i]<<endl;
        vector<int> videoindx;
        getVideoIndx_youtubedata(videoindx,category[i],train_test);

        // read in the vst file

        vector<int> V;
        vector<int> S;
        //vector<float> numFeatures;
        writingFeat_readVSFileName(V,S,category[i],train_test);
        // for each  video, read the shots for it
        for(int tv=0; tv<videoindx.size(); tv++)
        {
            // read through the v file.. if it has tv, then get the corresponding shot
            //   cout<<"Inside for video "<<videoindx[tv]<<endl;
            for(int vd=0; vd<V.size(); vd++)
            {
                if(V[vd] == videoindx[tv])
                {
                    int shotforthisvideo = S[vd];

                    cout<<category[i]<<" Shot :  "<<shotforthisvideo<<" for Video:  "<<videoindx[tv]<<endl;
                    // get features for this shot for this video !
                    // what is the dir name where the images of the shots are ?
                    string shotdirstr = getShotsDir_youtube(videoindx[tv],shotforthisvideo,category[i]);


                    string imagefilelist = shotdirstr +  "imagelist";
//                    cout<<"Save file name is "<<savefile<<endl;
                    //string featuresavefile = getBaselineFeatureFileName(category[i],videoindx[tv],shotforthisvideo);
                    string motionfileloc = getMotionFeatureFileLocation(category[i],videoindx[tv],shotforthisvideo);

                    //   string basesaveimg = getBaselineImageFileName(category[i],videoindx[tv],shotforthisvideo);
                    //     int numfeatures = youtubeBaselineFeatures(imagefilelist, shotdirstr, featuresavefile,basesaveimg);
                    int numfeatures = doTracking_youtubeData_2(imagefilelist, shotdirstr, motionfileloc);

                    if(numfeatures == -1)
                    {
                        ignore<<videoindx[tv]<<" "<<shotforthisvideo<<endl;
                    }
                    else
                    {
                        finalist<<category[i]<<" "<<videoindx[tv]<<" "<<shotforthisvideo<<" "<<numfeatures<<endl;
                    }
                    countshot ++;
                }
            }
        }
    }
    cout<<"Total shot count is "<<countshot<<endl;
//   numFeat.close();
    ignore.close();
    finalist.close();
}
