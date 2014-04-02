#include "youtubedata.h"
#include <stack>

string actualbasedir = "/media/Expansion Drive/ferrari_data/categories/";
//string actualbasedir = "/media/BC9D-FB31/histograms/";
void posCatNames(vector<string> &poscategory)
{
    ifstream poscat("positiveCat",ios::in);
    if(!poscat.good())
        cout<<"cannot open file "<<"positiveCat"<<endl;

    while(poscat.good())
    {
        string p;
        poscat>>p;
        if(p == "" || p == " ")
            continue;
        poscategory.push_back(p);
    }
    poscat.close();
}

void negCatNames(vector<string> &negcategory)
{
    ifstream negcat("negativeCat",ios::in);
    if(!negcat.good())
        cout<<"cannot open file "<<"negativeCat"<<endl;

    while(negcat.good())
    {
        string p;
        negcat>>p;
        if(p == "" || p == " ")
            continue;
        negcategory.push_back(p);
    }
    negcat.close();
}

string fourFormatNumber(int number)
{
    stringstream numbertoret;
    if(number < 10)
    {
        numbertoret<<0<<0<<0<<number;
    }
    else if(number < 99)
    {
        numbertoret<<0<<0<<number;
    }
    else if (number < 999)
    {
        numbertoret<<0<<number;
    }
    else
    {
        numbertoret<<number;
    }

    return numbertoret.str();
}


string threeFormatNumber(int number)
{
    stringstream numbertoret;
    if(number > 99)
    {
        numbertoret<<number;
        return numbertoret.str();
    }

    if(number < 10)
    {
        numbertoret<<0<<0<<number;
    }
    else
        numbertoret<<0<<number;

    return numbertoret.str();
}

bool sortShots(pair<int,int> i, pair<int,int> j)
{
    if(i.second > j.second)
        return true;
    else
        return false;
}

void getVideoIndx_youtubedata_sorted(vector<int> & videoindx, string category, string train_test)
{
    vector<int> videoindx1;
    vector< pair<int,int> > video_shot;
    getVideoIndx_youtubedata(videoindx1,category,train_test);

    vector<int> V;
    vector<int> S;
    writingFeat_readVSFileName(V,S,category,train_test);

    int tv =0;
    int numvideo = videoindx1.size();
    for(tv=0; tv<numvideo; tv++)
    {
        int numShots = 0;
        for(int vd=0; vd<V.size(); vd++)
        {
            if(V[vd] == videoindx1[tv])
            {
                numShots ++;
            }
        }
        video_shot.push_back(pair<int,int>(videoindx1[tv],numShots));
    }
    // now arrange them in order
    sort(video_shot.begin(),video_shot.end(),sortShots);
    // now the new video index is !!
    for(int i=0; i<video_shot.size(); i++)
    {
        // cout<<video_shot[i].first<<":"<<video_shot[i].second<<endl;
        videoindx.push_back(video_shot[i].first);
    }
}

void getVideoIndx_youtubedata(vector<int> & videoindx, string category, string train_test)
{
    string train_test_filename;
    if(train_test == "train")
    {
        train_test_filename = "trainvideo";
    }
    else if( train_test == "test")
    {
        train_test_filename = "testvideo";
    }
    else
    {
        cout<<"Wrong option ! "<<train_test<<endl;
        exit(0);
    }
    string traincatfile = actualbasedir +  category + "/" + train_test_filename;
  //  cout<<"Reading video indexes from file "<<traincatfile<<endl;
    ifstream trainvidfile(traincatfile.c_str(),ios::in);
    if(!trainvidfile.good())
    {
        cout<<"Cannot open file "<<traincatfile<<endl;
        exit(0);
    }

    int d = 0;
    int prev_tr = 0;
    do
    {
        int t;
        trainvidfile>>t;
        if(d != 0 && prev_tr == t )
            continue;
        d ++;
        prev_tr = t;
        videoindx.push_back(t);
    }
    while(trainvidfile.good());
    trainvidfile.close();
    cout<<"The number of videos are "<<videoindx.size()<<endl;
}

void writingFeat_readVSFileName(vector<int> &V, vector<int> &S,string category, string train_test)
{
    string vstfilename ;

    vstfilename = actualbasedir + category + "/selected_shots2.list";
    ifstream vstfile(vstfilename.c_str(),ios::in);
    if(!vstfile.good())
    {
        cout<<"Cannot open file "<<vstfilename<<endl;
        exit(0);
    }
    while(vstfile.good())
    {
        int v,s;
        vstfile>>v>>s;
        V.push_back(v);
        S.push_back(s);
    }
    if(V.size() > 2)
    {
        if(V[V.size()-1] == V[V.size() - 2])
        {
            if(S[S.size()-1] == S[S.size()- 2])
            {
                V.erase(V.begin() + V.size()-1);
                S.erase(S.begin() + S.size()-1);
            }
        }
    }
    vstfile.close();
 //   cout<<"Read vst file with rows "<<V.size()<<endl;
}
/*
void readingFeat_readVSFileName(vector<int> &V, vector<int> &S,vector<float>& numFeatures,string category, string train_test)
{
    string vstfilename ;
    if(train_test == "train")
    {
        vstfilename = actualbasedir + category + "/my_selected_shots.list";
    }
    else if (train_test == "test")
    {
        vstfilename = actualbasedir + category + "/my_selected_shots.list";
    }
    else
    {
        cout<<"Wrong option ! "<<train_test<<endl;
        exit(0);
    }

    ifstream vstfile(vstfilename.c_str(),ios::in);
    if(!vstfile.good())
    {
        cout<<"Cannot open file "<<vstfilename<<endl;
        exit(0);
    }
    while(vstfile.good())
    {
        int v,s;
        float t;
        vstfile>>v>>s>>t;
        V.push_back(v);
        S.push_back(s);
        numFeatures.push_back(t);
    }

    if(V.size() > 2)
    {
        if(V[V.size()-1] == V[V.size() - 2])
        {
            if(S[S.size()-1] == S[S.size()- 2])
            {
                V.erase(V.begin() + V.size()-1);
                S.erase(S.begin() + S.size()-1);
                numFeatures.erase(numFeatures.begin() + numFeatures.size()-1);
            }
        }
    }
    vstfile.close();
    cout<<"Read vst file with rows "<<V.size()<<endl;
}
*/

string getFeatureFileName(string shotsdir, int video, int shot)
{
    stringstream savefile;
    savefile<<shotsdir<<video<<"_"<<shot;
    return savefile.str();
}

string getShotsDir_youtube(int videoNumber, int shotNumber, string category)
{
    stringstream shotdirstr;
    shotdirstr<<actualbasedir<<category<<"/data/"<<fourFormatNumber(videoNumber)<<"/shots/"<<threeFormatNumber(shotNumber)<<"/";
    return shotdirstr.str();
}

string getBaselineFeatLocation(string category)
{
    string ret = actualbasedir + category + "/baseline_new_features/";
    return ret;
}
string getMotionFeatLocation(string category)
{
    string ret = actualbasedir + category + "/motion_features_new/";
    return ret;
}

string getBaselineImageLocation(string category)
{
    string ret = actualbasedir + category + "/baseline_new_images/";
    return ret;
}

string getMotionFeatureFileName(string category, int video, int shot,string featurename)
{
    stringstream savefile;
    savefile<<getMotionFeatureFileLocation(category,video,shot)<<"_"<<featurename;
    return savefile.str();
}

string getMotionFeatureFileLocation(string category, int video, int shot)
{
    stringstream savefile;
    savefile<<getMotionFeatLocation(category)<<video<<"_"<<shot;
    return savefile.str();
}

string getFeatureFileLocation(string category, int video, int shot)
{
    stringstream savefile;
    savefile<<getBaselineFeatLocation(category)<<video<<"_"<<shot;
    return savefile.str();
}

string getBaselineFeatureFileName(string category, int video, int shot)
{
    stringstream savefile;
    savefile<<getBaselineFeatLocation(category)<<video<<"_"<<shot<<"_baseline";
    return savefile.str();
}

string getBaselineImageFileName(string category, int video, int shot)
{
    stringstream savefile;
    savefile<<getBaselineImageLocation(category)<<video<<"_"<<shot<<"_baseimage";
    return savefile.str();
}


string getCodewordfilename_youtube(string category,int video,string featurename)
{
    stringstream codewordfilename;
    codewordfilename<<actualbasedir << category << "/hist_" <<category<<featurename<<"_"<<video;
    return codewordfilename.str();
}

string getCodewordfilename_youtube_eachshot(string category,string featurename,int video,int shot)
{
    stringstream codewordfilename;
    codewordfilename<<actualbasedir << category <<"/histogram/"<<dictionarySize<<"/hist_" <<category<<"_"<<featurename<<"_"<<video<<"_"<<shot;
    return codewordfilename.str();
}

string getRootDir()
{
    return actualbasedir;
}
/*
void getNnumberdStack(int N)
{
   int array[N];
    srand((unsigned)time(0));

    for(int i=0; i<size; i++){
        array[i] = (rand()%N)+1;
        cout << array[i] << endl;
 }

    stack<int> Nnums;


    Nnums.push(1);
    Nnums.push(2);
    Nnums.push(3);
    Nnums.push(4);
}
*/

