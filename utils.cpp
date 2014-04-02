#include "projectHeader.h"
#include "projectFunctions.h"


void getallconfig(vector<string> &configurations)
{
   //configurations.push_back("hof");
//    configurations.push_back("hog");
    configurations.push_back("mbh");
//    configurations.push_back("baseline");
    /*configurations.push_back("hog hof");
    configurations.push_back("hof mbh");
    configurations.push_back("mbh baseline");
    configurations.push_back("hof baseline");
*/
  /*  configurations.push_back("hog mbh");
    configurations.push_back("hog baseline");
   /* configurations.push_back("hog hof mbh baseline");
    configurations.push_back("hof mbh");
    configurations.push_back("hof mbh baseline");
    configurations.push_back("hog mbh baseline");
    configurations.push_back("hog hof mbh");*/
}

void getparts(string fullname,vector<string> &elements)
{
    stringstream ss(fullname);
    string buf;
    while(ss >> buf)
    {
        elements.push_back(buf);
    }

    for(int i=0; i<elements.size(); i++)
    {
        cout<<elements[i]<<endl;
    }
}

cv::Mat reduceFeatures(cv::Mat& Feat, int maxFeatures)
{
    if(Feat.rows < maxFeatures)
    {
        return Feat;
    }
    cv::Mat retFeat = cvCreateMat(0,Feat.cols,Feat.type());
// get a random number.. put that row in the new matrix.. return the new matrix
    vector<int> randomnumbers;
    srand((unsigned)time(0));
    int rd;
    do
    {
        rd = (rand() % Feat.rows);
        if(find(randomnumbers.begin(),randomnumbers.end(),rd) == randomnumbers.end())
        {
            // new random number !
            retFeat.push_back(Feat.row(rd));
            randomnumbers.push_back(rd);
        }
    }while(randomnumbers.size() < maxFeatures);

    if(retFeat.rows != maxFeatures)
    {
        cout<<"Was not able to fill the matrix with maxFetures ! The number of rows are "<<retFeat.rows<<endl;
    }

    return retFeat;
}
cv::Mat concatenateFeatures_reduce(cv::Mat &hog, cv::Mat& hof, cv::Mat &mbh,int maxFeatures)
{
    int combinedfeatdim = hog.cols + hof.cols + mbh.cols;
    if(hog.rows != hof.rows)
    {
        cout<<"rows mismatch "<<endl;
        exit(0);
    }
    if(hog.rows != mbh.rows)
    {
        cout<<"rows mismatch "<<endl;
        exit(0);
    }

    int total_rows = hog.rows;
    cout<<"Total features are "<<total_rows<<endl;
    cv::Mat combinedFeat = cvCreateMat(total_rows,combinedfeatdim,CV_32FC1);
    int col_count ;
    for(int i=0; i<total_rows; i++)
    {
        col_count = 0;
        for(int j=0; j<hog.cols; j++)
        {
            combinedFeat.at<float>(i,col_count++) =hog.at<float>(i,j);
        }
        //     cout<<"col_count after hog is "<<col_count<<endl;
        for(int j=0; j<hof.cols; j++)
        {
            combinedFeat.at<float>(i,col_count++) =hof.at<float>(i,j);
        }
        //   cout<<"col_count after hof is "<<col_count<<endl;
        for(int j=0; j<mbh.cols; j++)
        {
            combinedFeat.at<float>(i,col_count++) =mbh.at<float>(i,j);
        }
        // cout<<"col_count after mbh is "<<col_count<<endl;
    }

// reduce features to max
    if(maxFeatures == -1) // do not reduce !
    {
        return combinedFeat;
    }
    else
    {
        cv::Mat combineReduce = reduceFeatures(combinedFeat,maxFeatures);
        cout<<"Reduced features to "<<combineReduce.rows<<endl;
        return combineReduce;
    }
}
