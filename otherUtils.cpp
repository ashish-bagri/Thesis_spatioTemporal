#include <iostream>
#include <fstream>
#include "projectHeader.h"
#include "projectFunctions.h"
using namespace std;

void readAnalysisfile(string filename, vector<string>& data,int N)
{
    ifstream input(filename.c_str(),ios::in);
    if(!input.good())
    {
        cout<<"Cannot read file "<<filename<<endl;
        return;
    }
    int start,end,key;
    string part1,part2;
    for(int i=0;i<N;i++){
        input>>start>>end>>key>>part1;
    //    cout<<"Part1 is "<<part1;

        input>>part2;

        part1.append(part2);

        data.push_back(part1);

    }

    return;
}
void readBaselineAnalysis(string filename, vector<string>& data,int N)
{

    ifstream input(filename.c_str(),ios::in);
    if(!input.good())
    {
        cout<<"Cannot read file "<<filename<<endl;
        return;
    }

    string part1;
    for(int i=0;i<N;i++){
        input>>part1;
        data.push_back(part1);
    }
    return;
}

void readAllanalysis()
{
    int N = NUM_POS_TEST + NUM_NEG_TEST;

    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> size;
    readInputShotsFile(test_pos_shotfilename,startframes,endframes,keyframes,size,NUM_POS_TEST);

    vector<int> startframes2;
    vector<int> endframes2;
    vector<int> keyframes2;
    vector<int> size2;
    readInputShotsFile(test_neg_shotfilename,startframes2,endframes2,keyframes2,size2,NUM_NEG_TEST);


    vector<string> hog;
    readAnalysisfile("svm_analysis_hog",hog,N);

    vector<string> hof;
    readAnalysisfile("svm_analysis_hof",hof,N);

    vector<string> mbh;
    readAnalysisfile("svm_analysis_mbh",mbh,N);

    vector<string> baseline;
    readBaselineAnalysis("tiger_classificationCat_baseline",baseline,N);


    for(int i=0;i<NUM_POS_TEST;i++)
    {
        cout<<keyframes[i] + 25 <<"\t"<<(startframes[i]+endframes[i]) / 2 + 25<<"\t"<<hog[i]<<"\t"<<hof[i]<<"\t"<<mbh[i]<<"\t"<<baseline[i]<<endl;
    }


    for(int i=0;i<NUM_NEG_TEST;i++)
    {
        cout<<keyframes2[i]+25<<"\t"<<(startframes2[i]+endframes2[i]) / 2 + 25<<"\t"<<hog[i+NUM_POS_TEST]<<"\t"<<hof[i+NUM_POS_TEST]<<"\t"<<mbh[i+NUM_POS_TEST]<<"\t"<<baseline[i+NUM_POS_TEST]<<endl;
    }

}


























