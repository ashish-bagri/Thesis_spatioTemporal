#include "projectHeader.h"
#include "projectFunctions.h"

void getImagenamelist(string shotfilename,string imagedir, int num_examples,vector<string> &imageList)
{
    vector<int> startframes;
    vector<int> endframes;
    vector<int> keyframes;
    vector<int> sizes;

    readInputShotsFile(shotfilename,startframes,endframes,keyframes,sizes,num_examples);
    cout<<"The total number of  shots are "<<startframes.size()<<endl;
    for(int i=0; i<startframes.size(); i++)
    {
        int middle = (startframes[i] + endframes[i]) / 2 + 25;
        stringstream imagename;
        imagename<<imagedir<<"/image_"<<middle<<".jpg";
        imageList.push_back(imagename.str());
    }
}

string getActualName(string filename)
{
    int firstindx = filename.find_last_of("/");
    if(firstindx == filename.npos)
    {
        firstindx = 0;
    }
    else
    {
        firstindx = firstindx + 1;
    }

    int lastindx = filename.find_last_of(".");


    if(lastindx == filename.npos)
    {
        lastindx = filename.length() - 1;
    }

    int len = lastindx - firstindx;
    string actualstr = filename.substr(firstindx,len);
    return actualstr;
}

int readInputShotsFile(string inputshotfilename,vector<int>& startframes,vector<int>& endframes,vector<int>& keyframes,vector<int>& sizes,int num_max)
{
    cout<<"Inside  readInputShotsFile"<<endl;
    cout<<"Reading file "<<inputshotfilename<<endl;
    ifstream inputframenos(inputshotfilename.c_str(),ios::in);
    if(!inputframenos.good())
    {
        cout<<"Cannot open file "<<inputshotfilename<<endl;
        exit(0);
    }

    int prev_start, prev_end;
    int framenum = 0;
    int start_frame = 0;
    int end_frame = 1000000;
    int key = 0;
    int size = 0;
    do
    {
        // The first "shot"
        inputframenos>>start_frame;
        inputframenos>>end_frame;

        inputframenos>>key>>size;
        //    inputframenos>>inputlabel;
        if(framenum > 0)
        {
            if(start_frame == prev_start && end_frame == prev_end)
                continue;
        }

        prev_start = start_frame;
        prev_end = end_frame;

        startframes.push_back(start_frame);
        endframes.push_back(end_frame);
        keyframes.push_back(key);
        sizes.push_back(size);
        framenum++;
    }
    while(inputframenos.good());
    inputframenos.close();

    if(keyframes.size() > num_max)
    {
        keyframes.erase(keyframes.begin()+num_max,keyframes.end());
        startframes.erase(startframes.begin()+num_max,startframes.end());
        endframes.erase(endframes.begin()+num_max,endframes.end());
    }
    return framenum;
}
