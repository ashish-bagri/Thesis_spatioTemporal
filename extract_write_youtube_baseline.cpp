#include "extract_write.h"
#include "youtubedata.h"

void extract_write_youtube_baseline()
{
    vector<string> positivecat;
    vector<string> negativecat;
    posCatNames(positivecat);
    negCatNames(negativecat);

    //  doextract_write_youtube_baseline(positivecat);
 //  doextract_write_youtube_baseline(negativecat);
}

int youtubeBaselineFeatures(string imagefilelist, string dirToGetShot, string dirToSaveFeat, string dirToSaveImage)
{
    vector<string> imagelist;
    readImagefileList(imagefilelist,dirToGetShot,imagelist);
    if(imagelist.size() <= 15)
        return -1;
    //cout<<"Number of frames in this shot is "<<imagelist.size()<<endl;
    int middleframe = imagelist.size() / 2;

    string imgfullname = imagelist[middleframe];
   // cout<<"Read Image "<<imgfullname<<endl;

    IplImage* aframe = cvLoadImage(imgfullname.c_str());
    if(aframe == 0)
    {
        cout<<"Cannot open image "<<imgfullname<<endl;
        return -1;
    }
    IplImage* frame = cvCreateImage(cvSize(640,360),aframe->depth,aframe->nChannels);
    cvResize(aframe,frame,CV_INTER_CUBIC);
/*
    cvNamedWindow("BASEIMAGE");
    cvShowImage("BASELINE",frame);
    char k = cvWaitKey(0);
    cout<<"K is "<<k<<endl;

    if(k == 'n' || k == 'N' )
    {
        cout<<"Ignore"<<endl;
        return -1;
    }else
    {
        string savefile = dirToSaveImage + ".jpg";
        cvSaveImage(savefile.c_str(),frame);
    }
    cvDestroyWindow("BASELINE");
    return 0;
*/
    IplImage* gray = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
    cvCvtColor(frame,gray,CV_BGR2GRAY);

    stringstream descFileName;
    descFileName<<dirToSaveFeat;

    stringstream keyFileName ;
    keyFileName<<dirToSaveFeat<<"_siftKeypoint";

    int num = getDSIFTdesc_writeToFile(gray,descFileName.str(),keyFileName.str());

    cvReleaseImage(&frame);
    cvReleaseImage(&gray);

    return num;

}
/*
void doextract_write_youtube_baseline(vector<string> categories)
{
    for(int i=0; i<categories.size(); i++)
    {
        cout<<"Inside category "<<categories[i]<<endl;

        string imagelocation =  getBaselineLocation(categories[i]);

        string listname = imagelocation + "middleframes";

        ifstream middleframes(listname.c_str(),ios::in);
        cout<<"The file location of middle frames is "<<listname<<endl;
        while(middleframes.good())
        {
            cout<<"Inside middle frames  "<<endl;
            string imagename;
            middleframes>>imagename;
            if(imagename == "" || imagename == " ")
                continue;
            string imgfullname = imagelocation + imagename;

            IplImage* frame = cvLoadImage(imgfullname.c_str());
            if(frame == 0)
            {
                cout<<"Cannot open image "<<imagename<<endl;
                continue;
            }
            cout<<"Read Image "<<imagename<<endl;

            IplImage* gray = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
            cvCvtColor(frame,gray,CV_BGR2GRAY);

            string img = getActualName(imagename);

            string descFileName = imagelocation +  img + "_siftDesc";
            string keyFileName = imagelocation + img + "_siftKeypoint";
            getDSIFTdesc_writeToFile(gray,descFileName,keyFileName);
            cvReleaseImage(&frame);
            cvReleaseImage(&gray);
        }
        middleframes.close();
    }
}

*/
