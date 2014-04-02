#include "extract_write.h"
using namespace std;

void extract_write_baseline()
{
    // only writing train pos here
    vector<string> trainposfiles;
    string tigerbaselineloc = "/media/FreeAgent\ Drive/thesis/TIGER_OUTPUT/baseline_features_new/tiger-baseline-testneg";
    getImagenamelist(test_neg_shotfilename,tigerbaselineloc, NUM_NEG_TEST,trainposfiles);

  //  getImagenamelist("leopard-train-30-pos","leopard-baseline-trainpos", 461,trainposfiles);

    //getImagenamelist("leopard-test-30-pos","/media/F85F-3841/leopard-baseline-testpos", 358,trainposfiles);

    //getImagenamelist("leopard-test-30-neg","/media/F85F-3841/leopard-baseline-testneg", 301,trainposfiles);

    int imgnum = 0;
    for(; imgnum < trainposfiles.size(); imgnum++ )
    {
        cout<<"Processing image "<<imgnum<<endl;
        readImage_getDesc(trainposfiles[imgnum],tigerbaselineloc);
    }
}

void readImage_getDesc(string fileName, string dirname)
{
    IplImage *img;
    string name = getActualName(fileName);
    cout<<"Actual Name is "<<name<<endl;

    img = cvLoadImage(fileName.c_str());
    if(img == 0)
    {
        cout<<"Cannot open image "<<fileName<<endl;
        return;
    }
    cout<<"Read Image "<<fileName<<endl;
    IplImage* gray = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
    cvCvtColor(img,gray,CV_BGR2GRAY);

     stringstream descFileName ;
    descFileName<<dirname<<"/"<<name<<"_descriptor";

    stringstream keyFileName ;
    keyFileName<<dirname<<"/"<<name<<"_keypoints";


    getDSIFTdesc_writeToFile(gray,descFileName.str(),keyFileName.str());
    cvReleaseImage(&img);
    cvReleaseImage(&gray);
}

int getDSIFTdesc_writeToFile(IplImage* img, string descFileName, string keyFileName)
{
    int img_width = img->width;
    int img_height = img->height;
    int steps = img->widthStep/sizeof(uchar);

    float* frame = (float*)malloc(img_height*img_width*sizeof(float));

    uchar* Ldata    = (uchar *)img->imageData;

    for(int i=0; i<img_height; i++)
    {
        for(int j=0; j<img_width; j++)
        {
            frame[j+steps*i] = ((uchar *)(img->imageData + i*img->widthStep))[j] / 255.0f;
        }
    }
    //cout<<"Create a filter"<<endl;
    VlDsiftFilter* filter_image = vl_dsift_new(img_width,img_height);
    setFilterParams(filter_image);

   // cout<<"Start the dsift process"<<endl;
    vl_dsift_process(filter_image,frame);


    int num_keypoints = vl_dsift_get_keypoint_num(filter_image);
   // cout<<"The number of keypoints are "<<num_keypoints<<endl;

    int desc_size = vl_dsift_get_descriptor_size(filter_image);
  //  cout<<"The desc size is "<<desc_size<<endl;

    const VlDsiftKeypoint *keypoints = vl_dsift_get_keypoints(filter_image);

    const float * desc = vl_dsift_get_descriptors(filter_image);
    cout<<"Completed !"<<endl;
    return 0;   //

    ofstream des(descFileName.c_str(),ios::out);
    ofstream keyfile(keyFileName.c_str(),ios::out);

    for(int i=0; i<num_keypoints; i++)
    {
        keyfile<<keypoints[i].x<<"\t"<<keypoints[i].y<<"\t"<<keypoints[i].s<<"\t"<<keypoints[i].norm<<endl;
        for(int j=0; j<desc_size; j++)
        {
            des<<desc[j+ i*desc_size]<<"\t";
        }
        des<<endl;
    }
    vl_dsift_delete(filter_image);
    des.close();
    keyfile.close();
    return num_keypoints;
}

void setFilterParams(VlDsiftFilter* filter_image)
{
    int stepX = 4;
    int stepY = 4;
    cout<<"Set the steps"<<endl;
    vl_dsift_set_steps (filter_image,stepX,stepY);
    cout<<"create a geonetry object"<<endl;

    VlDsiftDescriptorGeometry* filter_geo = new VlDsiftDescriptorGeometry;

    filter_geo->binSizeX = 10; // 5 -> 7.5 starting , 15 -> 30 starting 10 -> 15 starting
    filter_geo->binSizeY = 10;
    filter_geo->numBinT = 8;
    filter_geo->numBinX = 4;    // this influences the size of the keypoint ! 4 is fine
    filter_geo->numBinY = 4;// this influences the size of the keypoint ! 4 is fine
    cout<<"Set the geometry "<<endl;
    vl_dsift_set_geometry(filter_image,filter_geo);
}

