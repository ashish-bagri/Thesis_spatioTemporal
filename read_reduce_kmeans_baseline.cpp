#include "read_reduce_kmeans.h"

void read_readuce_kmeans_baseline()
{
    vector<string> trainposfiles;
    getImagenamelist(train_pos_shotfilename,"trainpos", NUM_POS_TRAIN,trainposfiles);
    cout<<"Total number of examples is "<<trainposfiles.size()<<endl;

    cv::Mat allfeatures = cvCreateMat(0,featureDimension_baseline,CV_32FC1);

    int imgnum = 0;
    for(; imgnum < NUM_POS_TRAIN; imgnum++ )
    {
        cout<<"inside image # "<<imgnum<<endl;
        read_reduce_baseline_desc(trainposfiles[imgnum],allfeatures,5000,"trainpos");
    }

    cout<<"Total number of features used for kmeans are "<<allfeatures.rows<<endl;
    cout<<"Read all the training features. run k means to create a vocabulary"<<endl;

    cout<<"Start k means "<<endl;
    cv::Mat dictionary = cvCreateMat(dictionarySize,featureDimension_baseline,CV_32FC1);
    cv::Mat labels = cvCreateMat(allfeatures.rows,1,CV_32FC1);

    double* retkmeans =  doKmeans(allfeatures,dictionary,dictionarySize,labels);
    cout<<"Completed Kmeans. Going to write the dictionary now"<<endl;

// WRITE DICTIONARY
    writeDictionary(dictionary,dictionarySize,retkmeans[0],retkmeans[1],"baseline");
    cout<<"completed writing the dictionary"<<endl;

}
