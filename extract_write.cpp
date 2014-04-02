/**
Step 1:
Extracting all features writing them into file.
This step is to be done seperately for each of the kind of features that you have .

ie. motion features, combination features, baseline features.
each of them may take seperate inputs. but the above class should not worry about that.. it calls the specfic method responsible for
that kind of feature

**/

// this is a generic function responsible to extract and write the features.. this calls specific functions for each of the individual classes.
#include "extract_write.h"
#include "DenseTrack.h"
#include "Descriptors.h"
#include "Initialize.h"
#include "time.h"
void extract_write_features()
{
    cout<<"In extract_write_features"<<endl;
    extract_write_youtubeFeatures();
}

void write_feat_to_file_motion(cv::Mat &feat, int start, int end, string featname, string dirname)
{
    stringstream outfilename ;
    outfilename<<dirname<<"/"<<featname<<"_"<<start<<"_"<<end;
    ofstream out(outfilename.str().c_str(),ios::out);
    if(!out.good())
    {
        cout<<"cannot create file "<<outfilename.str();
        exit(0);
    }
    out<<start<<"\t"<<end<<"\t"<<feat.rows<<"\t"<<feat.cols<<endl;
    for(int i=0; i<feat.rows; i++)
    {
        for(int j=0; j<feat.cols; j++)
        {
            out<<feat.at<float>(i,j)<<"\t";
        }
        out<<endl;
    }
    out.close();
}
void write_feat_to_file_combined1(cv::Mat &feat, int start, int end, string featname, string dirname)
{
    stringstream outfilename ;
    outfilename<<dirname<<"/"<<featname<<"_"<<start<<"_"<<end;
    ofstream out(outfilename.str().c_str(),ios::out);
    if(!out.good())
    {
        cout<<"cannot create file "<<outfilename.str();
        exit(0);
    }
    out<<start<<"\t"<<end<<"\t"<<feat.rows<<"\t"<<feat.cols<<endl;
    for(int i=0; i<feat.rows; i++)
    {
        for(int j=0; j<feat.cols; j++)
        {
            out<<feat.at<float>(i,j)<<"\t";
        }
        out<<endl;
    }
    out.close();

}

int doTracking(int start_frame_abs, int end_frame_abs,char* video, cv::Mat &hogFeat, cv::Mat &hofFeat, cv::Mat &mbhXYFeat,int keyframenum)
{

    int middleframe = (start_frame_abs + end_frame_abs ) / 2 ;

    float* fscales = 0; // float scale values
    IplImageWrapper image, prev_image, grey, prev_grey;
    IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;
    string videoname = video;
    string videoactualname = videoname.substr(0,videoname.find_last_of("."));
    int frameNum = 0;
    TrackerInfo tracker;
    DescInfo hogInfo;
    DescInfo hofInfo;
    DescInfo mbhInfo;

    CvCapture* capture = 0;

    InitTrackerInfo(&tracker, track_length, init_gap);
    InitDescInfo(&hogInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&hofInfo, 9, 1, 1, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&mbhInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);

    capture = cvCreateFileCapture(video);

    if( !capture )
    {
        printf( "Could not initialize capturing..\n" );
        return -1;
    }

    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, double(start_frame_abs));

    double nextframe = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES);


    int start_frame = 0;
    int end_frame = end_frame_abs - start_frame_abs;

    std::vector<std::list<Track> > xyScaleTracks;
    int init_counter = 0; // indicate when to detect new feature points
    while( true )
    {
        IplImage* frame = 0;
        int i, j, c;

        // get a new frame
        if(frameNum > end_frame)
        {
            cout<<"Going to release the capture"<<endl;
            cvReleaseCapture(&capture);
            return 0;
        }

        frame = cvQueryFrame( capture );

        if(start_frame_abs+frameNum == keyframenum)
        {
            stringstream keyimagename;
            keyimagename<<"/media/F85F-3841/leopard-keyframes-testneg/keyimage_"<<keyframenum-FRAME_OFFSET<<".jpg";
            cvSaveImage(keyimagename.str().c_str(),frame);
            cout<<"The key frame number is "<<start_frame_abs+frameNum<<endl;
        }

        if(start_frame_abs + frameNum == middleframe)
        {
            stringstream middleframename;
            middleframename<<"/media/F85F-3841/leopard-baseline-testneg/middleframe_"<<middleframe-FRAME_OFFSET<<".jpg";
            cvSaveImage(middleframename.str().c_str(),frame);
            cout<<"The middle frame number is "<<start_frame_abs+frameNum<<endl;
        }

        if( !frame )
        {
            cout<<"Cannot read frame! "<<frameNum<<" Going to exit"<<endl;
            break;
        }
        if( frameNum >= start_frame && frameNum <= end_frame )
        {
            if( !image )
            {
                // initailize all the buffers
                image = IplImageWrapper( cvGetSize(frame), 8, 3 );
                image->origin = frame->origin;
                prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
                prev_image->origin = frame->origin;
                grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, scale_stride );

                cvCopy( frame, image, 0 );
                cvCvtColor( image, grey, CV_BGR2GRAY );
                grey_pyramid.rebuild( grey );

                // how many scale we can have
                scale_num = std::min<std::size_t>(scale_num, grey_pyramid.numOfLevels());

                fscales = (float*)cvAlloc(scale_num*sizeof(float));
                xyScaleTracks.resize(scale_num);

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    fscales[ixyScale] = pow(scale_stride, ixyScale);

                    // find good features at each scale separately
                    IplImage *grey_temp = 0, *eig_temp = 0;
                    std::size_t temp_level = (std::size_t)ixyScale;
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                    eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
                    std::vector<CvPoint2D32f> points(0);
                    cvDenseSample(grey_temp, eig_temp, points, quality, min_distance);

                    // save the feature points
                    for( i = 0; i < points.size(); i++ )
                    {
                        Track track(tracker.trackLength);
                        PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
                        track.addPointDesc(point);
                        tracks.push_back(track);
                    }
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &eig_temp );
                }
            }

            // build the image pyramid for the current frame
            cvCopy( frame, image, 0 );
            cvCvtColor( image, grey, CV_BGR2GRAY );
            grey_pyramid.rebuild(grey);

            if( frameNum > 0 )
            {
                init_counter++;
                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    // track feature points in each scale separately
                    std::vector<CvPoint2D32f> points_in(0);
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack)
                    {
                        CvPoint2D32f point = iTrack->pointDescs.back().point;
                        points_in.push_back(point); // collect all the feature points
                    }
                    int count = points_in.size();
                    IplImage *prev_grey_temp = 0, *grey_temp = 0;
                    std::size_t temp_level = ixyScale;
                    prev_grey_temp = cvCloneImage(prev_grey_pyramid.getImage(temp_level));
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));

                    cv::Mat prev_grey_mat = cv::cvarrToMat(prev_grey_temp);
                    cv::Mat grey_mat = cv::cvarrToMat(grey_temp);

                    std::vector<int> status(count);
                    std::vector<CvPoint2D32f> points_out(count);

                    // compute the optical flow

                    IplImage* flow = cvCreateImage(cvGetSize(grey_temp), IPL_DEPTH_32F, 2);
                    cv::Mat flow_mat = cv::cvarrToMat(flow);
                    cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,
                                                  sqrt(2.0f)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );	// get the flow as a matrix using previous and current
                    // track feature points by median filtering
                    OpticalFlowTracker(flow, points_in, points_out, status);

                    int width = grey_temp->width;
                    int height = grey_temp->height;
                    // compute the integral histograms
                    DescMat* hogMat = InitDescMat(height, width, hogInfo.nBins);
                    HogComp(prev_grey_temp, hogMat, hogInfo);

                    DescMat* hofMat = InitDescMat(height, width, hofInfo.nBins);
                    HofComp(flow, hofMat, hofInfo);

                    DescMat* mbhMatX = InitDescMat(height, width, mbhInfo.nBins);
                    DescMat* mbhMatY = InitDescMat(height, width, mbhInfo.nBins);
                    MbhComp(flow, mbhMatX, mbhMatY, mbhInfo);

                    i = 0;
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i)
                    {
                        if( status[i] == 1 )   // if the feature point is successfully tracked
                        {
                            PointDesc& pointDesc = iTrack->pointDescs.back();

                            CvPoint2D32f prev_point = points_in[i];
                            // get the descriptors for the feature point

                            CvScalar rect = getRect(prev_point, cvSize(width, height), hogInfo);
                            pointDesc.hog = getDesc(hogMat, rect, hogInfo);
                            pointDesc.hof = getDesc(hofMat, rect, hofInfo);
                            pointDesc.mbhX = getDesc(mbhMatX, rect, mbhInfo);
                            pointDesc.mbhY = getDesc(mbhMatY, rect, mbhInfo);

                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            iTrack->addPointDesc(point);
                            ++iTrack;
                        }
                        else // remove the track, if we lose feature point
                            iTrack = tracks.erase(iTrack);
                    }
                    ReleDescMat(hogMat);
                    ReleDescMat(hofMat);
                    ReleDescMat(mbhMatX);
                    ReleDescMat(mbhMatY);
                    cvReleaseImage( &prev_grey_temp );
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &flow );
                }

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale]; // output the features for each scale
                    for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); )
                    {
                        if( iTrack->pointDescs.size() >= tracker.trackLength+1 )   // if the trajectory achieves the length we want
                        {
                            //		    cout<<"***trajectory to be outputted***"<<endl;
                            std::vector<CvPoint2D32f> trajectory(tracker.trackLength+1);
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            std::list<PointDesc>::iterator iDesc = descs.begin();

                            for (int count = 0; count <= tracker.trackLength; ++iDesc, ++count)
                            {
                                trajectory[count].x = iDesc->point.x*fscales[ixyScale];
                                trajectory[count].y = iDesc->point.y*fscales[ixyScale];
                            }
                            float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

                            if( isValid(trajectory, mean_x, mean_y, var_x, var_y, length) == 1 )
                            {
                                iDesc = descs.begin();
                                int t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);
                                for( int n = 0; n < hogInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hogInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hogInfo.dim; m++ )
                                            vec[m] += iDesc->hog[m];
                                    cv::Mat hogtemp = cvCreateMat(1,hogInfo.dim,CV_32FC1);
                                    for( int m = 0; m < hogInfo.dim; m++ )
                                    {
                                        hogtemp.at<float>(0,m) = vec[m]/float(t_stride);
                                    }
                                    hogFeat.push_back(hogtemp);
                                }

                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
                                for( int n = 0; n < hofInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hofInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hofInfo.dim; m++ )
                                            vec[m] += iDesc->hof[m];
                                    cv::Mat hoftemp = cvCreateMat(1,hofInfo.dim,CV_32FC1);
                                    for( int m = 0; m < hofInfo.dim; m++ )
                                    {
                                        hoftemp.at<float>(0,m) = vec[m]/float(t_stride);
                                    }
                                    hofFeat.push_back(hoftemp);
                                }
                                // combined mbhX and mbhY to mbhXY -- nend to test it !
                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
                                for( int n = 0; n < mbhInfo.ntCells; n++ )
                                {
                                    std::vector<float> vecX(mbhInfo.dim);
                                    std::vector<float> vecY(mbhInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < mbhInfo.dim; m++ )
                                        {
                                            vecX[m] += iDesc->mbhX[m];
                                            vecY[m] += iDesc->mbhY[m];
                                        }
                                    cv::Mat mbhXYtemp = cvCreateMat(1,2*mbhInfo.dim,CV_32FC1);
                                    for(int m = 0; m < mbhInfo.dim; m++ )
                                    {
                                        mbhXYtemp.at<float>(0,m) = vecX[m]/float(t_stride);
                                    }
                                    for(int m = mbhInfo.dim; m < 2*mbhInfo.dim; m++ )
                                    {
                                        mbhXYtemp.at<float>(0,m) = vecY[m-mbhInfo.dim]/float(t_stride);
                                    }
                                    mbhXYFeat.push_back(mbhXYtemp);
                                }
                            }
                            iTrack = tracks.erase(iTrack);
                        }
                        else
                            iTrack++;
                    }
                }

                if( init_counter == tracker.initGap )   // detect new feature points every initGap frames
                {
                    init_counter = 0;
                    for (int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale)
                    {

                        std::list<Track>& tracks = xyScaleTracks[ixyScale];
                        std::vector<CvPoint2D32f> points_in(0);
                        std::vector<CvPoint2D32f> points_out(0);
                        for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++)
                        {
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            CvPoint2D32f point = descs.back().point; // the last point in the track
                            points_in.push_back(point);
                        }

                        IplImage *grey_temp = 0, *eig_temp = 0;
                        std::size_t temp_level = (std::size_t)ixyScale;
                        grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                        eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));

                        cvDenseSample(grey_temp, eig_temp, points_in, points_out, quality, min_distance);
                        // save the new feature points
                        for( i = 0; i < points_out.size(); i++)
                        {
                            Track track(tracker.trackLength);
                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            track.addPointDesc(point);
                            tracks.push_back(track);
                        }
                        cvReleaseImage( &grey_temp );
                        cvReleaseImage( &eig_temp );
                    }
                }

            }

            cvCopy( frame, prev_image, 0 );
            cvCvtColor( prev_image, prev_grey, CV_BGR2GRAY );
            prev_grey_pyramid.rebuild(prev_grey);
        }
        // get the next frame
        frameNum++;
    }
    return 0;
}
void readImagefileList(string imagefilelist,string dirToGetShot, vector<string> &imagelist)
{
    ifstream file(imagefilelist.c_str(),ios::in);
    if(!file.good())
    {
        cerr<<"Cannot open file "<<imagefilelist<<endl;
        return;
    }
    string name;
    string prev_name;
    int num = 0;
    while(file.good())
    {
        file>>name;
        if(name == "" || name == " ")
            continue;
        imagelist.push_back(dirToGetShot + name);
    }
    file.close();
    if(imagelist.size() == 0 || imagelist.size() == 1 || imagelist.size() == 2)
    {
        return ;
    }
    if(imagelist[imagelist.size() - 1] == imagelist[imagelist.size() - 2])
    {
        imagelist.erase(imagelist.end());
    }

    return;
}

// first saves it in a matrix and then writes it to the file
int doTracking_youtubeData(string imagefilelist, string dirToGetShot, string dirToSaveFeat)
{
    vector<string> imagelist;
    readImagefileList(imagefilelist,dirToGetShot,imagelist);
    if(imagelist.size() == 0)
        return -1;
    if(imagelist.size() <=  track_length)
        return 0;
    cout<<"Number of frames in this shot is "<<imagelist.size()<<endl;

    cv::Mat hogFeat = cvCreateMat(0,featureDimension_hog,CV_32FC1);
    cv::Mat hofFeat = cvCreateMat(0,featureDimension_hof,CV_32FC1);
    cv::Mat mbhXYFeat = cvCreateMat(0,featureDimension_mbhXY,CV_32FC1);
    int middleframe = imagelist.size() / 2;

    float* fscales = 0; // float scale values
    IplImageWrapper image, prev_image, grey, prev_grey;
    IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;


    int frameNum = 0;
    TrackerInfo tracker;
    DescInfo hogInfo;
    DescInfo hofInfo;
   DescInfo mbhInfo;

    CvCapture* capture = 0;

    InitTrackerInfo(&tracker, track_length, init_gap);
    InitDescInfo(&hogInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);

   InitDescInfo(&hofInfo, 9, 1, 1, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&mbhInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);

    int start_frame = 0;
    int end_frame = imagelist.size() - 1;

    std::vector<std::list<Track> > xyScaleTracks;
    int init_counter = 0; // indicate when to detect new feature points
    clock_t begin = clock();
    float opticaltime = 0;
    while( true )
    {
        IplImage* frame = 0;
        int i, j, c;

        // get a new frame
        if(frameNum >= imagelist.size())
        {
            cout<<"Completed all frames"<<endl;
            break;
        }

//       frame = cvQueryFrame( capture );
//       cout<<"Reading image "<<imagelist[frameNum]<<endl;
        frame = cvLoadImage(imagelist[frameNum].c_str());

        if( !frame )
        {
            cout<<"Cannot read frame! "<<frameNum<<" Going to exit"<<endl;
            break;
        }
        if( frameNum >= start_frame && frameNum <= end_frame )
        {
            if( !image )
            {
                // initailize all the buffers
                image = IplImageWrapper( cvGetSize(frame), 8, 3 );
                image->origin = frame->origin;
                prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
                prev_image->origin = frame->origin;
                grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, scale_stride );

                cvCopy( frame, image, 0 );
                cvCvtColor( image, grey, CV_BGR2GRAY );
                grey_pyramid.rebuild( grey );

                // how many scale we can have
                scale_num = std::min<std::size_t>(scale_num, grey_pyramid.numOfLevels());

                fscales = (float*)cvAlloc(scale_num*sizeof(float));
                xyScaleTracks.resize(scale_num);

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    fscales[ixyScale] = pow(scale_stride, ixyScale);

                    // find good features at each scale separately
                    IplImage *grey_temp = 0, *eig_temp = 0;
                    std::size_t temp_level = (std::size_t)ixyScale;
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                    eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
                    std::vector<CvPoint2D32f> points(0);
                    cvDenseSample(grey_temp, eig_temp, points, quality, min_distance);

                    // save the feature points
                    for( i = 0; i < points.size(); i++ )
                    {
                        Track track(tracker.trackLength);
                        PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
                        track.addPointDesc(point);
                        tracks.push_back(track);
                    }
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &eig_temp );
                }
            }

            // build the image pyramid for the current frame
            cvCopy( frame, image, 0 );
            cvCvtColor( image, grey, CV_BGR2GRAY );
            grey_pyramid.rebuild(grey);

            if( frameNum > 0 )
            {
                init_counter++;
                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    // track feature points in each scale separately
                    std::vector<CvPoint2D32f> points_in(0);
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack)
                    {
                        CvPoint2D32f point = iTrack->pointDescs.back().point;
                        points_in.push_back(point); // collect all the feature points
                    }
                    int count = points_in.size();
                    IplImage *prev_grey_temp = 0, *grey_temp = 0;
                    std::size_t temp_level = ixyScale;
                    prev_grey_temp = cvCloneImage(prev_grey_pyramid.getImage(temp_level));
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));

                    cv::Mat prev_grey_mat = cv::cvarrToMat(prev_grey_temp);
                    cv::Mat grey_mat = cv::cvarrToMat(grey_temp);

                    std::vector<int> status(count);
                    std::vector<CvPoint2D32f> points_out(count);

                    // compute the optical flow

                    IplImage* flow = cvCreateImage(cvGetSize(grey_temp), IPL_DEPTH_32F, 2);
                    cv::Mat flow_mat = cv::cvarrToMat(flow);
                    clock_t startoptical = clock();
                    cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,
                                                  sqrt(2.0f)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );	// get the flow as a matrix using previous and current
                    // track feature points by median filtering
                    OpticalFlowTracker(flow, points_in, points_out, status);
                    opticaltime = opticaltime + float(clock() - startoptical) / CLOCKS_PER_SEC;

                    int width = grey_temp->width;
                    int height = grey_temp->height;
                    // compute the integral histograms
                    DescMat* hogMat = InitDescMat(height, width, hogInfo.nBins);
                    HogComp(prev_grey_temp, hogMat, hogInfo);

                    DescMat* hofMat = InitDescMat(height, width, hofInfo.nBins);
                    HofComp(flow, hofMat, hofInfo);

                    DescMat* mbhMatX = InitDescMat(height, width, mbhInfo.nBins);
                    DescMat* mbhMatY = InitDescMat(height, width, mbhInfo.nBins);
                    MbhComp(flow, mbhMatX, mbhMatY, mbhInfo);

                    i = 0;
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i)
                    {
                        if( status[i] == 1 )   // if the feature point is successfully tracked
                        {
                            PointDesc& pointDesc = iTrack->pointDescs.back();

                            CvPoint2D32f prev_point = points_in[i];
                            // get the descriptors for the feature point

                            CvScalar rect = getRect(prev_point, cvSize(width, height), hogInfo);
                            pointDesc.hog = getDesc(hogMat, rect, hogInfo);
                            pointDesc.hof = getDesc(hofMat, rect, hofInfo);
                            pointDesc.mbhX = getDesc(mbhMatX, rect, mbhInfo);
                            pointDesc.mbhY = getDesc(mbhMatY, rect, mbhInfo);

                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            iTrack->addPointDesc(point);
                            ++iTrack;
                        }
                        else // remove the track, if we lose feature point
                            iTrack = tracks.erase(iTrack);
                    }
                    ReleDescMat(hogMat);
                    ReleDescMat(hofMat);
                    ReleDescMat(mbhMatX);
                    ReleDescMat(mbhMatY);
                    cvReleaseImage( &prev_grey_temp );
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &flow );
                }

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale]; // output the features for each scale
                    for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); )
                    {
                        if( iTrack->pointDescs.size() >= tracker.trackLength+1 )   // if the trajectory achieves the length we want
                        {
                            //		    cout<<"***trajectory to be outputted***"<<endl;
                            std::vector<CvPoint2D32f> trajectory(tracker.trackLength+1);
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            std::list<PointDesc>::iterator iDesc = descs.begin();

                            for (int count = 0; count <= tracker.trackLength; ++iDesc, ++count)
                            {
                                trajectory[count].x = iDesc->point.x*fscales[ixyScale];
                                trajectory[count].y = iDesc->point.y*fscales[ixyScale];
                            }
                            float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

                            if( isValid(trajectory, mean_x, mean_y, var_x, var_y, length) == 1 )
                            {
                                iDesc = descs.begin();
                                int t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);
                                for( int n = 0; n < hogInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hogInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hogInfo.dim; m++ )
                                            vec[m] += iDesc->hog[m];
                                    cv::Mat hogtemp = cvCreateMat(1,hogInfo.dim,CV_32FC1);
                                    for( int m = 0; m < hogInfo.dim; m++ )
                                    {
                                        hogtemp.at<float>(0,m) = vec[m]/float(t_stride);
                                    }
                                    hogFeat.push_back(hogtemp);
                                    hogtemp.release();
                                }

                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
                                for( int n = 0; n < hofInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hofInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hofInfo.dim; m++ )
                                            vec[m] += iDesc->hof[m];
                                    cv::Mat hoftemp = cvCreateMat(1,hofInfo.dim,CV_32FC1);
                                    for( int m = 0; m < hofInfo.dim; m++ )
                                    {
                                        hoftemp.at<float>(0,m) = vec[m]/float(t_stride);
                                    }
                                    hofFeat.push_back(hoftemp);
                                    hoftemp.release();
                                }
                                // combined mbhX and mbhY to mbhXY -- nend to test it !
                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
                                for( int n = 0; n < mbhInfo.ntCells; n++ )
                                {
                                    std::vector<float> vecX(mbhInfo.dim);
                                    std::vector<float> vecY(mbhInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < mbhInfo.dim; m++ )
                                        {
                                            vecX[m] += iDesc->mbhX[m];
                                            vecY[m] += iDesc->mbhY[m];
                                        }
                                    cv::Mat mbhXYtemp = cvCreateMat(1,2*mbhInfo.dim,CV_32FC1);
                                    for(int m = 0; m < mbhInfo.dim; m++ )
                                    {
                                        mbhXYtemp.at<float>(0,m) = vecX[m]/float(t_stride);
                                    }
                                    for(int m = mbhInfo.dim; m < 2*mbhInfo.dim; m++ )
                                    {
                                        mbhXYtemp.at<float>(0,m) = vecY[m-mbhInfo.dim]/float(t_stride);
                                    }
                                    mbhXYFeat.push_back(mbhXYtemp);
                                    mbhXYtemp.release();
                                }
                            }
                            iTrack = tracks.erase(iTrack);
                        }
                        else
                            iTrack++;
                    }
                }

                if( init_counter == tracker.initGap )   // detect new feature points every initGap frames
                {
                    init_counter = 0;
                    for (int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale)
                    {

                        std::list<Track>& tracks = xyScaleTracks[ixyScale];
                        std::vector<CvPoint2D32f> points_in(0);
                        std::vector<CvPoint2D32f> points_out(0);
                        for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++)
                        {
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            CvPoint2D32f point = descs.back().point; // the last point in the track
                            points_in.push_back(point);
                        }

                        IplImage *grey_temp = 0, *eig_temp = 0;
                        std::size_t temp_level = (std::size_t)ixyScale;
                        grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                        eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));

                        cvDenseSample(grey_temp, eig_temp, points_in, points_out, quality, min_distance);
                        // save the new feature points
                        for( i = 0; i < points_out.size(); i++)
                        {
                            Track track(tracker.trackLength);
                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            track.addPointDesc(point);
                            tracks.push_back(track);
                        }
                        cvReleaseImage( &grey_temp );
                        cvReleaseImage( &eig_temp );
                    }
                }
            }

            cvCopy( frame, prev_image, 0 );
            cvCvtColor( prev_image, prev_grey, CV_BGR2GRAY );
            prev_grey_pyramid.rebuild(prev_grey);
        }
        // get the next frame
        cvReleaseImage(&frame);
        frameNum++;
    }

    ~image;
    ~prev_image;
    ~grey;
    ~prev_grey;
    ~grey_pyramid;
    ~prev_grey_pyramid;
    ~eig_pyramid;
    float textract = float(clock() - begin) / CLOCKS_PER_SEC;

    cout<<"Total time for feature extraction2 "<<textract<<endl;
    cout<<"Total time for optical flow2 "<<opticaltime<<endl;
    cout<<"Ratio "<<opticaltime/textract *100.0<<endl;
    cout<<"Total frame numbers "<<imagelist.size()<<endl;
    cout<<"Total number of rows "<<hogFeat.rows<<endl;
    return 0;
    // write the features into file..
    cout<<"Going to write features into the file "<<endl;
    stringstream hogfeatfile;
    stringstream hoffeatfile;
    stringstream mbhfeatfile;

    hogfeatfile<<dirToSaveFeat<<"_"<<"hog";
    hoffeatfile<<dirToSaveFeat<<"_"<<"hof";
    mbhfeatfile<<dirToSaveFeat<<"_"<<"mbh";

    ofstream hogout(hogfeatfile.str().c_str(),ios::out);
    ofstream hofout(hoffeatfile.str().c_str(),ios::out);
    ofstream mbhout(mbhfeatfile.str().c_str(),ios::out);
    if(!hogout.good() || !hofout.good() || !mbhout.good())
    {
        cerr<<"cannot create file to write feature";
        exit(0);
    }

    hogout<<hogFeat.rows<<"\t"<<hogFeat.cols<<endl;
    hofout<<hofFeat.rows<<"\t"<<hofFeat.cols<<endl;
    mbhout<<mbhXYFeat.rows<<"\t"<<mbhXYFeat.cols<<endl;

    for(int i=0; i<hogFeat.rows; i++)
    {
        for(int j=0; j<hogFeat.cols; j++)
        {
            hogout<<hogFeat.at<float>(i,j)<<"\t";
        }
        hogout<<endl;

        for(int j=0; j<hofFeat.cols; j++)
        {
            hofout<<hofFeat.at<float>(i,j)<<"\t";
        }
        hofout<<endl;

        for(int j=0; j<mbhXYFeat.cols; j++)
        {
            mbhout<<mbhXYFeat.at<float>(i,j)<<"\t";
        }
        mbhout<<endl;
    }

    hogout.close();
    hofout.close();
    mbhout.close();
    int rows = hogFeat.rows;
    hogFeat.release();
    hofFeat.release();
    mbhXYFeat.release();

    return rows;
}

// writes directly to the file
int doTracking_youtubeData_2(string imagefilelist, string dirToGetShot, string dirToSaveFeat)
{
    int rows = 0;
    vector<string> imagelist;
    readImagefileList(imagefilelist,dirToGetShot,imagelist);
    if(imagelist.size() == 0)
        return -1;
    // cout<<"Number of frames in this shot is "<<imagelist.size()<<endl;

    stringstream hogfeatfile ;
    stringstream hoffeatfile;
    stringstream mbhfeatfile;

    hogfeatfile<<dirToSaveFeat<<"_"<<"hog";
    hoffeatfile<<dirToSaveFeat<<"_"<<"hof";
    mbhfeatfile<<dirToSaveFeat<<"_"<<"mbh";

    ofstream hogout(hogfeatfile.str().c_str(),ios::out);
    ofstream hofout(hoffeatfile.str().c_str(),ios::out);
    ofstream mbhout(mbhfeatfile.str().c_str(),ios::out);
    if(!hogout.good() || !hofout.good() || !mbhout.good())
    {
        cerr<<"cannot create file to write feature";
        exit(0);
    }

//   int middleframe = imagelist.size() / 2;

    float* fscales = 0; // float scale values
    IplImageWrapper image, prev_image, grey, prev_grey;
    IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;


    int frameNum = 0;
    TrackerInfo tracker;
    DescInfo hogInfo;
    DescInfo hofInfo;
    DescInfo mbhInfo;

    CvCapture* capture = 0;

    InitTrackerInfo(&tracker, track_length, init_gap);
    InitDescInfo(&hogInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&hofInfo, 9, 1, 1, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&mbhInfo, 8, 0, 1, patch_size, nxy_cell, nt_cell);

    int start_frame = 0;
    int end_frame = imagelist.size() - 1;

    std::vector<std::list<Track> > xyScaleTracks;
    int init_counter = 0; // indicate when to detect new feature points
    while( true )
    {
        IplImage* aframe = 0;
        IplImage* frame = 0;
        int i, j, c;

        // get a new frame
        if(frameNum >= imagelist.size())
        {
            cout<<"Completed all frames"<<endl;
            break;
        }

//       frame = cvQueryFrame( capture );
//       cout<<"Reading image "<<imagelist[frameNum]<<endl;

        aframe = cvLoadImage(imagelist[frameNum].c_str());
        if( !aframe )
        {
            cout<<"Cannot read frame! "<<imagelist[frameNum]<<" Going to exit"<<endl;
            break;
        }

        frame = cvCreateImage(cvSize(640,360),aframe->depth,aframe->nChannels);
        cvResize(aframe,frame,CV_INTER_CUBIC);


        if( frameNum >= start_frame && frameNum <= end_frame )
        {
            if( !image )
            {
                // initailize all the buffers
                image = IplImageWrapper( cvGetSize(frame), 8, 3 );
                image->origin = frame->origin;
                prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
                prev_image->origin = frame->origin;
                grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
                prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, scale_stride );
                eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, scale_stride );

                cvCopy( frame, image, 0 );
                cvCvtColor( image, grey, CV_BGR2GRAY );
                grey_pyramid.rebuild( grey );

                // how many scale we can have
                scale_num = std::min<std::size_t>(scale_num, grey_pyramid.numOfLevels());

                fscales = (float*)cvAlloc(scale_num*sizeof(float));
                xyScaleTracks.resize(scale_num);

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    fscales[ixyScale] = pow(scale_stride, ixyScale);

                    // find good features at each scale separately
                    IplImage *grey_temp = 0, *eig_temp = 0;
                    std::size_t temp_level = (std::size_t)ixyScale;
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                    eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));
                    std::vector<CvPoint2D32f> points(0);
                    cvDenseSample(grey_temp, eig_temp, points, quality, min_distance);

                    // save the feature points
                    for( i = 0; i < points.size(); i++ )
                    {
                        Track track(tracker.trackLength);
                        PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
                        track.addPointDesc(point);
                        tracks.push_back(track);
                    }
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &eig_temp );
                }
            }

            // build the image pyramid for the current frame
            cvCopy( frame, image, 0 );
            cvCvtColor( image, grey, CV_BGR2GRAY );
            grey_pyramid.rebuild(grey);

            if( frameNum > 0 )
            {
                init_counter++;
                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    // track feature points in each scale separately
                    std::vector<CvPoint2D32f> points_in(0);
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack)
                    {
                        CvPoint2D32f point = iTrack->pointDescs.back().point;
                        points_in.push_back(point); // collect all the feature points
                    }
                    int count = points_in.size();
                    IplImage *prev_grey_temp = 0, *grey_temp = 0;
                    std::size_t temp_level = ixyScale;
                    prev_grey_temp = cvCloneImage(prev_grey_pyramid.getImage(temp_level));
                    grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));

                    cv::Mat prev_grey_mat = cv::cvarrToMat(prev_grey_temp);
                    cv::Mat grey_mat = cv::cvarrToMat(grey_temp);

                    std::vector<int> status(count);
                    std::vector<CvPoint2D32f> points_out(count);

                    // compute the optical flow

                    IplImage* flow = cvCreateImage(cvGetSize(grey_temp), IPL_DEPTH_32F, 2);
                    cv::Mat flow_mat = cv::cvarrToMat(flow);
                    cv::calcOpticalFlowFarneback( prev_grey_mat, grey_mat, flow_mat,
                                                  sqrt(2.0f)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );	// get the flow as a matrix using previous and current
                    // track feature points by median filtering
                    OpticalFlowTracker(flow, points_in, points_out, status);

                    int width = grey_temp->width;
                    int height = grey_temp->height;
                    // compute the integral histograms
                    DescMat* hogMat = InitDescMat(height, width, hogInfo.nBins);
                    HogComp(prev_grey_temp, hogMat, hogInfo);

                    DescMat* hofMat = InitDescMat(height, width, hofInfo.nBins);
                    HofComp(flow, hofMat, hofInfo);

                    DescMat* mbhMatX = InitDescMat(height, width, mbhInfo.nBins);
                    DescMat* mbhMatY = InitDescMat(height, width, mbhInfo.nBins);
                    MbhComp(flow, mbhMatX, mbhMatY, mbhInfo);

                    i = 0;
                    for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i)
                    {
                        if( status[i] == 1 )   // if the feature point is successfully tracked
                        {
                            PointDesc& pointDesc = iTrack->pointDescs.back();

                            CvPoint2D32f prev_point = points_in[i];
                            // get the descriptors for the feature point

                            CvScalar rect = getRect(prev_point, cvSize(width, height), hogInfo);
                            pointDesc.hog = getDesc(hogMat, rect, hogInfo);
                            pointDesc.hof = getDesc(hofMat, rect, hofInfo);
                            pointDesc.mbhX = getDesc(mbhMatX, rect, mbhInfo);
                            pointDesc.mbhY = getDesc(mbhMatY, rect, mbhInfo);

                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            iTrack->addPointDesc(point);
                            ++iTrack;
                        }
                        else // remove the track, if we lose feature point
                            iTrack = tracks.erase(iTrack);
                    }
                    ReleDescMat(hogMat);
                    ReleDescMat(hofMat);
                    ReleDescMat(mbhMatX);
                    ReleDescMat(mbhMatY);
                    cvReleaseImage( &prev_grey_temp );
                    cvReleaseImage( &grey_temp );
                    cvReleaseImage( &flow );
                }

                for( int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale )
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale]; // output the features for each scale
                    for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); )
                    {
                        if( iTrack->pointDescs.size() >= tracker.trackLength+1 )   // if the trajectory achieves the length we want
                        {
                            //		    cout<<"***trajectory to be outputted***"<<endl;
                            std::vector<CvPoint2D32f> trajectory(tracker.trackLength+1);
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            std::list<PointDesc>::iterator iDesc = descs.begin();

                            for (int count = 0; count <= tracker.trackLength; ++iDesc, ++count)
                            {
                                trajectory[count].x = iDesc->point.x*fscales[ixyScale];
                                trajectory[count].y = iDesc->point.y*fscales[ixyScale];
                            }
                            float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

                            if( isValid(trajectory, mean_x, mean_y, var_x, var_y, length) == 1 )
                            {
                                iDesc = descs.begin();
                                int t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);
                                for( int n = 0; n < hogInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hogInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hogInfo.dim; m++ )
                                            vec[m] += iDesc->hog[m];
                                    for( int m = 0; m < hogInfo.dim; m++ )
                                    {
                                        hogout<<vec[m]/float(t_stride)<<"\t";
                                    }
                                    hogout<<endl;
                                    rows++;
                                }

                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
                                for( int n = 0; n < hofInfo.ntCells; n++ )
                                {
                                    std::vector<float> vec(hofInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < hofInfo.dim; m++ )
                                            vec[m] += iDesc->hof[m];
                                    for( int m = 0; m < hofInfo.dim; m++ )
                                    {
                                        hofout<<vec[m]/float(t_stride)<<"\t";
                                    }
                                    hofout<<endl;
                                }
                                // combined mbhX and mbhY to mbhXY -- nend to test it !
                                iDesc = descs.begin();
                                t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
                                for( int n = 0; n < mbhInfo.ntCells; n++ )
                                {
                                    std::vector<float> vecX(mbhInfo.dim);
                                    std::vector<float> vecY(mbhInfo.dim);
                                    for( int t = 0; t < t_stride; t++, iDesc++ )
                                        for( int m = 0; m < mbhInfo.dim; m++ )
                                        {
                                            vecX[m] += iDesc->mbhX[m];
                                            vecY[m] += iDesc->mbhY[m];
                                        }
                                    for(int m = 0; m < mbhInfo.dim; m++ )
                                    {
                                        mbhout<<vecX[m]/float(t_stride)<<"\t";
                                    }
                                    for(int m = mbhInfo.dim; m < 2*mbhInfo.dim; m++ )
                                    {
                                        mbhout<<vecY[m-mbhInfo.dim]/float(t_stride)<<"\t";
                                    }
                                    mbhout<<endl;
                                }
                            }
                            iTrack = tracks.erase(iTrack);
                        }
                        else
                            iTrack++;
                    }
                }

                if( init_counter == tracker.initGap )   // detect new feature points every initGap frames
                {
                    init_counter = 0;
                    for (int ixyScale = start_scale_num; ixyScale < end_scale_num; ++ixyScale)
                    {

                        std::list<Track>& tracks = xyScaleTracks[ixyScale];
                        std::vector<CvPoint2D32f> points_in(0);
                        std::vector<CvPoint2D32f> points_out(0);
                        for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++)
                        {
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            CvPoint2D32f point = descs.back().point; // the last point in the track
                            points_in.push_back(point);
                        }

                        IplImage *grey_temp = 0, *eig_temp = 0;
                        std::size_t temp_level = (std::size_t)ixyScale;
                        grey_temp = cvCloneImage(grey_pyramid.getImage(temp_level));
                        eig_temp = cvCloneImage(eig_pyramid.getImage(temp_level));

                        cvDenseSample(grey_temp, eig_temp, points_in, points_out, quality, min_distance);
                        // save the new feature points
                        for( i = 0; i < points_out.size(); i++)
                        {
                            Track track(tracker.trackLength);
                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            track.addPointDesc(point);
                            tracks.push_back(track);
                        }
                        cvReleaseImage( &grey_temp );
                        cvReleaseImage( &eig_temp );
                    }
                }
            }

            cvCopy( frame, prev_image, 0 );
            cvCvtColor( prev_image, prev_grey, CV_BGR2GRAY );
            prev_grey_pyramid.rebuild(prev_grey);
        }
        // get the next frame
        cvReleaseImage(&frame);
        frameNum++;
    }

    ~image;
    ~prev_image;
    ~grey;
    ~prev_grey;
    ~grey_pyramid;
    ~prev_grey_pyramid;
    ~eig_pyramid;

    // write the features into file..
    hogout.close();
    hofout.close();
    mbhout.close();

    return rows;
}

