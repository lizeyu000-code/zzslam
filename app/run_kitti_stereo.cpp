//
// Created by huws on 23-03-19.
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "vo_system.h"

using namespace std;

// DEFINE_string(config_file, "./config/default.yaml", "config file path");
std::string config_file_path = "../config/default.yaml";

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv) 
{
    if (argc != 2) {
        cout << "usage: ../bin/run_kitti_stereo data_path" << endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../logs/");
    google::SetStderrLogging(google::INFO);  
    FLAGS_logbufsecs = 0;  

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[1]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // vo init here
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry(config_file_path));
    bool init_success = vo->Init();
    if(!init_success) {
        LOG(FATAL) << "VO init failed!,please check.";
        return -1;
    }
    // vector<float> vTimesTrack;
    // vTimesTrack.resize(nImages);

    cv::Mat imLeft, imRight;
    //main loop
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty()) 
        {
            LOG(ERROR) << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return -1;
        }
        // cv::imshow("curren frame", imLeft);
        
        myslam::Multi_Sensor_Data data_input;
        data_input.time_stamp_ = tframe;
        data_input.left_img = imLeft;
        data_input.right_img = imRight;

        SE3 tcw_now = vo->TrackStereo(data_input);

        int k = cv::waitKey(1);
        if (k==27)
            break;
        
    }

    LOG(INFO) << "the dataset is run out.";
    // myslam::VisualOdometry::Ptr vo(
    //     new myslam::VisualOdometry(FLAGS_config_file));
    // assert(vo->Init() == true);
    // vo->Run();

    return 0;
}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

