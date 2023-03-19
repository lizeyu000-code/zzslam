//
// Created by huws on 23-03-19.
//
#include "vo_system.h"
#include <opencv2/core/eigen.hpp>
#include <chrono>

namespace myslam {

static cv::Mat NewCameraMatrix;
static cv::Mat map1, map2;
static const double alpha = 0;

VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    // if (Config::SetParameterFile(config_file_path_) == false) {
    //     return false;
    // }
    // dataset_ =
    //     Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    // CHECK_EQ(dataset_->Init(), true);
    bool success = ParameterSet();
    if(!success){

        return false;
    }

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(cameras_[0]);

    backend_->SetMap(map_);
    backend_->SetCameras(cameras_[0]);

    viewer_->SetMap(map_);

    return true;
}

SE3 VisualOdometry::TrackStereo(Multi_Sensor_Data data_input_)
{
    TicToc t1;
    // check data 
    if (data_input_.left_img.data == nullptr 
        || data_input_.right_img.data == nullptr) {
        std::cerr << "left or right image is empty!!! please check." << std::endl;
        return SE3;
    }

    cv::Mat left_image_undistored, right_image_undistored;
    //去畸变
    if (0) {
        remap(data_input_.left_img, left_image_undistored, map1, map2, cv::INTER_LINEAR);  
        remap(data_input_.right_img, right_image_undistored, map1, map2, cv::INTER_LINEAR); 
    }else{
        left_image_undistored = data_input_.left_img.clone();
        right_image_undistored = data_input_.right_img.clone();
    }

    //图像锐化
    // if(0){
    //     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	//     clahe->apply(left_image_undistored, left_image_undistored);  
    //     clahe->apply(right_image_undistored, right_image_undistored);  
    // }

    // 创建帧
    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = left_image_undistored;
    new_frame->right_img_ = right_image_undistored;
    new_frame->imu_raw_data_ = data_input_.Imu_Data;
    new_frame->Wheel_Odom_ = data_input_.Wheel_Odom_data;

    // track
    SE3 tcw_ = frontend_->AddFrame(new_frame);
    LOG(INFO) << "vo" << t1.toc() << "ms";

    return tcw_;
} 

bool VisualOdometry::ParameterSet()
{
    cv::FileStorage fsSettings(config_file_path_, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    
    Mat33 K;
    K.setIdentity();
    fsSettings["camera.fx"] >> K(0, 0);
    fsSettings["camera.fy"] >> K(1, 1);
    fsSettings["camera.cx"] >> K(0, 2);
    fsSettings["camera.cy"] >> K(1, 2);

    double k1,k2,p1,p2,k3;
    fsSettings["camera.k1"] >> k1;
    fsSettings["camera.k2"] >> k2;
    fsSettings["camera.p1"] >> p1;
    fsSettings["camera.p2"] >> p2;
    fsSettings["camera.k3"] >> k3;

    double base_line;
    fsSettings["camera.baseline"] >> base_line;

    int img_row, img_col;
    fsSettings["image.width"] >> img_col;
    fsSettings["image.height"] >> img_row;

    Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), base_line));
    new_camera->set_D(k1, k2, p1, p2, k3);
    cameras_.push_back(new_camera);

    cv::Mat cv_K, cv_D;
    cv::eigen2cv(cameras_[0]->K() , cv_K);
    cv::eigen2cv(cameras_[0]->D() , cv_D);
    cv::Size imageSize(img_col, img_row);
    NewCameraMatrix = getOptimalNewCameraMatrix(cv_K, cv_D, imageSize, alpha, imageSize, 0);
    initUndistortRectifyMap(cv_K, cv_D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);

    std::cout<< "camera k is: " << cv_K << std::endl;
    std::cout<< "camera d is: " << cv_D << std::endl;

    fsSettings.release();
    printf("dataset init success. \n");
    return true;
} 


// void VisualOdometry::Run() {
//     while (1) {
//         LOG(INFO) << "VO is running";
//         if (Step() == false) {
//             break;
//         }
//     }
//     backend_->Stop();
//     viewer_->Close();
//     LOG(INFO) << "VO exit";
// }

// bool VisualOdometry::Step() {
//     Frame::Ptr new_frame = dataset_->NextFrame();
//     if (new_frame == nullptr) return false;
//     auto t1 = std::chrono::steady_clock::now();
//     bool success = frontend_->AddFrame(new_frame);
//     auto t2 = std::chrono::steady_clock::now();
//     auto time_used =
//         std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
//     LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
//     return success;
// }


}  // namespace myslam
