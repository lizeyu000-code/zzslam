#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "common_include.h"
#include "frontend.h"
#include "backend.h"
#include "viewer.h"
#include "camera.h"

namespace myslam {

/**
 * VO 对外数据
 */

struct Multi_Sensor_Data{
    cv::Mat left_img;
    cv::Mat right_img;
    // Eigen::Vector4d GPS_Pose;
    SE3 Wheel_Odom_data;
    Vec9 Imu_Data;
};

/**
 * VO 对外接口
 */
class VisualOdometry {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * vo port
     */
    SE3 TrackStereo(Multi_Sensor_Data data_input_);

    /**
     * Parameter Set
     */
    bool ParameterSet();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

   private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    // Dataset::Ptr dataset_ = nullptr;
    std::vector<Camera::Ptr> cameras_;
};
}  // namespace myslam

#endif  // MYSLAM_VISUAL_ODOMETRY_H
