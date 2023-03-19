#pragma once

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "common_include.h"

namespace myslam {

// Pinhole stereo camera model
class Camera {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
           baseline_ = 0;  // Camera intrinsics
    double k1_ = 0, k2_ = 0, p1_ = 0, p2_ = 0, k3_ = 0;
    SE3 pose_;             // extrinsic, from stereo camera to single camera
    SE3 pose_inv_;         // inverse of extrinsics

    Camera();

    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose_.inverse();
    }

    Camera(double fx, double fy, double cx, double cy, double baseline)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline) {}

    void set_D(double k1, double k2, double p1, double p2, double k3) 
    {
        k1_ = k1;
        k2_ = k2;
        p1_ = p1;
        p2_ = p2;
        k3_ = k3;
    }

    SE3 pose() const { return pose_; }

    // return intrinsic matrix
    Mat33 K() const {
        Mat33 k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }

    Vec5 D() const {
        Vec5 d;
        d << k1_, k2_, p1_, p2_, k3_;
        return d;
    }
    
    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);

    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
};

}  // namespace myslam
#endif  // MYSLAM_CAMERA_H
