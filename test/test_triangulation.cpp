//
// Created by huws on 23-03-23.
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "frontend.h"

using namespace std;

void GridFastDetector(
    const cv::Mat& image_input, 
    std::vector<cv::Point2f>& kps, 
    const cv::Mat& mask,
    Eigen::Vector2d& GaryThreshold, 
    int blockSize);

int FindFeaturesInRight(
    const cv::Mat& left_image_input, 
    const cv::Mat& right_image_input, 
    std::vector<cv::Point2f>& kps_left,
    std::vector<cv::Point2f>& kps_right
    );

double SSD(const cv::Mat& img0_input, const cv::Mat& img1_input, 
    const cv::Point2f& pt_ref, const cv::Point2f& pt_cur);

int main(int argc, char **argv) 
{
    if (argc != 3) {
        cout << "usage: ../bin/test_triangulation image0_path image1_path" << endl;
        return -1;
    }
    
    cv::Mat left_img, right_img;
    left_img = cv::imread(string(argv[1]), cv::IMREAD_GRAYSCALE);
    right_img = cv::imread(string(argv[1]), cv::IMREAD_GRAYSCALE);

    std::vector<cv::Point2f> fpts;
    Eigen::Vector2d th(25, 10);
    cv::Mat mask;
    GridFastDetector(left_img, fpts, mask, th, 40);



    return 0;
}


int FindFeaturesInRight(
    const cv::Mat& left_image_input, 
    const cv::Mat& right_image_input, 
    std::vector<cv::Point2f>& kps_left,
    std::vector<cv::Point2f>& kps_right
    ) {
    // 光流匹配
    for (auto &kp : kps_left) {
        kps_left.push_back(kp);
        kps_right.push_back(kp);
        // auto mp = kp->map_point_.lock();
        // if (mp) {
        //     // use projected points as initial guess
        //     auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
        //     kps_right.push_back(cv::Point2f(px[0], px[1]));
        // } else {
        //     // use same pixel in left iamge
        //     kps_right.push_back(kp->position_.pt);
        // }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        left_image_input, right_image_input, kps_left,
        kps_right, status, error, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // 是否可以用更简单的方式，直接默认极线是水平的
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            float left_high = kps_left[i].y;
            float right_high = kps_right[i].y;
            if (abs(left_high - right_high)>2) {
                status[i] = 0;
            }
        }
    }
    // todo : 学VINS ，双向光流验证？ 不过要考虑时间效率问题

    // int num_good_pts = 0;
    // for (size_t i = 0; i < status.size(); ++i) {
    //     if (status[i]) {
    //         cv::KeyPoint kp(kps_right[i], 7);
    //         Feature::Ptr feat(new Feature(current_frame_, kp));
    //         feat->is_on_left_image_ = false;
    //         current_frame_->features_right_.push_back(feat);
    //         num_good_pts++;
    //     } else {
    //         current_frame_->features_right_.push_back(nullptr);
    //     }
    // }

    int num_good_pts = 0;
    //show stereo match result
    cv::Mat stereo_match_show;
    cv::cvtColor(left_image_input, stereo_match_show, cv::COLOR_GRAY2BGR);
    for(size_t i = 0; i < kps_left.size(); ++i) {
        if (status[i])
        {
            cv::circle(stereo_match_show, kps_left[i], 3, cv::Scalar(0,0,255), -1);
            cv::line(stereo_match_show, kps_left[i], kps_right[i], cv::Scalar(0,255,0), 2, -1);
            num_good_pts++;
        }
    }
    cv::imshow("stereo_match_show", stereo_match_show);
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";


    // SSD 匹配
    double best_ssd = 999;
    cv::Point2f best_px_cur;
    int dmax = 128;
    std::vector<bool> status1;
    // 遍历所有特征点
    for (int i = 0; i < kps_left.size(); ++i)
    {
        //沿着极线进行SSD搜索
        for (int j = kps_left[i].x - dmax; j < kps_left[i].x + dmax; j++)
        {
            cv::Point2f tmp_pt(kps_left[i].y, j);
            double ssd = SSD(left_image_input, right_image_input, kps_left[i], tmp_pt);
            if (ssd < best_ssd)
            {
                best_ssd = ssd;
                best_px_cur = tmp_pt;
            }
        }
        if (best_ssd < 0.15) {
            status1[i] = true;
        }
        else{
            status1[i] = false;
        }
    }

    
    return num_good_pts;
}

//
double SSD(const cv::Mat& img0_input, const cv::Mat& img1_input, 
    const cv::Point2f& pt_ref, const cv::Point2f& pt_cur) 
{

}


void GridFastDetector(
    const cv::Mat& image_input, 
    std::vector<cv::Point2f>& kps, 
    const cv::Mat& mask,
    Eigen::Vector2d& GaryThreshold, 
    int blockSize)
{
    // TicToc t_1;
    const float W = blockSize;
    const int Border = 3;

    const float width = image_input.cols-6;
    const float height = image_input.rows-6;
    const int nCols = width/W;
    const int nRows = height/W;
    const int wCell = ceil(width/nCols);
    const int hCell = ceil(height/nRows);

    // kps.reserve(nCols * nRows);
    for(int i = 0; i < nRows; i++)
    {
        const float iniY = i * hCell;
        float maxY = iniY + hCell + 1;

        if(iniY >= height - hCell - 2)
            continue;

        for(int j = 0; j < nCols; j++)
        {
            const float iniX = j * wCell;
            float maxX = iniX + wCell + 1;
            if(iniX >= width - wCell - 2)
                continue;
            std::vector<cv::KeyPoint> vKeysCell;
            FAST(image_input.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, GaryThreshold(0), true); // 20
            // cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(GaryThreshold(0), cv::FastFeatureDetector::THRESHOLD, cv::FastFeatureDetector::TYPE_9_16);
            // detector -> detect(current_frame_->left_img_.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, mask);

            if(vKeysCell.empty())
            {
                // cv::Ptr<cv::FastFeatureDetector> detector2 = cv::FastFeatureDetector::create(GaryThreshold(0), cv::FastFeatureDetector::THRESHOLD, cv::FastFeatureDetector::TYPE_9_16);
                // detector2 -> detect(current_frame_->left_img_.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, mask);
                FAST(image_input.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, GaryThreshold(1), true); // 10
            }

            if(!vKeysCell.empty())
            {
                // std::cout << "vKeysCell.size() " << vKeysCell.size() << std::endl;
                float maxResponse = 0.0;
                cv::KeyPoint maxPoint;
                for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                {
                    (*vit).pt.x+=j*wCell;
                    (*vit).pt.y+=i*hCell;
                    if(maxResponse < (*vit).response)
                    {
                        maxResponse = (*vit).response;
                        maxPoint = (*vit);
                    }
                    // kps.emplace_back(*vit);
                }
                if(mask.at<uchar>(maxPoint.pt) > 0)
                    kps.emplace_back(maxPoint.pt);
            }   
        }
    }
    // printf("GridFastDetector cost %f ms.\n",t_1.toc());
    cv::Mat img_show;
    cv::cvtColor(image_input, img_show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < kps.size(); i++) {
        cv::circle(img_show, kps[i], 2, cv::Scalar(0,255,0), -1);
    }
    cv::imshow("tracked ", img_show);
    // cv::waitKey(0);
}


