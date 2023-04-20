//
// Created by huws on 23-03-20.
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "algorithm.h"
#include "backend.h"
#include "feature.h"
#include "frontend.h"
#include "g2o_types.h"
#include "map.h"
#include "viewer.h"

#define DEBUG 1
namespace myslam {

Frontend::Frontend() {
    // gftt_ = cv::GFTTDetector::create(num_features_, 0.01, 20);
    // num_features_init_ = Config::Get<int>("num_features_init");
    // num_features_ = Config::Get<int>("num_features");
}

bool Frontend::inBorder(const cv::Point2f &pt)
{
    int col = current_frame_->left_img_.cols;
    int row = current_frame_->left_img_.rows;
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

SE3 Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    SE3 tcw_ = current_frame_->Pose();

    last_frame_ = current_frame_;
    return tcw_;
}

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    num_match_mpts = 0;
    tracking_inliers_ = 0;

    int num_track_last = TrackLastFrame();
    if (num_match_mpts > 15) {
        tracking_inliers_ = EstimateCurrentPose();
    }
    
    //
    LOG(INFO) << "tracking_inliers_ is : " << tracking_inliers_;
    //
    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        status_ = FrontendStatus::TRACKING_BAD;
        // lost
        // status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();
    // detect new features
    DetectFeatures();  
    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    // backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    // std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            // std::vector<Vec3> points{
            //     camera_left_->pixel2camera(
            //         Vec2(current_frame_->features_left_[i]->position_.pt.x,
            //              current_frame_->features_left_[i]->position_.pt.y)),
            //     camera_right_->pixel2camera(
            //         Vec2(current_frame_->features_right_[i]->position_.pt.x,
            //              current_frame_->features_right_[i]->position_.pt.y))};
            float disparity = current_frame_->features_left_[i]->position_.pt.x -
                    current_frame_->features_right_[i]->position_.pt.x;
            if (disparity < 2) {
                continue;
            }
            double depth = camera_left_->fx_ * camera_left_->baseline_ / disparity;
            // Vec3 pworld = Vec3::Zero();
            Vec3 pworld = camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                        current_frame_->features_left_[i]->position_.pt.y),
                        depth);

            // if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            if(pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;
        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };
            if (iteration == 2) {
                e->setRobustKernel(nullptr); //为什么第二次迭代要去掉鲁棒核函数？
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            // feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

// use LK flow to estimate points in the last image
int Frontend::TrackLastFrame() {
    // Step 0 准备匹配点
    LOG(INFO) << "Now tracking " << current_frame_->id_ <<" frame." 
        << " And now stamp is" << current_frame_->time_stamp_;
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        // if (kp->map_point_.lock()) {
        //     // use project point
        //     auto mp = kp->map_point_.lock();
        //     auto px =
        //         camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
        //     kps_last.push_back(kp->position_.pt);
        //     kps_current.push_back(cv::Point2f(px[0], px[1]));
        // } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        // }
    }

    // Step 1 计算光流
    // TicToc t_1;
    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last, kps_current, 
        status, error, cv::Size(15, 15), 8,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // LOG(INFO) << "tracker calcOpticalFlowPyrLK cost : " << t_1.toc() << " ms.";
    
    // Step 2 匹配点极限约束剔除outlier ,  opencv接口计算本质矩阵，某种意义也是一种对级约束的outlier剔除
    std::vector<uchar> status_2;
    cv::findFundamentalMat(kps_last, kps_current, cv::FM_RANSAC, 1.0, 0.99, status_2);
    for (size_t i = 0; i < status_2.size(); i++) {
        if (status[i] && !status_2[i]) {
            status[i] = 0;
        }
    }

    // step 3 
    int num_good_pts = 0;
    // int num_match_mpts = 0;
    cv::Mat kp_im_show = current_frame_->left_img_.clone();
    cv::cvtColor(kp_im_show, kp_im_show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && inBorder(kps_current[i])) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->id_ = last_frame_->features_left_[i]->id_;  //把追踪到的上一帧特征点ID给到当前帧
            feature->track_cnt = last_frame_->features_left_[i]->track_cnt + 1;  //追踪次数+1
            if ( last_frame_->features_left_[i]->map_point_.lock()) {
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;  //匹配上了就直接把上一帧的地图点拿过来
                // num_match_mpts++;
            }
            current_frame_->features_left_.push_back(feature);
            // num_good_pts++;
        }
    }

    // step 4 mask剔除点,优先保留观测次数多的点
    cv::Mat mask_for_redu_feat(current_frame_->left_img_.size(), CV_8UC1, cv::Scalar(255));
    // 根据追踪次数给特征点重新排个序 , 利用光流特点，追踪多的稳定性好，排前面
    std::vector< std::pair<int, Feature::Ptr>> cnt_pts_id;
    for (int i = 0; i < current_frame_->features_left_.size(); i++) {
        cnt_pts_id.push_back(std::make_pair(current_frame_->features_left_[i]->track_cnt, current_frame_->features_left_[i]));
    }
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
    [](const std::pair<int, Feature::Ptr> &a, const std::pair<int, Feature::Ptr> &b){
        return a.first > b.first;
    });
    //清除原来的feat
    current_frame_->features_left_.clear();
    //重新塞
    for (auto &it : cnt_pts_id) {
        if (mask_for_redu_feat.at<uchar>(it.second->position_.pt) == 255) {
            // opencv函数，把周围一个圆内全部置0,这个区域不允许别的特征点存在，避免特征点过于集中
            cv::circle(mask_for_redu_feat, it.second->position_.pt, MIN_DIS, 0, -1);
            num_good_pts++;
            current_frame_->features_left_.push_back(it.second);
            if ( it.second->map_point_.lock()) {
                num_match_mpts++;
            }
        }
    }

    #if DEBUG
    // draw optical tracker result
    std::vector<cv::Point2f> Last_kps_pt, Current_kps_pt;
    std::vector<int> Last_frame_matched_index;
    std::vector<int> Cur_frame_matched_index;
    int cnt_match = 0;
    cnt_match = FindMatch(last_frame_, current_frame_, 
        Last_kps_pt, Current_kps_pt,
        Last_frame_matched_index, Cur_frame_matched_index);
    for (size_t i = 0; i < Current_kps_pt.size(); i++)
    {
        double len = std::min(1.0, 1.0 * current_frame_->features_left_[Cur_frame_matched_index[i]]->track_cnt / 7);
        cv::circle(kp_im_show, Current_kps_pt.at(i), 3, cv::Scalar(255 * (1 - len), 0, 255 * len), cv::FILLED); //BGR
        cv::line(kp_im_show, Last_kps_pt.at(i), Current_kps_pt.at(i), cv::Scalar(0, 255, 0), 1);
    }
    LOG(INFO) << "CalcOpticalFlowPyrLK has tracked " << num_good_pts << " last frame feature point." 
            << " And matchd " << num_match_mpts << " Map Point.";
    cv::imshow("kp_im_show",kp_im_show);
    #endif
    
    // step 5 detect new feature point
    if(num_good_pts < 150) {
        DetectFeatures();
    }

    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        LOG(INFO) << "StereoInit Sccuess!";
        return true;
    }
    LOG(INFO) << "StereoInit failed!";
    return false;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::circle(mask, feat->position_.pt, MIN_DIS, 0, cv::FILLED);
    }

    std::vector<cv::Point2f> keypoints;
    Eigen::Vector2d th(20, 10);
    GridFastDetector(current_frame_->left_img_, keypoints, mask, th, 40);
    // cv::goodFeaturesToTrack(current_frame_->left_img_, keypoints, 200 - current_frame_->features_left_.size(), 0.01, MIN_DIS, mask);
    //
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        cv::KeyPoint kp_tmp(kp, 7);
        Feature::Ptr feature_new = Feature::CreateFeature();
        feature_new->frame_ = current_frame_;
        feature_new->position_ = kp_tmp;
        current_frame_->features_left_.push_back(feature_new);
        cnt_detected++;
    }
    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

void Frontend::GridFastDetector(
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

            if(vKeysCell.empty()) {
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
    // cv::Mat img_show;
    // cv::cvtColor(current_frame_->left_img_, img_show, cv::COLOR_GRAY2BGR);
    // for (size_t i = 0; i < kps.size(); i++) {
    //     cv::circle(img_show, kps[i], 2, cv::Scalar(0,255,0), -1);
    // }
    // cv::imshow("tracked ", img_show);
    // cv::waitKey(0);
}

// use LK flow to estimate points in the right image
int Frontend::FindFeaturesInRight() {
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        kps_right.push_back(kp->position_.pt);
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
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(15, 15), 8,
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

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }

    #if DEBUG
    // show stereo match result
    cv::Mat stereo_match_show = current_frame_->right_img_.clone();
    cv::cvtColor(stereo_match_show, stereo_match_show, cv::COLOR_GRAY2BGR);
    for(size_t i = 0; i < current_frame_->features_right_.size(); ++i) {
        if (current_frame_->features_right_[i] != nullptr)
        {
            cv::circle(stereo_match_show,
                current_frame_->features_right_[i]->position_.pt,
                3, cv::Scalar(0,0,255), -1);
            cv::line(stereo_match_show, 
            current_frame_->features_right_[i]->position_.pt,
            current_frame_->features_left_[i]->position_.pt,
            cv::Scalar(0,255,0),2, -1);
        }
    }
    cv::imshow("stereo_match_show", stereo_match_show);
    #endif
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    // std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;

        // create map point from triangulation
        // std::vector<Vec3> points{
        //     camera_left_->pixel2camera(
        //         Vec2(current_frame_->features_left_[i]->position_.pt.x,
        //              current_frame_->features_left_[i]->position_.pt.y)),
        //     camera_right_->pixel2camera(
        //         Vec2(current_frame_->features_right_[i]->position_.pt.x,
        //              current_frame_->features_right_[i]->position_.pt.y))};

        float disparity = current_frame_->features_left_[i]->position_.pt.x -
                            current_frame_->features_right_[i]->position_.pt.x;

        if (disparity < 2) {
            continue;
        }
        
        double depth = camera_left_->fx_ * camera_left_->baseline_ / disparity;
        // Vec3 pworld = Vec3::Zero();
        Vec3 pworld = camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y),
                     depth);

        // if (triangulation(poses, points, pworld) && pworld[2] > 0) {
        if (pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    LOG(INFO) << "Initial map created with " << cnt_init_landmarks << " map points";
    // 
    if (cnt_init_landmarks > 80) {
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        // backend_->UpdateMap();
        // LOG(INFO) << "Initial map created with " << cnt_init_landmarks << " map points";
        LOG(INFO) << "Initial map success!";
    }
    else {
        map_->RetMap();
        LOG(INFO) << "Initial map failed!";
        return false;
    }

    return true;
}

//功能：找任意两帧之间的匹配点（或者说共视点），咳咳，注意特征点多的放img1
//参数：（上一帧，当前帧，上一帧特征点位置，当前帧特征点位置，上一帧特征点下标, 是否是正常匹配模式）如果为false,则是找新地图点
//返回：匹配点个数
int Frontend::FindMatch(const Frame::Ptr &image_last, const Frame::Ptr &image_cur,
    std::vector<cv::Point2f> &imgLast_feat_pt, std::vector<cv::Point2f> &imgCur_feat_pt, 
    std::vector<int> &imgLast_matched_index, std::vector<int> &imgCur_matched_index,
    bool is_find_mappoint)
{
    int match_cnt = 0;
    for (int i = 0; i < image_cur->features_left_.size(); i++){
        if ( image_cur->features_left_[i]->map_point_.lock() && is_find_mappoint) { 
            continue; // 当前的特征点已经关联地图点
        }
        else
        {
            for (int j = 0; j < image_last->features_left_.size(); j++) {
                if (image_cur->features_left_[i]->id_ == image_last->features_left_[j]->id_) {
                    imgLast_feat_pt.emplace_back(image_last->features_left_[j]->position_.pt);
                    imgCur_feat_pt.emplace_back(image_cur->features_left_[i]->position_.pt);
                    imgLast_matched_index.emplace_back(j);
                    imgCur_matched_index.emplace_back(i);
                    match_cnt++;
                    break;
                }
            }
        }
    }
    return match_cnt;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam