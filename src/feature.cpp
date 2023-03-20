//
// Created by gaoxiang on 19-5-2.
//

#include "feature.h"

namespace myslam {

Feature::Ptr Feature::CreateFeature() {
    static long factory_id = 0;
    Feature::Ptr new_feature(new Feature);
    new_feature->id_ = factory_id++;
    new_feature->track_cnt = 1;
    return new_feature;
}


}