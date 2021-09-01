/**
* @program: dl_core
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-29 13:49
**/
//
// Created by 周炳诚 on 2021/3/29.
//

#include <torch/script.h>
#include <iostream>
#include "base_core.h"
#include "util.h"

inline void
dl_model::BaseModel::create(torch::jit::script::Module &_module, const std::string &mode, int index, bool _half) {
    if (mode == "gpu") {
        device = at::Device(at::kCUDA, index);
    } else {
        device = at::kCPU;
    }
    module = _module;
    module.to(device);
    module.eval();
    half = _half;
    if (half) {
        module.to(torch::kHalf);
    }
};

dl_model::BaseModel::BaseModel(const std::string &model_path, const std::string &mode, int index, bool _half) {
    torch::jit::script::Module _module = torch::jit::load(model_path);
    create(_module, mode, index, _half);
};

dl_model::BaseModel::BaseModel(torch::jit::script::Module &_module, const std::string &mode, int index, bool _half) {
    create(_module, mode, index, _half);
};


sRotatedRect::sRotatedRect(const sRotatedRect &rect) {
    angle = rect.angle;
    center = rect.center;
    score = rect.score;
    size = rect.size;
    inner_points = rect.inner_points;
}

sRotatedRect::sRotatedRect(sRotatedRect &rect) {
    angle = rect.angle;
    center = rect.center;
    score = rect.score;
    size = rect.size;
    inner_points = rect.inner_points;
}

// 如果需要支持sRotatedRect(minAreaRect(points))的方式，那么就需要支持下面的方法
sRotatedRect::sRotatedRect(const cv::RotatedRect &rect, float _score, float scale) {
    angle = rect.angle;
    center.x = rect.center.x / scale;
    center.y = rect.center.y / scale;
    score = _score;
    size.width = rect.size.width / scale;
    size.height = rect.size.height / scale;
}

sRotatedRect::sRotatedRect(cv::RotatedRect &rect, float _score, float scale) {
    angle = rect.angle;
    center.x = rect.center.x / scale;
    center.y = rect.center.y / scale;
    score = _score;
    size.width = rect.size.width / scale;
    size.height = rect.size.height / scale;
}

sRotatedRect::sRotatedRect(std::vector<cv::Point> _points, float _score, float scale) {
    cv::RotatedRect rect = minAreaRect(_points);
    angle = rect.angle;
    center.x = rect.center.x / scale;
    center.y = rect.center.y / scale;
    score = _score;
    size.width = rect.size.width / scale;
    size.height = rect.size.height / scale;
    if (scale == 1) {
        inner_points = _points;
    } else {
        std::vector<cv::Point> n_points;
        for (int i = 0; i < _points.size(); ++i) {
            int x = std::round(_points.at(i).x / scale);
            int y = std::round(_points.at(i).y / scale);
            n_points.push_back(cv::Point(x, y));
        }
        inner_points = n_points;
    }
}

sRotatedRect::sRotatedRect(std::vector<cv::Point2f> _points, float _score, float scale) {
    cv::RotatedRect rect = minAreaRect(_points);
    angle = rect.angle;
    center.x = rect.center.x / scale;
    center.y = rect.center.y / scale;
    score = _score;
    size.width = rect.size.width / scale;
    size.height = rect.size.height / scale;
    std::vector<cv::Point> n_points;
    for (int i = 0; i < _points.size(); ++i) {
        int x = std::round(_points.at(i).x / scale);
        int y = std::round(_points.at(i).y / scale);
        n_points.push_back(cv::Point(x, y));
    }
    inner_points = n_points;
}

bool sRotatedRect::is_flat() {
    if (abs(angle) <= 1 || abs(angle + 90) <= 1)
        return true;
    return false;
}

void sRotatedRect::order_points(cv::Point2f *pts) const {
    cv::Point2f rect_points[4];
    this->points(rect_points);
    int rect_offset = 0;
    if (this->size.width > this->size.height) {
        rect_offset = -1;
    }
    pts[0] = rect_points[(rect_offset + 4) % 4];
    pts[1] = rect_points[(rect_offset + 1) % 4];
    pts[2] = rect_points[(rect_offset + 2) % 4];
    pts[3] = rect_points[(rect_offset + 3) % 4];
}

bool sRotatedRect::inside(cv::Point2f & point) {
    cv::Point2f rect_points[4];
    this->order_points(rect_points);
    float min_side = std::min(this->size.height, this->size.width);
    float proj_dist = this->proj_dist_x(point);
    if (proj_dist >= 1.2 * min_side)
        return false;
    cv::Point P(point);
    cv::Point A1(rect_points[0]);
    cv::Point B1(rect_points[1]);
    cv::Point A2(rect_points[3]);
    cv::Point B2(rect_points[2]);
    float distance1 = get_point2line_dis(P, A1, B1);
    float distance2 = get_point2line_dis(P, A2, B2);

    if (distance1 * distance2 <= 0 && abs(distance1) <= min_side && abs(distance2) <= min_side){
        return true;
    }
    return false;
}


float sRotatedRect::dist(const sRotatedRect &other) {
    return dist(other.center);
}

float sRotatedRect::dist(const cv::Point2f & other_center) {
    float distance2 = std::pow(center.x - other_center.x, 2) + std::pow(center.y - other_center.y, 2);
    return std::sqrt(distance2);
}

float sRotatedRect::angle_dist(const sRotatedRect &other) {
    if (size.width > size.height && other.size.width > other.size.height)
        return std::abs(angle - other.angle);
    if (size.width < size.height && other.size.width < other.size.height)
        return std::abs(angle - other.angle);
    return std::abs(angle + 90 - other.angle);
}

float sRotatedRect::proj_dist_x(const sRotatedRect &other) {
    return proj_dist_x(other.center);
}

float sRotatedRect::proj_dist_x(const cv::Point2f & other_center) {
    float dis_angle; // 两个center距离中心的角度,角度都以逆时针的为准
    if (std::abs(center.x - other_center.x) == 0) {
        dis_angle = CV_PI / 2;
    } else {
        dis_angle = std::atan(-(center.y - other_center.y) / (center.x - other_center.x));
    }
    float org_angle = std::abs(angle) / 180 * CV_PI; // 原始短边所在线的角度
    if ((size.width > size.height))
        org_angle += CV_PI / 2;
    float distance = this->dist(other_center);
    return std::abs(distance * std::cos(org_angle - dis_angle));

}

bool is_same_col(const sRotatedRect &rect1, const sRotatedRect &rect2) {
    float rect1_max_side = std::max(rect1.size.height, rect1.size.width);
    float rect2_max_side = std::max(rect2.size.height, rect2.size.width);
    float rect1_min_side = std::min(rect1.size.height, rect1.size.width);
    float rect2_min_side = std::min(rect2.size.height, rect2.size.width);
    bool small_rect_flag = rect1_max_side / rect1_min_side <= 3 && rect2_max_side / rect2_min_side <= 3;

    float org_angle_1 = rect1.angle / 180 * CV_PI; // 原始短边所在线的角度
    float org_angle_2 = rect2.angle / 180 * CV_PI; // 原始短边所在线的角度
    if (rect1.size.width < rect1.size.height) {
        if (std::abs(org_angle_1) == 0) {
            org_angle_1 = 0;
        } else {
            org_angle_1 += CV_PI / 2;
        }
    }
    if (rect2.size.width < rect2.size.height) {
        if (std::abs(org_angle_2) == 0) {
            org_angle_2 = 0;
        } else {
            org_angle_2 += CV_PI / 2;
        }
    }
    float center_distance = std::sqrt(std::pow(rect1.center.x - rect2.center.x, 2) +
                                      std::pow(rect1.center.y - rect2.center.y, 2));
    float center_dis_angle;
    if (std::abs(rect1.center.x - rect2.center.x) == 0) {
        center_dis_angle = CV_PI / 2;
    } else {
        center_dis_angle = std::atan((rect1.center.y - rect2.center.y) / (rect1.center.x - rect2.center.x));
    }
//    float test_proj_dis1 = std::abs(center_distance * std::sin(org_angle_1 + center_dis_angle));
//    float test_proj_dis2 = std::abs(center_distance * std::sin(org_angle_2 + center_dis_angle));
    float center_proj_dis1 = std::abs(center_distance * std::cos(org_angle_1 + center_dis_angle));
    float center_proj_dis2 = std::abs(center_distance * std::cos(org_angle_2 + center_dis_angle));
    bool center_flag = false;
    if (small_rect_flag){
        //如果框较小直接以中心判断，有一个满足即可.
        center_flag = center_proj_dis1 / rect1_max_side <= 0.5 || center_proj_dis2 / rect2_max_side <= 0.5;
    }else{
        center_flag = center_proj_dis1 / rect1_max_side <= 0.5 && center_proj_dis2 / rect2_max_side <= 0.5;
    }
    if (center_flag)
        return true;
    cv::Point2f rect1_points[4];
    cv::Point2f rect2_points[4];
    rect1.order_points(rect1_points);
    rect2.order_points(rect2_points);

    cv::Point2f rect1_left_point((rect1_points[1].x + rect1_points[2].x) / 2,
                                 (rect1_points[1].y + rect1_points[2].y) / 2);
    cv::Point2f rect2_left_point((rect2_points[1].x + rect2_points[2].x) / 2,
                                 (rect2_points[1].y + rect2_points[2].y) / 2);
    cv::Point2f rect1_right_point((rect1_points[0].x + rect1_points[3].x) / 2,
                                  (rect1_points[0].y + rect1_points[3].y) / 2);
    cv::Point2f rect2_right_point((rect2_points[0].x + rect2_points[3].x) / 2,
                                  (rect2_points[0].y + rect2_points[3].y) / 2);

    float left_distance = std::sqrt(std::pow(rect1_left_point.x - rect2_left_point.x, 2) +
                                      std::pow(rect1_left_point.y - rect2_left_point.y, 2));
    float left_dis_angle;
    if (std::abs(rect1_left_point.x - rect2_left_point.x) == 0) {
        left_dis_angle = CV_PI / 2;
    } else {
        left_dis_angle = std::atan((rect1_left_point.y -rect2_left_point.y) / (rect1_left_point.x - rect2_left_point.x));
    }
    float left_proj_dis1 = std::abs(left_distance * std::cos(org_angle_1 + left_dis_angle));
    float left_proj_dis2 = std::abs(left_distance * std::cos(org_angle_2 + left_dis_angle));
    bool left_flag = false;
    if (small_rect_flag){
        //如果框较小直接以中心判断，有一个满足即可.
        left_flag = left_proj_dis1 / rect1_max_side <= 0.5 || left_proj_dis2 / rect2_max_side <= 0.5;
    }else{
        left_flag = left_proj_dis1 / rect1_max_side <= 0.5 && left_proj_dis2 / rect2_max_side <= 0.5;
    }
    if (left_flag)
        return true;
    float right_distance = std::sqrt(std::pow(rect1_right_point.x - rect2_right_point.x, 2) +
                                    std::pow(rect1_right_point.y - rect2_right_point.y, 2));
    float right_dis_angle;
    if (std::abs(rect1_right_point.x - rect2_right_point.x) == 0) {
        right_dis_angle = CV_PI / 2;
    } else {
        right_dis_angle = std::atan((rect1_right_point.y -rect2_right_point.y) /
                (rect1_right_point.x - rect2_right_point.x));
    }
    float right_proj_dis1 = std::abs(right_distance * std::cos(org_angle_1 + right_dis_angle));
    float right_proj_dis2 = std::abs(center_distance * std::cos(org_angle_2 + right_dis_angle));
    bool right_flag = false;
    if (small_rect_flag){
        //如果框较小直接以中心判断，有一个满足即可.
        right_flag = right_proj_dis1 / rect1_max_side <= 0.5 || right_proj_dis2 / rect2_max_side <= 0.5;
    }else{
        left_flag = right_proj_dis1 / rect1_max_side <= 0.5 && right_proj_dis2 / rect2_max_side <= 0.5;
    }
    if (right_flag)
        return right_flag;
    return false;
}

// 不要写成sRotatedRect::operator<<,友元函数不需要照着上面的写法
std::ostream &operator<<(std::ostream &os, const sRotatedRect &s_rect) {
    os << "center:" << s_rect.center << ";";
    os << "size:" << s_rect.size << ";";
    os << "angle:" << s_rect.angle << ";";
    os << "score:" << s_rect.score << ";";
    if (s_rect.inner_points.size() > 0) {
        os << "inner_point_nums:" << s_rect.inner_points.size() << ";";
        os << "size:" << s_rect.size.width * s_rect.size.height << "";
    }
    return os;
}

TextRect::TextRect(const TextRect &rect) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = rect.text;
}

TextRect::TextRect(TextRect &rect) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = rect.text;
}

TextRect::TextRect(const cv::Rect &rect) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = L"";
}

TextRect::TextRect(cv::Rect &rect) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = L"";
}

TextRect::TextRect(const cv::Rect &rect, const std::wstring &_text) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = _text;
}

TextRect::TextRect(cv::Rect &rect, const std::wstring &_text) {
    height = rect.height;
    width = rect.width;
    x = rect.x;
    y = rect.y;
    text = _text;
}

nlohmann::json TextRect::dump() {
    nlohmann::json object = {
            {"x", x},
            {"y", y},
            {"height", height},
            {"width", width},
            {"text", text}
    };
    return object;
}


