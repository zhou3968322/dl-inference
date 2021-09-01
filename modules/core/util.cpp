/**
* @program: dl-inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-26 11:12
**/
//
// Created by 周炳诚 on 2021/3/26.
//

#include <unordered_map>
#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "util.h"

using namespace torch::indexing;



bool rect_sort_by_ratio(cv::Rect &rect1, cv::Rect &rect2) {
    assert(rect1.height > 0 && rect2.height > 0);
    return rect1.width / ((double) rect1.height) < rect2.width / ((double) rect2.height);
}

//template <typename T, typename Compare>
//std::vector<size_t> sort_indexes(const std::vector<T> &v, Compare (*_compare)(T & t1, T & t2)) {
//
//    // initialize original index locations
//    std::vector<size_t> idx(v.size());
//    std::iota(idx.begin(), idx.end(), 0);
//
//    std::stable_sort(idx.begin(), idx.end(),
//                [&v, &_compare](size_t i1, size_t i2) {return _compare(v[i1], v[i2]);});
//
//    return idx;
//}

std::vector<size_t> sort_indexes_by_x(std::vector<sRotatedRect> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return rotate_rect_sort_by_x(v[i1], v[i2]); });

    return idx;
}

/*!
 * 图像中旋转矩形按y排序
 * @param s_rect1
 * @param s_rect2
 * @return
 */
bool rotate_rect_sort_by_y(sRotatedRect &s_rect1, sRotatedRect &s_rect2) {
    return s_rect1.center.y < s_rect2.center.y;
}

bool rotate_rect_sort_by_x(sRotatedRect &s_rect1, sRotatedRect &s_rect2) {
    return s_rect1.center.x < s_rect2.center.x;
}


void get_rect_tensor(std::vector<sRotatedRect> &rot_vec, at::Tensor &box_tensor, at::Tensor &box_prop_tensor) {
    int rot_size = (int) rot_vec.size();
    box_tensor = torch::zeros({rot_size, 4, 2}, torch::kFloat32);
    float *pBoxArray = box_tensor.data_ptr<float>();
    box_prop_tensor = torch::zeros({rot_size, 6}, torch::kFloat32);
    float *pBoxProperty = box_prop_tensor.data_ptr<float>();
    for (int i = 0; i < rot_vec.size(); ++i) {
        cv::Mat boxPoints2f;
        cv::boxPoints(rot_vec.at(i), boxPoints2f);
        pBoxProperty[i * 6] = rot_vec.at(i).angle;
        pBoxProperty[i * 6 + 1] = rot_vec.at(i).size.width;
        pBoxProperty[i * 6 + 2] = rot_vec.at(i).size.height;
        pBoxProperty[i * 6 + 3] = rot_vec.at(i).score;
        pBoxProperty[i * 6 + 4] = rot_vec.at(i).center.x;
        pBoxProperty[i * 6 + 5] = rot_vec.at(i).center.y;

        for (int j = 0; j < 4; ++j) {
            float *data = boxPoints2f.ptr<float>(j);
            pBoxArray[i * 8 + j * 2] = data[0];
            pBoxArray[i * 8 + j * 2 + 1] = data[1];
        }
    }
}

std::pair<at::Tensor, at::Tensor> get_rect_tensor(std::vector<sRotatedRect> &rot_vec) {
    at::Tensor box_tensor;
    at::Tensor box_prop_tensor;
    get_rect_tensor(rot_vec, box_tensor, box_prop_tensor);
    return std::make_pair(box_tensor, box_prop_tensor);
}

void get_rect_tensor(std::vector<sRotatedRect> &rot_vec, at::Tensor &box_tensor) {
    int rot_size = (int) rot_vec.size();
    box_tensor = torch::zeros({rot_size, 4, 2}, torch::kFloat32);
    float *pBoxArray = box_tensor.data_ptr<float>();
    for (int i = 0; i < rot_vec.size(); ++i) {
        cv::Mat boxPoints2f;
        cv::boxPoints(rot_vec.at(i), boxPoints2f);
        for (int j = 0; j < 4; ++j) {
            float *data = boxPoints2f.ptr<float>(j);
            pBoxArray[i * 8 + j * 2] = data[0];
            pBoxArray[i * 8 + j * 2 + 1] = data[1];
        }
    }
}


at::Tensor get_rect_box_tensor(std::vector<sRotatedRect> &rot_vec) {
    at::Tensor box_tensor;
    get_rect_tensor(rot_vec, box_tensor);
    return box_tensor;
}

/*!
 *
 * @param rot_vec 旋转矩形
 * @param img_width 图片宽度
 * @return 图片相对于顺时针的中值角度.
 */
float get_median_angle(const std::vector<sRotatedRect> &rot_vec, int img_width) {
    std::vector<float> hor_angle_vec;
    std::vector<float> ver_angle_vec;
    for (int i = 0; i < rot_vec.size(); ++i) {
        if (std::abs(rot_vec.at(i).angle) < 1 and rot_vec.at(i).size.width > img_width * 0.5)
            return rot_vec.at(i).angle;
        if (rot_vec.at(i).score > 0.91 && std::abs(std::abs(rot_vec.at(i).angle) - 45) > 3) {
            if (rot_vec.at(i).size.width > rot_vec.at(i).size.height) {
                if (std::abs(rot_vec.at(i).angle) < 45)
                    hor_angle_vec.push_back(-std::abs(rot_vec.at(i).angle));
                else
                    ver_angle_vec.push_back(-90 - std::abs(rot_vec.at(i).angle));
            } else {
                if (rot_vec.at(i).size.width > rot_vec.at(i).size.height) {
                    if (std::abs(rot_vec.at(i).angle) < 45)
                        ver_angle_vec.push_back(-std::abs(rot_vec.at(i).angle));
                    else
                        hor_angle_vec.push_back(-90 - std::abs(rot_vec.at(i).angle));
                }
            }
        }
    }
    std::vector<float> angle_vec = hor_angle_vec.size() >= ver_angle_vec.size() ? hor_angle_vec : ver_angle_vec;
    if (angle_vec.size() != 0 and angle_vec.size() % 2 == 0) {
        std::sort(angle_vec.begin(), angle_vec.end());
        angle_vec.pop_back();
    }
    int vec_size = angle_vec.size();
    if (vec_size == 0)
        return 0;
    at::Tensor angle_tensor;
    angle_tensor = at::from_blob(angle_vec.data(), {vec_size}, at::TensorOptions().dtype(at::kFloat));
    float media_angle = angle_tensor.median().item<float>();
    return media_angle;
}

/*!
 *
 * @param rot_vec 旋转矩形
 * @param img_width 图片宽度
 * @return 图片相对于顺时针的平均角度.
 */
float get_mean_angle(const std::vector<sRotatedRect> &rot_vec, int img_width, int min_side, float min_rect_ratio) {
    std::vector<float> hor_angle_vec;
    std::vector<float> ver_angle_vec;

    for (int i = 0; i < rot_vec.size(); ++i) {
        if (std::abs(rot_vec.at(i).size.width >= img_width * 0.5 || rot_vec.at(i).size.height >= img_width * 0.5))
            return rot_vec.at(i).angle;
        if (rot_vec.at(i).size.height <= min_side || rot_vec.at(i).size.width <= min_side)
            continue;
        if (rot_vec.at(i).size.width / rot_vec.at(i).size.height <= min_rect_ratio &&
            rot_vec.at(i).size.height / rot_vec.at(i).size.width <= min_rect_ratio) {
            continue;
        }
        if (rot_vec.at(i).score >= 0.9 && std::abs(std::abs(rot_vec.at(i).angle) - 45) > 5) {
            if (rot_vec.at(i).size.width > rot_vec.at(i).size.height) {
                if (std::abs(rot_vec.at(i).angle) < 45)
                    hor_angle_vec.push_back(-std::abs(rot_vec.at(i).angle));
                else
                    ver_angle_vec.push_back(-90 - std::abs(rot_vec.at(i).angle));
            } else {
                if (rot_vec.at(i).size.width > rot_vec.at(i).size.height) {
                    if (std::abs(rot_vec.at(i).angle) < 45)
                        ver_angle_vec.push_back(-std::abs(rot_vec.at(i).angle));
                    else
                        hor_angle_vec.push_back(-90 - std::abs(rot_vec.at(i).angle));
                }
            }
        }
    }
    std::vector<float> angle_vec = hor_angle_vec.size() >= ver_angle_vec.size() ? hor_angle_vec : ver_angle_vec;
    if (angle_vec.size() != 0 and angle_vec.size() % 2 == 0) {
        std::sort(angle_vec.begin(), angle_vec.end());
        angle_vec.pop_back();
    }
    int vec_size = angle_vec.size();
    if (vec_size == 0)
        return 0;
    at::Tensor angle_tensor;
    angle_tensor = at::from_blob(angle_vec.data(), {vec_size}, at::TensorOptions().dtype(at::kFloat));
    float mean_angle = angle_tensor.mean().item<float>();
    return mean_angle;
}

bool is_horizon_line(const cv::Vec4f & line){
    if (std::abs(line[1]) <= 0.01)
        return true;
    return false;
}

bool is_vertical_line(const cv::Vec4f & line){
    if (std::abs(line[0]) <= 0.01)
        return true;
    return false;
}

bool is_horizontal(const std::vector<sRotatedRect> &rot_vec, int min_side, float min_rect_ratio, float hor_ratio) {
    int angle_0_nums = 0;
    int valid_nums = 0;

    for (int i = 0; i < rot_vec.size(); ++i) {
        if (rot_vec.at(i).size.height <= min_side || rot_vec.at(i).size.width <= min_side)
            continue;
        if (rot_vec.at(i).size.width / rot_vec.at(i).size.height <= min_rect_ratio &&
            rot_vec.at(i).size.height / rot_vec.at(i).size.width <= min_rect_ratio) {
            continue;
        }
        if (rot_vec.at(i).score >= 0.9){
            if (std::abs(std::abs(rot_vec.at(i).angle) - 90) <= 0.2 || std::abs(rot_vec.at(i).angle) <= 0.2) {
            angle_0_nums += 1;
            }
            valid_nums += 1;
        }

    }
    if (valid_nums == 0)
        return false;
    if (angle_0_nums / (float) valid_nums >= hor_ratio){
        return true;
    }
    return false;
}

void rotate_img_box_points(cv::Mat &img, std::vector<sRotatedRect> &rot_vec, float angle) {
    if (std::abs(angle) <= 0.1)
        return;
#ifdef DEBUG
    std::cout << "rows:" << img.rows << " cols:" << img.cols << std::endl;
#endif
//    float rad = CV_PI / 180 * angle;
//    int h_new = (int) (std::ceil(std::abs(img.rows * std::cos(rad)) + std::abs(img.cols * std::sin(rad))));
//    int w_new = (int) (std::ceil(std::abs(img.cols * std::cos(rad)) + std::abs(img.rows * std::sin(rad))));
//    int h_center = h_new / 2;
//    int w_center = w_new / 2;
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat rotate_matrix = cv::getRotationMatrix2D(center, angle, 1);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
    // 这里不能是float，必须是double，否则会出现无法改变值的问题
    rotate_matrix.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
    rotate_matrix.at<double>(1, 2) += bbox.width / 2.0 - img.cols / 2.0;
    cv::warpAffine(img, img, rotate_matrix, bbox.size(), cv::INTER_AREA, cv::BORDER_REPLICATE,
                   cv::Scalar(255, 255, 255));
    std::vector<std::vector<cv::Point>> rect_points_vecs;
    std::vector<float> score_vec;
    for (int i = 0; i < rot_vec.size(); ++i) {
        std::vector<cv::Point> rect_points_vec;
        for (auto point:rot_vec.at(i).inner_points) {
            rect_points_vec.push_back(point);
        }
        rect_points_vecs.push_back(rect_points_vec);
        score_vec.push_back(rot_vec.at(i).score);
    }
    int origin_rot_size = rot_vec.size();
    rot_vec.clear();
    for (int i = 0; i < rect_points_vecs.size(); ++i) {
        cv::transform(rect_points_vecs.at(i), rect_points_vecs.at(i), rotate_matrix);
        sRotatedRect rotated_rect(rect_points_vecs.at(i), score_vec.at(i));
        rot_vec.push_back(rotated_rect);
    }
    assert(origin_rot_size == rot_vec.size());
#ifdef DEBUG
    std::cout << "rows:" << img.rows << " cols:" << img.cols << std::endl;
#endif
}

void rotate_img_box_points(cv::Mat &img, std::vector<sRotatedRect> &rot_vec, int method) {
    int img_width = img.cols;
    float angle;
    if (method == 0) {
        angle = get_median_angle(rot_vec, img_width);
    } else {
        angle = get_mean_angle(rot_vec, img_width);
    }
#ifdef DEBUG
    std::cout << "current img angle is:" << angle << std::endl;
#endif
    rotate_img_box_points(img, rot_vec, angle);
}

float point_distance(cv::Point2f & pointA, cv::Point2f & pointB){
    float distance2 = std::pow(pointA.x - pointB.x, 2) + std::pow(pointA.y - pointB.y, 2);
    return std::sqrt(distance2);
}

float get_point2line_dis(cv::Point &pointP, cv::Point &pointA, cv::Point &pointB) {
    int A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x * pointB.y - pointA.y * pointB.x;

    float distance = 0;
    distance = (float) (A * pointP.x + B * pointP.y + C) / ((float) std::sqrt(A * A + B * B));
    return distance;
}

bool intersection(cv::Point2f & o1, cv::Point2f & p1, cv::Point2f & o2, cv::Point2f & p2,
                  cv::Point2f &r)
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

bool intersection(cv::Vec4f & line1, cv::Vec4f & line2,
                  cv::Point2f &r, int max_x, int max_y){
    cv::Point2f p1;
    cv::Point2f p2;
    p1.x = line1[2];
    p1.y = line1[3];
    p2.x = line2[2];
    p2.y = line2[3];
    if (std::abs(p1.x - p2.x) < /*EPS*/ 1e-8 && std::abs(p1.y - p2.y) < /*EPS*/ 1e-8){
        r = p1;
        return true;
    }
//    double k1 = line1[1] / line1[0]; //slope
//    double k2 = line2[1] / line2[0]; //slope
//    if (abs(k1 - k2) < /*EPS*/ 1e-8){
//        return false;
//    }
//    r.x = (k1 * p1.x - k2 * p2.x -  (p1.y - p2.y)) / (k1 - k2);
//    r.y =  (k2 * p1.y - k1 * p2.y - k1 * k2 * (p1.x - p2.x))/ (k2 - k1);
    float cross = line1[1] * line2[0] - line2[1] * line1[0];
    if (abs(cross) < /*EPS*/ 1e-8){
        return false;
    }
    r.x = (line1[1] * line2[0] * p1.x - line2[1] * line1[0] * p2.x - line1[0] * line2[0] * (p1.y - p2.y)) / cross;
    r.y = (line2[1] * line1[0] * p1.y - line1[1] * line2[0] * p2.y - line1[1] * line2[1] * (p1.x - p2.x)) / (- cross);
    if (max_x == 0 && max_y == 0)
        return true;
    if (r.x <= max_x && r.x >= 0 && r.y >= 0 && r.y <= max_y)
        return true;
    return false;
}

cv::Vec4f get_line_by_point(cv::Point &pointA, cv::Point &pointB) {
    float A = 0, B = 0, D = 0;
//    point0.x = line[2];//point on the line
//    point0.y = line[3];
//    double k = line[1] / line[0]; //slope
    A = pointA.y - pointB.y;
    B = pointA.x - pointB.x;
    D = sqrt(pow(A, 2) + pow(B, 2));
    cv::Vec4f line;
    line[0] = B / D;
    line[1] = A / D;
    line[2] = (pointB.x + pointA.x) / 2.0;
    line[3] = (pointB.y + pointA.y) / 2.0;
    return line;
}

float get_points_x_error(std::vector<cv::Point2f> points, int method){
    int point_nums = points.size();
    float point_x_sum = 0.0;
    for (int i = 0; i < points.size(); ++i) {
        point_x_sum += points.at(i).x;
    }
    float point_x_mean = point_x_sum / point_nums;
    float dis_sum = 0.0;
    for (int i = 0; i < points.size(); ++i) {
        if (method == 1){
            dis_sum += std::pow(points.at(i).x - point_x_mean, 2);
        }else{
            dis_sum += abs(points.at(i).x - point_x_mean);
        }
    }
    if (method == 1){
        return sqrt(dis_sum / point_nums);
    }else{
        return dis_sum / point_nums;
    }
}

void get_min_error_points(std::vector<std::vector<cv::Point2f>>){

}

void rot_vec2rect(const std::vector<sRotatedRect> &rot_vec, std::vector<cv::Rect> &rect_vec) {
    assert(rect_vec.size() == 0);
    for (int i = 0; i < rot_vec.size(); ++i) {
        rect_vec.push_back(rot_vec.at(i).boundingRect());
    }
}

std::vector<cv::Rect> rot_vec2rect(const std::vector<sRotatedRect> &rot_vec) {
    std::vector<cv::Rect> rect_vec;
    rot_vec2rect(rot_vec, rect_vec);
    return rect_vec;
}

void
filter_rotated_rects(std::vector<sRotatedRect> &s_rotated_rect, std::vector<sRotatedRect> &p_rotated_rect, int min_side,
                     float min_rect_ratio) {
    assert(p_rotated_rect.size() == 0);
    for (int i = 0; i < s_rotated_rect.size(); ++i) {
        if (s_rotated_rect.at(i).size.height < min_side or s_rotated_rect.at(i).size.width < min_side)
            continue;
        if (std::abs(std::abs(s_rotated_rect.at(i).angle) - 45) <= 5)
            continue;
//        if (std::abs(std::abs(s_rotated_rect.at(i).angle) - 45) <= 5 ||
//            std::abs(std::abs(s_rotated_rect.at(i).angle)) <= 0.1 ||
//            std::abs(std::abs(s_rotated_rect.at(i).angle) - 90) <= 0.1)
//            continue;
        if (s_rotated_rect.at(i).size.width / s_rotated_rect.at(i).size.height >= min_rect_ratio ||
            s_rotated_rect.at(i).size.height / s_rotated_rect.at(i).size.width >= min_rect_ratio) {
            p_rotated_rect.push_back(s_rotated_rect.at(i));
        }
    }
}

std::vector<sRotatedRect>
filter_rotated_rects(std::vector<sRotatedRect> &s_rotated_rect, int min_side, float min_rect_ratio) {
    std::vector<sRotatedRect> p_rotated_rect;
    filter_rotated_rects(s_rotated_rect, p_rotated_rect, min_side, min_rect_ratio);
    return p_rotated_rect;
}

void get_row_rotated_rects(std::vector<sRotatedRect> &p_rotated_rect, std::vector<sRotatedRect> &row_rotated_rects,
                           int half_nums, bool debug, const std::string &img_path) {
    if (p_rotated_rect.size() <= 1)
        return;
    assert(row_rotated_rects.size() == 0);
    std::sort(p_rotated_rect.begin(), p_rotated_rect.end(), rotate_rect_sort_by_y);
    if (debug) {
        assert(img_path.size() > 0);
        cv::Mat img = cv::imread(img_path);
        debug_img_rect_order(img, p_rotated_rect, true, "./p_rotated_rect.jpeg");
    }
    int N = p_rotated_rect.size();
//    int H[N][N];
// 使用上面的方式在后面会有 非0元素的初始化
    int H[N][N] = {};
    for (int i = 0; i < N; ++i) {
        int bi = std::max(0, i - half_nums);
        int ei = std::min(i + half_nums + 1, N);
        for (int j = 0; j < N; ++j) {
            if (H[i][j] == -1 || H[i][j] == 1) {
                continue;
            }
            if (j < bi || j >= ei) {
                H[i][j] = -1;
                H[j][i] = -1;
                continue;
            }
            if (j == i) {
                H[i][j] = 1;
                continue;
            }

            float proj_d_1 = p_rotated_rect[i].proj_dist_x(p_rotated_rect[j]);
            float half_min_side_1 = std::min(p_rotated_rect[i].size.height, p_rotated_rect[i].size.width) / 2;
            if (proj_d_1 >= half_min_side_1) {
                H[i][j] = -1;
                H[j][i] = -1;
                continue;
            }
            float proj_d_2 = p_rotated_rect[j].proj_dist_x(p_rotated_rect[i]);
            float half_min_side_2 = std::min(p_rotated_rect[j].size.height, p_rotated_rect[j].size.width) / 2;
            if (proj_d_2 >= half_min_side_2) {
                H[i][j] = -1;
                H[j][i] = -1;
                continue;
            }
//            if (debug) {
//                cv::Mat test_img = cv::imread(img_path);
//                std::vector<sRotatedRect> test_rotated_rects = { p_rotated_rect[i], p_rotated_rect[j]};
//                debug_img_rect_order(test_img, test_rotated_rects, true, "./distance_rotated_rect.jpeg");
//                std::cout << "debugging" << std::endl;
//            }
            H[i][j] = 1;
            H[j][i] = 1;
            for (int k = 0; k < N && k != i && k != j; ++k) {
                if (H[j][k] == 1 || H[k][j] == 1) {
                    H[i][k] = 1;
                    H[k][i] = 1;
                } else if (H[i][k] == 1 || H[k][i] == 1) {
                    H[j][k] = 1;
                    H[k][j] = 1;
                }
            }
        }
    }
    int v_searched[N] = {};
    for (int i = 0; i < N; ++i) {
        if (!v_searched[i]) {
            std::vector<int> same_line_rects;
            for (int j = 0; j < N; ++j) {
                if (H[i][j] == 1) {
                    same_line_rects.push_back(j);
                }
            }
            if (same_line_rects.size() <= 3) {
                for (int j = 0; j < N && j != i; ++j) {
                    H[i][j] = -1;
                    H[j][i] = -1;
                }
                v_searched[i] = 1;
            } else {
//                float min_line_h = img.rows;
                float max_line_h = 0;
                std::vector<cv::Point> line_points;
                float sum_score = 0;
                for (auto point_index:same_line_rects) {
                    v_searched[point_index] = 1;
                    float min_side = std::min(p_rotated_rect.at(point_index).size.height,
                                              p_rotated_rect.at(point_index).size.width);
                    max_line_h = std::max(min_side, max_line_h);
                    for (auto point:p_rotated_rect.at(point_index).inner_points) {
                        line_points.push_back(point);
                    }
                    sum_score +=
                            p_rotated_rect.at(point_index).inner_points.size() * p_rotated_rect.at(point_index).score;
                }
                sRotatedRect row_rotated_rect(line_points, sum_score / line_points.size());
                float line_h = std::min(row_rotated_rect.size.height, row_rotated_rect.size.width);
                if (line_h < 1.2 * max_line_h) {
                    row_rotated_rects.push_back(row_rotated_rect);
                }
#ifdef DEBUG
                else {
                    std::cout << "current vertical line invalid:" << row_rotated_rect << std::endl;
                }
#endif
            }
        }
    }
}

std::vector<sRotatedRect> get_row_rotated_rects(std::vector<sRotatedRect> &p_rotated_rect, int half_nums) {
    std::vector<sRotatedRect> row_rotated_rects;
    get_row_rotated_rects(p_rotated_rect, row_rotated_rects, half_nums);
    return row_rotated_rects;
}

void get_col_rotated_rects(std::vector<sRotatedRect> &p_rotated_rect, std::vector<sRotatedRect> &col_rotated_rect,
                           int half_nums) {
    if (p_rotated_rect.size() <= 1)
        return;
    assert(col_rotated_rect.size() == 0);
    int N = p_rotated_rect.size();
    std::sort(p_rotated_rect.begin(), p_rotated_rect.end(), rotate_rect_sort_by_y);
    int C[N][N] = {}; //C 代表 p_rotated_rect中两两是否是同一列
    for (int i = 0; i < N; ++i) {
        int bi = std::max(0, i - half_nums);
        int ei = std::min(i + half_nums + 1, N);
        for (int j = 0; j < N; ++j) {
            if (C[i][j] == -1 || C[i][j] == 1) {
                continue;
            }
            if (j < bi || j >= ei) {
                C[i][j] = -1;
                C[j][i] = -1;
                continue;
            }
            if (j == i) {
                C[i][j] = 1;
                continue;
            }
            bool flag = is_same_col(p_rotated_rect.at(i), p_rotated_rect.at(j));
            if (!flag) {
                C[i][j] = -1;
                C[j][i] = -1;
                continue;
            }
            C[i][j] = 1;
            C[j][i] = 1;
            for (int k = 0; k < N && k != i && k != j; ++k) {
                if (C[j][k] == 1 || C[k][j] == 1) {
                    C[i][k] = 1;
                    C[k][i] = 1;
                } else if (C[i][k] == 1 || C[k][i] == 1) {
                    C[j][k] = 1;
                    C[k][j] = 1;
                }
            }
        }
    }
    int c_searched[N] = {};
    std::vector<std::vector<sRotatedRect>> col_rotated_rects;
    for (int i = 0; i < N; ++i) {
        if (!c_searched[i]) {
            std::vector<int> same_col_rects;
            for (int j = 0; j < N; ++j) {
                if (C[i][j] == 1) {
                    same_col_rects.push_back(j);
                }
            }
            if (same_col_rects.size() <= 4) {
                for (int j = 0; j < N && j != i; ++j) {
                    C[i][j] = -1;
                    C[j][i] = -1;
                }
                c_searched[i] = 1;
            } else {
                float max_line_w = 0;
                std::vector<cv::Point> col_points;
                std::vector<sRotatedRect> tmp_col_rotated_rects;
                for (auto rect_index:same_col_rects) {
                    tmp_col_rotated_rects.push_back(p_rotated_rect.at(rect_index));
                }
                col_rotated_rects.push_back(tmp_col_rotated_rects);
            }
        }
    }
    if (col_rotated_rects.size() == 0){
        return;
    }
    std::sort(col_rotated_rects.begin(), col_rotated_rects.end(),
              [](std::vector<sRotatedRect> &a, std::vector<sRotatedRect> &b) {
                  return a.size() > b.size();
              });
    col_rotated_rect = col_rotated_rects.at(0);
}

std::vector<sRotatedRect> get_col_rotated_rects(std::vector<sRotatedRect> &p_rotated_rect, int half_nums) {
    std::vector<sRotatedRect> col_rotated_rects;
    get_col_rotated_rects(p_rotated_rect, col_rotated_rects, half_nums);
    return col_rotated_rects;
}

/*!
 *
 * @param A AX = B中的X
 * @param B
 * @param s_rect
 * @param i
 * @param equation_nums 每个s_rect产生的方程数
 */
//void rect_to_transform_tensor(at::Tensor & A, at::Tensor & B, sRotatedRect & s_rect, int i, int equation_nums){
//    int offset_i = equation_nums * i;
//    assert(A.size(0) > offset_i && B.size(0) > offset_i &&
//           A.dim() == 2 && B.dim() == 2 && A.size(1) == 6);
//    assert(equation_nums == 3);
////    assert(args_num == 6 || args_num == 9);
//    int width = s_rect.size.width;
//    int height = s_rect.size.height;
//    cv::Point2f vertices[4];
//    s_rect.points(vertices);
//    int start_idx = 0;
//    if (std::abs(vertices[2].x - vertices[1].x) < std::abs(vertices[1].x - vertices[0].x)){
//        start_idx = 1;
//        width = s_rect.size.height;
//        height = s_rect.size.width;
//    }
//    A.index_put_({0 + offset_i, Slice(2,4)},
//                 torch::tensor({vertices[(start_idx + 5) % 4].x - vertices[(start_idx + 4) % 4].x,
//                                vertices[(start_idx + 5) % 4].y - vertices[(start_idx + 4) % 4].y}));
//    A.index_put_({0 + offset_i, Slice(4,6)},
//                 torch::tensor({height * (vertices[(start_idx + 5) % 4].x + vertices[(start_idx + 4) % 4].x) / 2,
//                                height * (vertices[(start_idx + 5) % 4].y + vertices[(start_idx + 4) % 4].y) / 2}));
//    B.index_put_({0 + offset_i}, - height);
//    A.index_put_({1 + offset_i, Slice(0, 2)},
//                 torch::tensor({vertices[(start_idx + 6) % 4].x - vertices[(start_idx + 5) % 4].x,
//                                vertices[(start_idx + 6) % 4].y - vertices[(start_idx + 5) % 4].y}));
//    A.index_put_({1 + offset_i, Slice(4, 6)},
//                 torch::tensor({- width * (vertices[(start_idx + 6) % 4].x + vertices[(start_idx + 5) % 4].x) / 2,
//                                - width * (vertices[(start_idx + 6) % 4].y + vertices[(start_idx + 5) % 4].y) / 2}));
//    B.index_put_({1 + offset_i}, width);
//    A.index_put_({2 + offset_i, Slice(2,4)},
//                 torch::tensor({vertices[(start_idx + 7) % 4].x - vertices[(start_idx + 6) % 4].x,
//                                vertices[(start_idx + 7) % 4].y - vertices[(start_idx + 6) % 4].y}));
//    A.index_put_({2 + offset_i, Slice(4, 6)},
//                 torch::tensor({- height * (vertices[(start_idx + 7) % 4].x + vertices[(start_idx + 6) % 4].x) / 2,
//                                - height * (vertices[(start_idx + 7) % 4].y + vertices[(start_idx + 6) % 4].y) / 2}));
//    B.index_put_({2 + offset_i}, height);
//}


void get_diag_invert(at::Tensor &diag, at::Tensor &diag_invert) {
    assert(diag.dim() == 1 && diag_invert.dim() == 2);
    for (int i = 0; i < diag.size(0); ++i) {
        if (std::abs(diag[i].item<double>()) != 0) {
            diag_invert.index_put_({i, i}, 1 / std::abs(diag[i].item<double>()));
        }
    }
}

at::Tensor get_diag_invert(at::Tensor &diag) {
    assert(diag.dim() == 1);
    int diag_num = diag.size(0);
    at::Tensor diag_invert = torch::zeros({diag_num, diag_num}).type_as(diag);
    get_diag_invert(diag, diag_invert);
    return diag_invert;
}


void solve_linear_system_by_svd(at::Tensor &A, at::Tensor &B, at::Tensor &X) {
    auto res = torch::svd(A);
    at::Tensor U = std::get<0>(res);
    at::Tensor S = std::get<1>(res);
    at::Tensor V = std::get<2>(res);
    at::Tensor S_invert;
    get_diag_invert(S, S_invert);
    X = (V * S_invert) * U.t() * B;
}

at::Tensor solve_linear_system_by_svd(at::Tensor &A, at::Tensor &B) {
    at::Tensor X;
    solve_linear_system_by_svd(A, B, X);
    return X;
}



////
////
//// DEBUG FUNCTIONS
////
////




void debug_img_rect(const cv::Mat &img, std::vector<TextRect> &rect) {
    cv::Mat debug_im = img.clone();
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        line_color = cv::Scalar(0);
    } else {
        line_color = cv::Scalar(255, 0, 0);
    }
    for (int i = 0; i < rect.size(); ++i) {
        point_vec.push_back(cv::Point(rect.at(i).x, rect.at(i).y));
        point_vec.push_back(cv::Point(rect.at(i).x + rect.at(i).width, rect.at(i).y));
        point_vec.push_back(cv::Point(rect.at(i).x + rect.at(i).width, rect.at(i).y + rect.at(i).height));
        point_vec.push_back(cv::Point(rect.at(i).x, rect.at(i).y + rect.at(i).height));
        point_vecs.push_back(point_vec);
        point_vec.clear();
    }
    cv::polylines(debug_im, point_vecs, true, line_color, 4, cv::LINE_AA);
    cv::imwrite("./debug.jpeg", debug_im);
}

void debug_img_rect(const cv::Mat &img, std::vector<cv::Rect> &rect) {
    cv::Mat debug_im = img.clone();
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        line_color = cv::Scalar(0);
    } else {
        line_color = cv::Scalar(255, 0, 0);
    }
    for (int i = 0; i < rect.size(); ++i) {
        point_vec.push_back(cv::Point(rect.at(i).x, rect.at(i).y));
        point_vec.push_back(cv::Point(rect.at(i).x + rect.at(i).width, rect.at(i).y));
        point_vec.push_back(cv::Point(rect.at(i).x + rect.at(i).width, rect.at(i).y + rect.at(i).height));
        point_vec.push_back(cv::Point(rect.at(i).x, rect.at(i).y + rect.at(i).height));
        point_vecs.push_back(point_vec);
        point_vec.clear();
    }
    cv::polylines(debug_im, point_vecs, true, line_color, 4, cv::LINE_AA);
    cv::imwrite("./debug.jpeg", debug_im);
}

void debug_img_rect(const cv::Mat &img, const std::vector<sRotatedRect> &rot_rect, int draw_flag,
                    const std::string &img_name) {
    cv::Mat debug_im = img.clone();
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Scalar point_color;
    cv::Scalar line_color;

    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
    }
    for (int i = 0; i < rot_rect.size(); ++i) {
        cv::Point2f vertices[4];
        rot_rect.at(i).points(vertices);
        cv::Point start_point(int(vertices[0].x), int(vertices[0].y));
        if (draw_flag == 0) {
            cv::circle(debug_im, start_point, 6, point_color, 6, cv::LINE_AA, 0);
        }
        if (draw_flag == 1) {
            float rect_min_y = 0;
            float rect_max_y = 0;
            float rect_max_x = 0;
            for (int k = 0; k < 4; ++k) {
                if (rect_min_y != 0) {
                    rect_min_y = std::min(rect_min_y, vertices[k].y);
                } else {
                    rect_min_y = vertices[k].y;
                }
                rect_max_y = std::max(rect_max_y, vertices[k].y);
                rect_max_x = std::max(rect_max_x, vertices[k].x);
            }
            cv::line(debug_im, cv::Point(0, (int) rect_min_y), cv::Point((int) rect_max_x, (int) rect_min_y),
                     line_color, 4, cv::LINE_AA);
            cv::line(debug_im, cv::Point(0, (int) rect_max_y), cv::Point((int) rect_max_x, (int) rect_max_y),
                     line_color, 4, cv::LINE_AA);
        }
        point_vec.push_back(start_point);
        point_vec.push_back(cv::Point(int(vertices[1].x), int(vertices[1].y)));
        point_vec.push_back(cv::Point(int(vertices[2].x), int(vertices[2].y)));
        point_vec.push_back(cv::Point(int(vertices[3].x), int(vertices[3].y)));
        point_vecs.push_back(point_vec);
        point_vec.clear();
    }
    cv::polylines(debug_im, point_vecs, true, line_color, 4, cv::LINE_AA);
    cv::imwrite(img_name, debug_im);
}


void debug_img_rect(const cv::Mat &img, const sRotatedRect &rot_rect, int draw_flag, const std::string &img_name) {
    cv::Mat debug_im = img.clone();
    cv::Scalar point_color;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
    }
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Point2f vertices[4];
    rot_rect.points(vertices);
    cv::Point start_point(int(vertices[0].x), int(vertices[0].y));
    if (draw_flag == 0) {
        if (img.type() == CV_8UC1) {
            cv::circle(debug_im, start_point, 8, 0, 4, cv::LINE_AA, 0);
        } else {
            cv::circle(debug_im, start_point, 8, (0, 0, 255), 4, cv::LINE_AA, 0);
        }
    }
    if (draw_flag == 1) {
        float rect_min_y = 0;
        float rect_max_y = 0;
        float rect_max_x = 0;
        for (int k = 0; k < 4; ++k) {
            if (rect_min_y != 0) {
                rect_min_y = std::min(rect_min_y, vertices[k].y);
            } else {
                rect_min_y = vertices[k].y;
            }
            rect_max_y = std::max(rect_max_y, vertices[k].y);
            rect_max_x = std::max(rect_max_x, vertices[k].x);
        }
        cv::line(debug_im, cv::Point(0, (int) rect_min_y), cv::Point((int) rect_max_x, (int) rect_min_y),
                 line_color, 4, cv::LINE_AA);
        cv::line(debug_im, cv::Point(0, (int) rect_max_y), cv::Point((int) rect_max_x, (int) rect_max_y),
                 line_color, 4, cv::LINE_AA);
    }
    if (img.type() == CV_8UC1) {
        cv::circle(debug_im, rot_rect.center, 8, 0, 4, cv::LINE_AA, 0);
    } else {
        cv::circle(debug_im, rot_rect.center, 8, (0, 0, 255), 4, cv::LINE_AA, 0);
    }
    point_vec.push_back(start_point);
    point_vec.push_back(cv::Point(int(vertices[1].x), int(vertices[1].y)));
    point_vec.push_back(cv::Point(int(vertices[2].x), int(vertices[2].y)));
    point_vec.push_back(cv::Point(int(vertices[3].x), int(vertices[3].y)));
    point_vecs.push_back(point_vec);
    point_vec.clear();
    if (img.type() == CV_8UC1)
        cv::polylines(debug_im, point_vecs, true, 255, 2, cv::LINE_AA);
    else
        cv::polylines(debug_im, point_vecs, true, (255, 0, 0), 2, cv::LINE_AA);
    cv::imwrite(img_name, debug_im);
}

void debug_img_rect_line(const cv::Mat &img, const std::vector<sRotatedRect> &rot_rect_line, bool draw_start,
                         const std::string &img_name) {
    cv::Mat debug_im = img.clone();
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Point pref_center(rot_rect_line.at(0).center);
    cv::Scalar point_color;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
    }
    for (int i = 0; i < rot_rect_line.size(); ++i) {
        cv::Point2f vertices[4];
        rot_rect_line.at(i).points(vertices);
        cv::Point start_point(int(vertices[0].x), int(vertices[0].y));
        if (i >= 1) {
            cv::Point c_center(rot_rect_line.at(i).center);
            cv::line(debug_im, pref_center, c_center, line_color, 4, cv::LINE_AA);
            pref_center = c_center;
        }
        if (draw_start) {
            cv::circle(debug_im, start_point, 8, point_color, 6, cv::LINE_AA, 0);
        }
        point_vec.push_back(start_point);
        point_vec.push_back(cv::Point(int(vertices[1].x), int(vertices[1].y)));
        point_vec.push_back(cv::Point(int(vertices[2].x), int(vertices[2].y)));
        point_vec.push_back(cv::Point(int(vertices[3].x), int(vertices[3].y)));
        point_vecs.push_back(point_vec);
        point_vec.clear();
    }
    cv::polylines(debug_im, point_vecs, true, line_color, 4, cv::LINE_AA);
    cv::imwrite(img_name, debug_im);
}

void
debug_img_rect_line(const cv::Mat &img, const std::vector<std::vector<sRotatedRect>> &rot_rect_lines, bool draw_start,
                    const std::string &img_name) {
    cv::Mat debug_im = img.clone();
    cv::Scalar point_color;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
    }
    for (int i = 0; i < rot_rect_lines.size(); ++i) {
        std::vector<sRotatedRect> rot_rect_line = rot_rect_lines.at(i);
        std::vector<std::vector<cv::Point>> point_vecs;
        std::vector<cv::Point> point_vec;

        cv::Point pref_center(rot_rect_line.at(0).center);

        for (int i = 0; i < rot_rect_line.size(); ++i) {
            cv::Point2f vertices[4];
            rot_rect_line.at(i).points(vertices);
            cv::Point start_point(int(vertices[0].x), int(vertices[0].y));
            if (i > 1) {
                cv::Point c_center(rot_rect_line.at(i).center);
                cv::line(debug_im, pref_center, c_center, line_color, 4, cv::LINE_AA);
                pref_center = c_center;
            }
            if (draw_start) {
                cv::circle(debug_im, start_point, 8, point_color, 6, cv::LINE_AA, 0);
            }
            point_vec.push_back(start_point);
            point_vec.push_back(cv::Point(int(vertices[1].x), int(vertices[1].y)));
            point_vec.push_back(cv::Point(int(vertices[2].x), int(vertices[2].y)));
            point_vec.push_back(cv::Point(int(vertices[3].x), int(vertices[3].y)));
            point_vecs.push_back(point_vec);
            point_vec.clear();
        }
        cv::polylines(debug_im, point_vecs, true, line_color, 4, cv::LINE_AA);
    }
    cv::imwrite(img_name, debug_im);
}


void debug_img_rect_order(const cv::Mat &img, const std::vector<sRotatedRect> &rot_rects, bool draw_start,
                          const std::string &img_name) {
    cv::Mat debug_im = img.clone();
    std::vector<std::vector<cv::Point>> point_vecs;
    std::vector<cv::Point> point_vec;
    cv::Scalar point_color;
    cv::Scalar line_color;
    cv::Scalar char_color;
    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
        char_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
        char_color = cv::Scalar(0, 255, 0);
    }
    for (int i = 0; i < rot_rects.size(); ++i) {
        cv::Point2f vertices[4];
        rot_rects.at(i).points(vertices);
        cv::Point start_point(int(vertices[0].x), int(vertices[0].y));
        cv::Point c_center(rot_rects.at(i).center);
        if (draw_start) {
            cv::circle(debug_im, start_point, 2, point_color, 1, cv::LINE_AA, 0);
        }
        point_vec.push_back(start_point);
        point_vec.push_back(cv::Point(int(vertices[1].x), int(vertices[1].y)));
        point_vec.push_back(cv::Point(int(vertices[2].x), int(vertices[2].y)));
        point_vec.push_back(cv::Point(int(vertices[3].x), int(vertices[3].y)));
        point_vecs.push_back(point_vec);
        point_vec.clear();
        cv::Size text_size;
//        int *base_line;
//        text_size = cv::getTextSize(std::to_string(i), cv::FONT_HERSHEY_SIMPLEX, 1, 1, base_line);
//        float min_rect_side = std::min(rot_rects.at(i).size.width, rot_rects.at(i).size.height);
//        float size_scale = min_rect_side / text_size.height;
        cv::putText(debug_im, std::to_string(i), c_center, cv::FONT_HERSHEY_SIMPLEX,
                    1, char_color, 1, cv::LINE_AA);
    }
    cv::polylines(debug_im, point_vecs, true, line_color, 1, cv::LINE_AA);
    cv::imwrite(img_name, debug_im);
}

void debug_img_line(const cv::Mat & img, const cv::Vec4f & line, const std::string &img_name) {
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        line_color = cv::Scalar(0);
    } else {
        line_color = cv::Scalar(255, 0, 0);
    }
    cv::Point point0;
    point0.x = line[2];//point on the line
    point0.y = line[3];
    double k = line[1] / line[0]; //slope

    //calculate the endpoint of the line (y = k(x - x0) + y0)
    cv::Point point1, point2;
    point1.y = 0;
    point1.x = (0 - point0.y) / k + point0.x;
    point2.y = img.rows;
    point2.x = (img.rows - point0.y) / k + point0.x;
    if (point1.x < 0 || point1.x >= img.cols - 1 || point2.x < 0 || point2.x >= img.cols - 1) {
        point1.y = (0 - point0.x) * k + point0.y;
        point1.x = 0;
        point2.y = (img.cols - point0.x) * k + point0.y;
        point2.x = img.cols;
    }
    cv::Mat debug_im = img.clone();
    cv::line(debug_im, point1, point2, line_color, 4, cv::LINE_AA);
    cv::imwrite(img_name, debug_im);
}

void debug_img_line(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec, const std::string &img_name) {
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        line_color = cv::Scalar(0);
    } else {
        line_color = cv::Scalar(255, 0, 0);
    }
    cv::Mat debug_im = img.clone();
    for (size_t i = 0; i < line_vec.size(); ++i) {
        cv::Point point0;
        cv::Vec4f line = line_vec.at(i);
        point0.x = line[2];//point on the line
        point0.y = line[3];
        double k = line[1] / line[0]; //slope

        //calculate the endpoint of the line (y = k(x - x0) + y0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = (0 - point0.y) / k + point0.x;
        point2.y = img.rows;
        point2.x = (img.rows - point0.y) / k + point0.x;
        if (point1.x < 0 || point1.x >= img.cols - 1 || point2.x < 0 || point2.x >= img.cols - 1) {
            point1.y = (0 - point0.x) * k + point0.y;
            point1.x = 0;
            point2.y = (img.cols - point0.x) * k + point0.y;
            point2.x = img.cols;
        }
        cv::line(debug_im, point1, point2, line_color, 4, cv::LINE_AA);
    }
    cv::imwrite(img_name, debug_im);
}

void debug_img_line_points(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec,
                           const std::vector<cv::Point2f> & points,
                           const std::string & img_name) {
    cv::Scalar point_color;
    cv::Scalar line_color;
    if (img.type() == CV_8UC1) {
        point_color = cv::Scalar(0);
        line_color = cv::Scalar(0);
    } else {
        point_color = cv::Scalar(0, 0, 255);
        line_color = cv::Scalar(255, 0, 0);
    }
    cv::Mat debug_im = img.clone();
    for (size_t i = 0; i < line_vec.size(); ++i) {
        cv::Point point0;
        cv::Vec4f line = line_vec.at(i);
        point0.x = line[2];//point on the line
        point0.y = line[3];
        double k = line[1] / line[0]; //slope

        //calculate the endpoint of the line (y = k(x - x0) + y0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = (0 - point0.y) / k + point0.x;
        point2.y = img.rows;
        point2.x = (img.rows - point0.y) / k + point0.x;
        if (point1.x < 0 || point1.x >= img.cols - 1 || point2.x < 0 || point2.x >= img.cols - 1) {
            point1.y = (0 - point0.x) * k + point0.y;
            point1.x = 0;
            point2.y = (img.cols - point0.x) * k + point0.y;
            point2.x = img.cols;
        }
        cv::line(debug_im, point1, point2, line_color, 4, cv::LINE_AA);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        cv::Point point(points.at(i).x, points.at(i).y);
        cv::circle(debug_im, point, 12, point_color, 4, cv::LINE_AA, 0);
    }
    cv::imwrite(img_name, debug_im);
}

void debug_img_line_points(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec,
                           const cv::Point2f  points [],
                           const std::string & img_name){
    int n = sizeof(points) / sizeof(points[0]);
    std::vector<cv::Point2f> point_vec(points, points + n);
    debug_img_line_points(img, line_vec, point_vec, img_name);
}

void showTensor(at::Tensor &input_tensor, int nc) {
    at::Tensor img_tensor = input_tensor.mul(255).clamp(0, 255).to(torch::kU8).clone();
    int num_dim = img_tensor.dim();
    int height;
    int width;
    if (num_dim == 4) {
        height = img_tensor.size(1);
        width = img_tensor.size(2);
    } else if (num_dim == 3) {
        if (img_tensor.size(2) == nc) {
            height = img_tensor.size(0);
            width = img_tensor.size(1);
        } else {
            height = img_tensor.size(1);
            width = img_tensor.size(2);
        }
    } else {
        height = img_tensor.size(0);
        width = img_tensor.size(1);
    }
    assert(height > 0 and width > 0);
    std::cout << "numel:" << img_tensor.numel() << " ";
    std::cout << "max:" << torch::max(img_tensor).item<float>() << " min:" << torch::min(img_tensor).item<float>()
              << std::endl;
    cv::Mat test_img;
    if (nc == 3) {
        test_img = cv::Mat(height, width, CV_8UC3);
    } else {
        test_img = cv::Mat(height, width, CV_8UC1);
    }

    std::memcpy((void *) test_img.data, img_tensor.data_ptr(), sizeof(torch::kU8) * img_tensor.numel());
    cv::imwrite("./debug.jpeg", test_img);
}

void printTensor(at::Tensor &input_tensor, const std::string &name) {
    int ndim = input_tensor.dim();
    std::cout << "tensor_name:" << name << " " << " ndim:" << ndim << "";
    for (int i = 0; i < ndim; ++i) {
        std::cout << " dim" << i << ":" << input_tensor.size(i) << "";
    }
    std::cout << " numel:" << input_tensor.numel() << "";
    std::cout << " dtype:" << input_tensor.dtype() << "";
    std::cout << " device:" << input_tensor.device() << std::endl;
}


void printTensorV(at::Tensor &input_tensor, const std::string &name) {
    std::cout << "tensor_name:" << name << " " << " ndim:" << input_tensor.dim() << "";
    std::cout << " mean:" << torch::mean(input_tensor).item<float>() << "";
    std::cout << " max:" << torch::max(input_tensor).item<float>() << "";
    std::cout << " min:" << torch::min(input_tensor).item<float>() << std::endl;
}

void showTime(int time, const std::string &name, const std::string &unit) {
    std::cout << "procedure:" << name << " cost " << time << unit << std::endl;
}

void logMsg(const std::string &msg) {
    std::cout << msg << std::endl;
}

