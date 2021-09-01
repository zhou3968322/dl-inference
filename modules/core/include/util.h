//
// Created by 周炳诚 on 2021/3/26.
//

#ifndef __UTIL_H__
#define __UTIL_H__

#include <unordered_map>
#include <torch/script.h> // One-stop header.
#include <math.h>
#include <opencv2/opencv.hpp>
#include "base_core.h"

template<typename T1, typename T2> T2 get_map_default_value(std::unordered_map<T1, T2> map, const T1 & key, const T2 & dvalue){
    if (map.find(key) == map.end())
        return map.at(key);
    else
        return dvalue;
};

/*!
 * template cannot put into cpp file directly
 * see https://stackoverflow.com/questions/3040480/c-template-function-compiles-in-header-but-not-implementation
 * @tparam K
 * @param my_map
 * @return
 */
template<typename K> std::pair<size_t, K> get_max_num_key(std::map<K, size_t> & my_map){
    std::vector<std::pair<size_t, K>> pair_vec;
    for (auto const &pair: my_map) {
        pair_vec.push_back(std::make_pair(pair.second, pair.first));
    }
    assert(pair_vec.size() > 0);
    std::sort(pair_vec.begin(), pair_vec.end(),
              [](std::pair<size_t, K> &a, std::pair<size_t, K> &b) {
                  if (a.first != b.first)
                      return a.first > b.first;
                  return a.second > b.second;
              });
    return pair_vec.at(0);
};

bool rect_sort_by_ratio(cv::Rect & rect1, cv::Rect & rect2);

std::vector<size_t> sort_indexes_by_x(std::vector<sRotatedRect> &v);

bool rotate_rect_sort_by_y(sRotatedRect & s_rect1, sRotatedRect & s_rect2);

bool rotate_rect_sort_by_x(sRotatedRect & s_rect1, sRotatedRect & s_rect2);

void get_rect_tensor(std::vector<sRotatedRect> & rot_vec, at::Tensor & box_tensor, at::Tensor & box_prop_tensor);

at::Tensor get_rect_box_tensor(std::vector<sRotatedRect> & rot_vec);

std::pair<at::Tensor, at::Tensor> get_rect_tensor(std::vector<sRotatedRect> & rot_vec);

void get_rect_tensor(std::vector<sRotatedRect> & rot_vec, at::Tensor & box_tensor);

float get_median_angle(const std::vector<sRotatedRect> & rot_vec, int img_width);

float get_mean_angle(const std::vector<sRotatedRect> &rot_vec, int img_width, int min_side=2, float min_rect_ratio=2.0);

bool is_horizontal(const std::vector<sRotatedRect> &rot_vec, int min_side=2, float min_rect_ratio=2.0, float hor_ratio=0.9);

bool is_horizon_line(const cv::Vec4f & line);

bool is_vertical_line(const cv::Vec4f & line);

/*!
 * 获取两点之间的距离
 * @param pointA
 * @param pointB
 * @return
 */
float point_distance(cv::Point2f & pointA, cv::Point2f & pointB);

/*!
 * 获取点P到直线AP的距离
 * @param pointP
 * @param pointA
 * @param pointB
 * @return
 */
float get_point2line_dis(cv::Point & pointP, cv::Point & pointA, cv::Point & pointB);

/*!
 * 通过点获取线的交点
 * @param o1
 * @param p1
 * @param o2
 * @param p2
 * @param r
 * @return
 */
bool intersection(cv::Point2f & o1, cv::Point2f & p1, cv::Point2f & o2, cv::Point2f & p2,
                  cv::Point2f &r);

bool intersection(cv::Vec4f & line1, cv::Vec4f & line2,
                  cv::Point2f &r, int max_x=0, int max_y=0);

/*!
 * 根据point 获取线
 * @param pointA
 * @param pointB
 * @return
 */
cv::Vec4f get_line_by_point(cv::Point & pointA, cv::Point & pointB);

/*!
 * 获取点之间的误差值
 * @param points 点集合
 * @param method 0代表l1, 1代表l2
 */
float get_points_x_error(std::vector<cv::Point2f> points, int method=0);

/*!
 *
 * @param img input 图像
 * @param rot_vec rotated rect vec 包含score
 */

void rotate_img_box_points(cv::Mat & img, std::vector<sRotatedRect> & rot_vec, int method=0);

/*!
 *
 * @param img input 图像
 * @param rot_vec rotated rect vec 包含score
 * @param angle 角度，角度制
 */
void rotate_img_box_points(cv::Mat & img, std::vector<sRotatedRect> & rot_vec, float angle);

void rot_vec2rect(const std::vector<sRotatedRect> & rot_vec, std::vector<cv::Rect> & rect_vec);

std::vector<cv::Rect> rot_vec2rect(const std::vector<sRotatedRect> & rot_vec);

void filter_rotated_rects(std::vector<sRotatedRect> & s_rotated_rect, std::vector<sRotatedRect> & p_rotated_rect, int min_side=2,float min_rect_ratio=2);

std::vector<sRotatedRect> filter_rotated_rects(std::vector<sRotatedRect> & s_rotated_rect, int min_side=2,float min_rect_ratio=2);

void get_row_rotated_rects(std::vector<sRotatedRect> & p_rotated_rect, std::vector<sRotatedRect> & row_rotated_rects, int half_nums=6, bool debug=false, const std::string & img_path = "");

std::vector<sRotatedRect> get_row_rotated_rects(std::vector<sRotatedRect> & p_rotated_rect, int half_nums=6);

void get_col_rotated_rects(std::vector<sRotatedRect> & p_rotated_rect, std::vector<sRotatedRect> & col_rotated_rects, int half_nums=6);

std::vector<sRotatedRect> get_col_rotated_rects(std::vector<sRotatedRect> & p_rotated_rect, int half_nums=6);

void rect_to_transform_tensor(at::Tensor & A, at::Tensor & B, sRotatedRect & s_rect, int i, int equation_nums=3);

void get_diag_invert(at::Tensor & diag, at::Tensor & diag_invert);

at::Tensor get_diag_invert(at::Tensor & diag);

void solve_linear_system_by_svd(at::Tensor & A, at::Tensor & B, at::Tensor & X);

at::Tensor solve_linear_system_by_svd(at::Tensor & A, at::Tensor & B);


////
////
//// DEBUG FUNCTIONS
////
////




void debug_img_rect(const cv::Mat & img, std::vector<TextRect> & rect);

void debug_img_rect(const cv::Mat & img, std::vector<cv::Rect> & rect);

/*!
 *
 * @param img
 * @param rot_rect
 * @param draw_flag 0:draw_start,画起始点; 1:draw_rect:画水平方向直线.
 * @param img_name
 */
void debug_img_rect(const cv::Mat & img, const std::vector<sRotatedRect> & rot_rect,
                    int draw_flag = 0, const std::string & img_name="./debug.jpeg");

void debug_img_rect(const cv::Mat & img, const sRotatedRect & rot_rect,
                    int draw_flag = 0, const std::string & img_name="./debug.jpeg");

void debug_img_rect_line(const cv::Mat & img, const std::vector<sRotatedRect> & rot_rect_line,
                    bool draw_start = false, const std::string & img_name="./debug.jpeg");

void debug_img_rect_line(const cv::Mat & img, const std::vector<std::vector<sRotatedRect>> & rot_rect_line,
                         bool draw_start = false, const std::string & img_name="./debug.jpeg");

void debug_img_rect_order(const cv::Mat & img, const std::vector<sRotatedRect> & rot_rects, bool draw_start = false,
                          const std::string & img_name = "./debug.jpeg");

void debug_img_line(const cv::Mat &img, const cv::Vec4f & line, const std::string &img_name = "./debug.jpeg");

void debug_img_line(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec, const std::string &img_name = "./debug.jpeg");

void debug_img_line_points(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec,
                           const std::vector<cv::Point2f> & points, const std::string & img_name);

void debug_img_line_points(const cv::Mat &img, const std::vector<cv::Vec4f> & line_vec,
                           const cv::Point2f  points [],
                           const std::string & img_name);

void showTensor(at::Tensor & input_tensor, int nc=1);

void printTensor(at::Tensor & input_tensor, const std::string & name = "");

void printTensorV(at::Tensor &input_tensor, const std::string &name = "");

void showTime(int time, const std::string & name="",  const std::string & unit="[ms]");

void logMsg(const std::string & msg);



// FILE UTILS


std::vector<std::string> get_glob_files(const std::string& pattern);


void get_glob_files(const std::string& pattern, std::vector<std::string> & file_names);

std::vector<std::string> split_path(const std::string & str, const std::set<char> delimiters);

#endif //__UTIL_H__
