/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-04-01 09:35
**/
//
// Created by 周炳诚 on 2021/4/1.
//
#include <sys/stat.h>
#include <opencv2/calib3d.hpp>
#include "td.h"

#define VIEW_TEST

using namespace torch::indexing;
using namespace cv;

bool perspective_img(Mat & org_img, std::vector<sRotatedRect> s_rotated_rect, const std::string & res_file_name="./debug.jpeg"){
    std::chrono::steady_clock::time_point test_begin = std::chrono::steady_clock::now();
    debug_img_rect_order(org_img, s_rotated_rect, true, res_file_name);
    std::vector<sRotatedRect> p_rotated_rect;
//    bool horizontal_flag = is_horizontal(s_rotated_rect);
//    if (horizontal_flag)
//        return false;
    filter_rotated_rects(s_rotated_rect, p_rotated_rect);
    debug_img_rect_order(org_img, p_rotated_rect, true, res_file_name);
    std::vector<sRotatedRect> row_rotated_rects;
    get_row_rotated_rects(p_rotated_rect, row_rotated_rects, 12);
//    get_row_rotated_rects(p_rotated_rect, row_rotated_rects,
//                          24, true, "/data/duser/bczhou_test/warp_bank_all_files/bad_case_imgs/867ec0f8-a1a3-11eb-bda2-02420a002c2f.jpg");
    if (row_rotated_rects.size() <= 3) {
        return false;
    }
    std::vector<std::vector<sRotatedRect>> line_rotated_rect_vecs;
    std::vector<sRotatedRect> new_row_rects;
    std::map<size_t , size_t> rect_num_counter;
    for (int j = 0; j < row_rotated_rects.size(); ++j) {
        std::vector<sRotatedRect> line_rotated_rect_vec;
        std::vector<Point> line_inner_points;
        //TODO optimize filter rotated rects.
        for (int k = 0; k < p_rotated_rect.size(); ++k) {
            if (row_rotated_rects.at(j).inside(p_rotated_rect.at(k).center)) {
                line_rotated_rect_vec.push_back(p_rotated_rect.at(k));
                for (auto point:p_rotated_rect.at(k).inner_points) {
                    line_inner_points.push_back(point);
                }
            }
        }
        auto it = rect_num_counter.find(line_rotated_rect_vec.size());
        if ( it == rect_num_counter.end()){
            rect_num_counter.insert(std::make_pair(line_rotated_rect_vec.size(),  1));
        }else{
            it->second += 1;
        }
        sRotatedRect new_row_rect(line_inner_points);
        new_row_rects.push_back(new_row_rect);
        std::sort(line_rotated_rect_vec.begin(), line_rotated_rect_vec.end(), rotate_rect_sort_by_x);
        line_rotated_rect_vecs.push_back(line_rotated_rect_vec);
        line_rotated_rect_vec.clear();
    }
#ifdef VIEW_TEST
    debug_img_rect_order(org_img, new_row_rects, false, res_file_name);
#endif
//    for (auto iter = rect_num_counter.begin(); iter!= rect_num_counter.end(); iter++) {
//        std::cout << "rect_num_counter:" << iter->first << "," << iter->second << std::endl;
//    }
    std::cout << "rect_num_counter:" << rect_num_counter << std::endl;
    auto max_num_key = get_max_num_key<size_t>(rect_num_counter);
    std::cout << "max_num_key:" << max_num_key << std::endl;
    Vec4f top_line;
    Vec4f bottom_line;
    std::vector<sRotatedRect> left_rotated_rect;
    std::vector<sRotatedRect> right_rotated_rect;
    for (int j = 0; j < new_row_rects.size(); ++j) {
        if (line_rotated_rect_vecs.at(j).size() >= 2){
            size_t row_size = line_rotated_rect_vecs.at(j).size();
            if (max_num_key.first >= new_row_rects.size() / 3 && max_num_key.first >= 4){
                if (line_rotated_rect_vecs.at(j).size() == max_num_key.second){
                    left_rotated_rect.push_back(line_rotated_rect_vecs.at(j).at(0));
                    right_rotated_rect.push_back(line_rotated_rect_vecs.at(j).at(row_size - 1));
                }
            }else{
                left_rotated_rect.push_back(line_rotated_rect_vecs.at(j).at(0));
                right_rotated_rect.push_back(line_rotated_rect_vecs.at(j).at(row_size - 1));
            }
            if (j == 0){
                Point2f rect_points[4];
                new_row_rects.at(j).order_points(rect_points);
                Point2f left_point((rect_points[1].x + rect_points[2].x) / 2, (rect_points[1].y + rect_points[2].y) / 2);
                Point2f right_point((rect_points[0].x + rect_points[3].x) / 2, (rect_points[0].y + rect_points[3].y) / 2);
                Point left_point_i(left_point.x, left_point.y);
                Point right_point_i(right_point.x, right_point.y);
                top_line = get_line_by_point(left_point_i, right_point_i);
            }
            else if (j == new_row_rects.size() - 1){
                Point2f rect_points[4];
                new_row_rects.at(j).order_points(rect_points);
                Point2f left_point((rect_points[1].x + rect_points[2].x) / 2, (rect_points[1].y + rect_points[2].y) / 2);
                Point2f right_point((rect_points[0].x + rect_points[3].x) / 2, (rect_points[0].y + rect_points[3].y) / 2);
                Point left_point_i(left_point.x, left_point.y);
                Point right_point_i(right_point.x, right_point.y);
                bottom_line = get_line_by_point(left_point_i, right_point_i);
            }
        }
    }
    std::vector<sRotatedRect> left_col_rotated_rect;
    std::vector<sRotatedRect> right_col_rotated_rect;
#ifdef VIEW_TEST
    debug_img_rect_order(org_img, left_rotated_rect, true, res_file_name);
#endif
    get_col_rotated_rects(left_rotated_rect, left_col_rotated_rect, 8);
#ifdef VIEW_TEST
    debug_img_rect_order(org_img, left_col_rotated_rect, true, res_file_name);
    debug_img_rect_order(org_img, right_rotated_rect, true, res_file_name);
#endif
    get_col_rotated_rects(right_rotated_rect, right_col_rotated_rect, 8);
#ifdef VIEW_TEST
    debug_img_rect_order(org_img, right_col_rotated_rect, true, res_file_name);
#endif

    if (left_col_rotated_rect.size() == 0 || right_col_rotated_rect.size() == 0){
        return false;
    }

    std::vector<std::vector<Point2f>> left_points(3);
    std::vector<std::vector<Point2f>> right_points(3);
    for (auto p=left_col_rotated_rect.begin(); p!=left_col_rotated_rect.end(); p++) {
        Point2f left_rect_points[4];
        p->order_points(left_rect_points);
        Point2f left_point1((left_rect_points[1].x + left_rect_points[2].x) / 2,
                            (left_rect_points[1].y + left_rect_points[2].y) / 2);
        Point2f left_point2 = p->center;
        Point2f left_point3((left_rect_points[0].x + left_rect_points[3].x) / 2,
                            (left_rect_points[0].y + left_rect_points[3].y) / 2);
        left_points.at(0).push_back(left_point1);
        left_points.at(1).push_back(left_point2);
        left_points.at(2).push_back(left_point3);
    }
    for (auto p=right_col_rotated_rect.begin(); p!=right_col_rotated_rect.end(); p++) {
        Point2f right_rect_points[4];
        p->order_points(right_rect_points);
        Point2f right_point1((right_rect_points[1].x + right_rect_points[2].x) / 2,
                            (right_rect_points[1].y + right_rect_points[2].y) / 2);
        Point2f right_point2 = p->center;
        Point2f right_point3((right_rect_points[0].x + right_rect_points[3].x) / 2,
                            (right_rect_points[0].y + right_rect_points[3].y) / 2);
        right_points.at(0).push_back(right_point1);
        right_points.at(1).push_back(right_point2);
        right_points.at(2).push_back(right_point3);
    }
    Vec4f left_line;
    Vec4f right_line;
    std::sort(right_points.begin(), right_points.end(),
              [](std::vector<Point2f> &a, std::vector<Point2f> &b) {return get_points_x_error(a) < get_points_x_error(b);});
    std::sort(left_points.begin(), left_points.end(),
              [](std::vector<Point2f> &a, std::vector<Point2f> &b) {return get_points_x_error(a) < get_points_x_error(b);});
    for (int i = 0; i < left_points.at(0).size(); ++i) {
        std::cout << "left point " << i << ":" << left_points.at(0).at(i) << std::endl;
    }
    for (int i = 0; i < right_points.at(0).size(); ++i) {
        std::cout << "right_points " << i << ":" << right_points.at(0).at(i) << std::endl;
    }
    fitLine(left_points.at(0), left_line, DIST_L1, 0, 1e-2, 1e-2);
    fitLine(right_points.at(0), right_line, DIST_L1, 0, 1e-2, 1e-2);
    if (is_horizon_line(top_line) && is_horizon_line(bottom_line) && is_vertical_line(left_line)){
        return false;
    }
    std::cout << "top_line:" << top_line << std::endl;
    std::cout << "bottom_line:" << bottom_line << std::endl;
    std::cout << "left_line:" << left_line << std::endl;
    std::cout << "right_line:" << right_line << std::endl;
    std::vector<Vec4f> lines;
    lines.push_back(top_line);
    lines.push_back(bottom_line);
    lines.push_back(left_line);
    lines.push_back(right_line);
    Point2f points[4];
    bool flag;
    flag = intersection(top_line, left_line, points[0], org_img.cols, org_img.rows); // lt
    if(!flag){
        return false;
    };
    flag = intersection(left_line, bottom_line, points[1], org_img.cols, org_img.rows); // lb
    if(!flag){
        return false;
    };
    flag = intersection(bottom_line, right_line, points[2], org_img.cols, org_img.rows); // rb
    if(!flag){
        return false;
    };
    flag = intersection(right_line, top_line, points[3], org_img.cols, org_img.rows); // rb
    if(!flag){
        return false;
    };
#ifdef VIEW_TEST
    debug_img_line_points(org_img, lines, points, res_file_name);
#endif
    Point2f left_mid_point = (points[0] + points[1]) / 2;
    Point2f right_mid_point = (points[2] + points[3]) / 2;
    Point2f top_mid_point = (points[0] + points[3]) / 2;
    Point2f bottom_mid_point = (points[1] + points[2]) / 2;
    Point2f w_dif_point(point_distance(left_mid_point, right_mid_point), 0);
    Point2f h_dif_point(0, point_distance(top_mid_point, bottom_mid_point));
    Point2f dst_points[4];
    dst_points[0] = points[0];
    dst_points[1] = points[0] + h_dif_point;
    dst_points[2] = points[0] + h_dif_point + w_dif_point;
    dst_points[3] = points[0] + w_dif_point;
    Mat transform_matrix = getPerspectiveTransform(points, dst_points);

//        std::cout << "angle is:" << angle << std::endl;
    std::cout << "s_rotated_rect size:" << s_rotated_rect.size() << std::endl;
    std::cout << "p_rotated_rect size:" << p_rotated_rect.size() << std::endl;
    std::cout << "row_line_nums:" << row_rotated_rects.size() << std::endl;
    std::cout << "transform_matrix:" << transform_matrix << std::endl;
    std::cout << "get current line intersection points,lt_point:" << points[0] << ",lb_point:" << points[1]
              << ",br_point:" << points[2] << ",rt_point:" << points[3] << std::endl;
    float transform_array [9];
    for (int m=0; m<3; ++m){
        for (int n=0; n<3; ++n){
            transform_array[m * 3 + n] = (float) transform_matrix.at<double>(m, n);
        }
    }
    transform_matrix.convertTo(transform_matrix, CV_32F);
    std::vector<std::vector<Point2f>> rect_points_vecs;
    std::vector<float> score_vec;
    for (int j = 0; j < s_rotated_rect.size(); ++j) {
        std::vector<Point2f> rect_points_vec;
        for (auto point:s_rotated_rect.at(j).inner_points) {
            rect_points_vec.push_back(Point2f(point.x, point.y));
        }
        rect_points_vecs.push_back(rect_points_vec);
        score_vec.push_back(s_rotated_rect.at(j).score);
    }
    int origin_rot_size = s_rotated_rect.size();

    std::cout << "current img rows:" << org_img.rows << ",img cols:" << org_img.cols << std::endl;
    warpPerspective(org_img, org_img, transform_matrix, Size(org_img.cols, org_img.rows),
                        INTER_AREA, BORDER_REPLICATE);
    std::cout << "current img rows:" << org_img.rows << ",img cols:" << org_img.cols << std::endl;
    std::vector<sRotatedRect> res_rotated_rect;
    for (int j = 0; j < rect_points_vecs.size(); ++j) {
        perspectiveTransform(rect_points_vecs.at(j), rect_points_vecs.at(j), transform_matrix);
        sRotatedRect rotated_rect(rect_points_vecs.at(j), score_vec.at(j));
        res_rotated_rect.push_back(rotated_rect);
//        if (rotated_rect.center.x > org_img.cols || rotated_rect.center.y > org_img.rows
//        || rotated_rect.center.x < 0 || rotated_rect.center.y < 0){
//            return false;
//        }
//        for (auto point:rect_points_vecs.at(j)) {
//            if (point.x > org_img.cols || point.y > org_img.rows || point.x < 0 || point.y <0){
//                return false;
//            }
//        }
    }
    assert(res_rotated_rect.size() == origin_rot_size);
    s_rotated_rect = res_rotated_rect;
//        std::cout << "transform_array:" << transform_array << std::endl;
    std::chrono::steady_clock::time_point test_end = std::chrono::steady_clock::now();
    int time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
    showTime(time_cost, "perspective transform aft process");

//    s_rotated_rect.clear();
    imwrite(res_file_name, org_img);
    p_rotated_rect.clear();
    row_rotated_rects.clear();
    return true;
}

int main(int argc, const char *argv[]) {
    std::string modelPath = "/workspace/data/online_models/jit_pse_document.pt";
    dl_model::td::PseModel pse_model(modelPath, "gpu", 0, true);
    std::string imgPath = "/workspace/test1.jpg";
    Mat img;
    img = imread(imgPath, IMREAD_COLOR);
    Mat org_img = img.clone();
    cvtColor(img, img, COLOR_BGR2RGB);
    std::vector<sRotatedRect> s_rotated_rect;
    pse_model.predict(img, s_rotated_rect, true);
    std::string res_file_name = "/workspace/res1.jpg";
    bool flag = perspective_img(org_img, s_rotated_rect, res_file_name);

    if(!flag){
        std::cout << "failed to perspective img:" << imgPath << std::endl;
        imwrite(res_file_name, org_img);
    }else{
        std::cout << "success to perspective img:" << imgPath << std::endl;
        Mat test_img = imread(imgPath, IMREAD_COLOR);
        imwrite(res_file_name, org_img);
    }
    s_rotated_rect.clear();
    return 0;
}

//int main(int argc, const char *argv[]) {
//
//    std::string modelPath = "/data/duser/models/ysocr_models/detection_models/pse/jit_pse_normal.pt";
////    std::string imgPath = "/data/duser/bczhou_test/warp_bank/yinhang_guangda_0_paizhao_0.JPG";
//    std::string path1 = "/data/duser/bczhou_test/4de229e8-f97c-11eb-bc5d-02420a02007d/*.jpg";
////    std::string path2 = "/data/duser/bczhou_test/warp_bank_all_files/allfiles/*.JPG";
//    std::string out_dir = "/data/duser/bczhou_test/warp_bank_all_files/bad_case_row_imgs/";
//    if (mkdir(out_dir.c_str(), 0777) == -1){
//        std::cout << "Error:" << "failed to create dir:" << out_dir << std::endl;
//    }
//    std::vector<std::string> file_names;
//    get_glob_files(path1, file_names);
////    get_glob_files(path2, file_names);
//    dl_model::td::PseModel pse_model(modelPath, "gpu", 0, true);
//    Mat img;
//    std::set<char> delims{'/'};
//    for (size_t i = 0; i < file_names.size(); ++i) {
//        if (file_names.at(i) != "/data/duser/bczhou_test/warp_bank_all_files/bad_case_imgs/5e635476-a65d-11eb-a363-02420a015bb6_page2.jpg") {
//            continue;
//        }
//        std::cout << "handling file:" << file_names.at(i) << std::endl;
//        std::vector<std::string> path_splits = split_path(file_names.at(i), delims);
//        std::string res_file_name = out_dir + path_splits.back();
//        img = imread(file_names.at(i), IMREAD_COLOR);
//        Mat org_img = img.clone();
//        cvtColor(img, img, COLOR_BGR2RGB);
//        std::vector<sRotatedRect> s_rotated_rect;
//        pse_model.predict(img, s_rotated_rect, true);
//        bool flag = perspective_img(org_img, s_rotated_rect, res_file_name);
//        if(!flag){
//            std::cout << "failed to perspective img:" << file_names.at(i) << std::endl;
//            imwrite(res_file_name, org_img);
//        }else{
//            std::cout << "success to perspective img:" << file_names.at(i) << std::endl;
//            Mat test_img = imread(file_names.at(i), IMREAD_COLOR);
//            imwrite(res_file_name, org_img);
//        }
//        s_rotated_rect.clear();
////        debug_img_rect_order(org_img, s_rotated_rect, true, res_file_name);
////        float angle = get_mean_angle(s_rotated_rect, img.cols);
////        rotate_img_box_points(org_img, s_rotated_rect, 1);
//
//    }
//
//    return 0;
//}

