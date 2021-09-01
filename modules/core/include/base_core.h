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

#ifndef __BASE_CORE_H__
#define __BASE_CORE_H__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <nlohmann/json.hpp>

#ifndef DEFAULT_DEVICE_ID
#define DEFAULT_DEVICE_ID 0
#endif

namespace dl_model{

    struct BaseConfig {

    };

    enum ModelType {
        PSE,
        CRNN,
    };

    struct BaseModel{

        ModelType mt;
        torch::jit::script::Module module;
        at::Device device = at::Device(at::kCPU);
        bool half = false;

        BaseModel(){};

        inline void create(torch::jit::script::Module & _module, const std::string & mode, int index, bool _half);

        BaseModel(const std::string & model_path, const std::string & mode="gpu", int index=DEFAULT_DEVICE_ID, bool _half=false);

        BaseModel(torch::jit::script::Module & _module, const std::string & mode="gpu", int index=DEFAULT_DEVICE_ID, bool _half=false);


    };
}

//score RotateRect
struct sRotatedRect: public cv::RotatedRect{
    using cv::RotatedRect::boundingRect;
    using cv::RotatedRect::boundingRect2f;
    using cv::RotatedRect::points;
    using cv::RotatedRect::RotatedRect;

    // average score
    float score;
    // split points
    std::vector<cv::Point> inner_points;

    /*!
     *
     * @param rect 原始旋转矩形
     * @param score 平均得分
     * @param scale 缩放比例
     */
    sRotatedRect(const sRotatedRect & rect);

    sRotatedRect(sRotatedRect & rect);

    sRotatedRect(std::vector<cv::Point> _points, float _score=1.0, float scale=1.0);

    sRotatedRect(std::vector<cv::Point2f> _points, float _score=1.0, float scale=1.0);

    sRotatedRect(const cv::RotatedRect & rect, float _score=1.0, float scale=1.0);

    sRotatedRect(cv::RotatedRect &rect, float _score=1.0, float scale=1.0);

    sRotatedRect(const cv::Point2f & _center, const cv::Size2f & _size, float _angle, float _score=1.0):
    RotatedRect(_center, _size, _angle), score{_score}{};

    sRotatedRect(const cv::Point2f & point1, const cv::Point2f & point2, const cv::Point2f & point3, float _score=1.0):
            RotatedRect(point1, point2, point3), score{_score}{};

    bool is_flat();

    float dist(const sRotatedRect & other);

    float dist(const cv::Point2f & other_center);

    /*!
     * 两个水平矩形框的other框到这个矩形框短边的角度距离
     * @param other
     * @return
     */
    float angle_dist(const sRotatedRect & other);

    /*!
     * 两个水平矩形框的other框到这个矩形框短边的投影距离，用于衡量两个矩形框是不是在同一行.
     * @param other
     * @return
     */
    float proj_dist_x(const sRotatedRect & other);

    float proj_dist_x(const cv::Point2f & other_center);

    void order_points(cv::Point2f pts[]) const;
    /*!
     * 判断point是否在point的两条长边之间，只考虑简单情况.
     * @param point
     * @return
     */
    bool inside(cv::Point2f & point);

    friend bool is_same_col(const sRotatedRect & rec1, const sRotatedRect & rec2);

    friend std::ostream& operator<<(std::ostream &os, const sRotatedRect & s_rect);

};

struct TextRect: public cv::Rect {
    using cv::Rect::size;
    using cv::Rect::area;
    using cv::Rect::br;
    using cv::Rect::tl;

    std::wstring text;
//    float d_score; //detection score

    TextRect(const TextRect & rect);

    TextRect(TextRect & rect);

    TextRect(const cv::Rect & rect);

    TextRect(cv::Rect & rect);

    TextRect(const cv::Rect & rect, const std::wstring & _text);

    TextRect(cv::Rect & rect, const std::wstring & _text);

    nlohmann::json dump();
};

#endif //#define __BASE_CORE_H__

