/**
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
* @program: ocr
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-26 11:33
**/
//
// Created by 周炳诚 on 2021/3/26.
//

#include "ocr.h"

namespace pipeline {
    using namespace dl_model;
    OcrPipeLine::OcrPipeLine(td::PseModel & _td_model, tr::CrnnModel & _tr_model){
        td_model = _td_model;
        tr_model = _tr_model;
    }

    OcrPipeLine::OcrPipeLine(const std::string & td_model_path, const std::string & tr_model_path, const std::string & vocab_path,
                int _max_height, int _tr_nc,  const std::string & mode, int index, bool td_half){
        td_model = td::PseModel(td_model_path, mode, index, td_half);
        tr_model = tr::CrnnModel(tr_model_path, vocab_path, _max_height, _tr_nc, mode, index);
    }


    void OcrPipeLine::predict(cv::Mat & img, std::vector<TextRect> & text_rect_vec){
        assert(text_rect_vec.size() == 0);
        std::vector<sRotatedRect> s_rotated_rect;
#ifdef DEBUG
        std::cout << "img height:" << img.rows << ";img width:" << img.cols << std::endl;
#endif
#ifdef DEBUG_TIME
        std::chrono::steady_clock::time_point test_begin = std::chrono::steady_clock::now();
#endif
        td_model.predict(img, s_rotated_rect, true);
        if (s_rotated_rect.size() == 0){
            return;
        }
#ifdef DEBUG_TIME
        std::chrono::steady_clock::time_point test_end = std::chrono::steady_clock::now();
        int time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
        showTime(time_cost, "all detection");
#endif
#ifdef DEBUG
        std::cout << "img height:" << img.rows << ";img width:" << img.cols
            <<";rotated_rect_size:" << s_rotated_rect.size() << std::endl;
        debug_img_rect(img, s_rotated_rect);
#endif
#ifdef DEBUG_TIME
        test_begin = std::chrono::steady_clock::now();
#endif
        rotate_img_box_points(img, s_rotated_rect);
#ifdef DEBUG
        debug_img_rect(img, s_rotated_rect);
#endif
        std::vector<cv::Rect> rect_vec;
        rot_vec2rect(s_rotated_rect, rect_vec);
#ifdef DEBUG_TIME
        test_end = std::chrono::steady_clock::now();
        time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
        showTime(time_cost, "rotated box and get rect");
#endif
#ifdef DEBUG
        debug_img_rect(img, rect_vec);
        std::cout << "img height:" << img.rows << ";img width:" << img.cols
                  <<";rect_vec:" << rect_vec.size() << std::endl;
#endif
#ifdef DEBUG_TIME
        test_begin = std::chrono::steady_clock::now();
#endif
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        tr_model.predict(img, rect_vec, text_rect_vec);
#ifdef DEBUG_TIME
        test_end = std::chrono::steady_clock::now();
        time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
        showTime(time_cost, "all recognition");
#endif
#ifdef DEBUG
        debug_img_rect(img, text_rect_vec);
#endif
    }

    std::vector<TextRect> OcrPipeLine::predict(cv::Mat & img){
        std::vector<TextRect> text_rect_vec;
        predict(img, text_rect_vec);
        return text_rect_vec;
    }

    void OcrPipeLine::predict(const std::string & img_path, std::vector<TextRect> & text_rect_vec){
        cv::Mat img;
        if (td_model.nc == 3){
            img = cv::imread(img_path, cv::IMREAD_COLOR);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        } else{
            img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        }
        predict(img, text_rect_vec);
    }

    std::vector<TextRect> OcrPipeLine::predict(const std::string & img_path){
        std::vector<TextRect> text_rect_vec;
        predict(img_path, text_rect_vec);
        return text_rect_vec;
    }
}


