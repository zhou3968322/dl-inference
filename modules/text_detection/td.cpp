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
* @create: 2021-03-26 11:10
**/
//
// Created by 周炳诚 on 2021/3/26.
//

#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include "util.h"
#include "td.h"
#include "td_config.h"
#include "test_pse.h"


namespace dl_model {

    namespace td {

        using namespace torch::indexing;
        namespace F = torch::nn::functional;

        inline void PseConfig::create(int _long_size, float _scale, int _kernel_nums, int _min_kernel_area,
                                      int _min_area, float _min_score, float _mag_ratio, int _nc) {
            long_size = _long_size;
            scale = _scale;
            kernel_nums = _kernel_nums;
            min_kernel_area = _min_kernel_area;
            min_area = _min_area;
            min_score = _min_score;
            mag_ratio = _mag_ratio;
            nc = _nc;
        }

        inline void PseConfig::create(const nlohmann::json & j){
            long_size = j.value("long_size", 2240);
            scale = j.value("scale", 1.0);
            kernel_nums = j.value("kernel_nums", 4);
            min_kernel_area = j.value("min_kernel_area", 10);
            min_area = j.value("min_area", 10);
            min_score = j.value("min_score", 0.8);
            mag_ratio = j.value("mag_ratio", 1.5);
            nc = j.value("nc", 3);
        };

        PseConfig::PseConfig(){
            create(2240, 1.0);
        };

        PseConfig::PseConfig(int _long_size, float _scale, int _kernel_nums, int _min_kernel_area,
                             int _min_area, float _min_score, float _mag_ratio, int _nc) {
            create(_long_size, _scale, _kernel_nums, _min_kernel_area, _min_area, _min_score, _mag_ratio, _nc);
        };


        PseConfig::PseConfig(const std::string & json_path){
            nlohmann::json j;
            std::ifstream jst(json_path);
            jst >> j;
            create(j);
        };

        PseConfig::PseConfig(const nlohmann::json & j) {
            create(j);
        };


        float PseModel::loadDetectionImage(const cv::Mat & img, at::Tensor & tensor) {
            int max_edge = std::max(img.rows, img.cols);
            float target_size = mag_ratio * max_edge;
            if (target_size > long_size)
                target_size = long_size;
            float ratio = target_size / max_edge;
            int target_w = (int) img.cols * ratio;
            int target_h = (int) img.rows * ratio;
            cv::Size size(target_w, target_h);
            cv::Mat tmpImg(target_h, target_w, CV_32FC3);
            cv::resize(img, tmpImg, size, 0, 0, cv::INTER_AREA);
            int target_h32 = target_h;
            int target_w32 = target_w;
            if (target_h32 % 32 != 0)
                target_h32 = target_h32 + (32 - target_h32 % 32);
            if (target_w32 % 32 != 0)
                target_w32 = target_w32 + (32 - target_w32 % 32);
            tensor = torch::zeros({1, target_h32, target_w32, 3}, torch::kFloat);
            tmpImg.convertTo(tmpImg, CV_32FC3, 1.0f / 255.0f);
            at::Tensor img_tensor = torch::from_blob(tmpImg.data, {target_h, target_w, 3});
            tensor.index_put_({0, Slice(0, target_h), Slice(0, target_w)}, img_tensor);
#ifdef DEBUG
            showTensor(tensor, 3);
#endif
            tensor = torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(
                    tensor.permute({0, 3, 1, 2}));
            return ratio;
        };

        float PseModel::loadDetectionImage(const std::string & img_path, at::Tensor & tensor){
            cv::Mat img;
            img = cv::imread(img_path, cv::IMREAD_COLOR);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            float ratio = loadDetectionImage(img, tensor);
            return ratio;
        };

        void PseModel::predict(const cv::Mat &img, std::vector<sRotatedRect> & rect_vec, bool return_points) {
            assert(rect_vec.size() == 0);
#ifdef DEBUG_TIME
            std::chrono::steady_clock::time_point test_begin = std::chrono::steady_clock::now();
#endif
            if (img.type() == CV_8UC1 and nc == 3){
                cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
            }else if (img.dims == CV_8UC3 and nc == 1){
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }
            at::Tensor input_tensor;
            float ratio = loadDetectionImage(img, input_tensor);
            assert(input_tensor.dim() == 4);
#ifdef DEBUG_TIME
            std::chrono::steady_clock::time_point test_end = std::chrono::steady_clock::now();
            int time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
            showTime(time_cost, "detection preprocess");
#endif
            auto options = at::TensorOptions().device(at::kCUDA).requires_grad(false);
            if (half){
                options = options.dtype(torch::kHalf);
            }
            input_tensor = input_tensor.to(options);
            int or_h = input_tensor.size(2);
            int or_w = input_tensor.size(3);
            torch::Tensor score;
            torch::Tensor kernels;
            float _scale = scale;
            {
#ifdef DEBUG_TIME
                test_begin = std::chrono::steady_clock::now();
#endif
                at::Tensor output;
                torch::NoGradGuard no_grad;
                output = module.forward({input_tensor}).toTensor();
                score = torch::sigmoid(output.index({Slice(), Slice(0, 1), Slice(), Slice()})).detach();
                output = (torch::sign(output - 1) + 1) / 2;
                kernels = output.index({Slice(), 0, Slice(), Slice()});
                kernels = (output.index({Slice(), Slice(0, kernel_nums), Slice(), Slice()}) * kernels).detach();
                if (or_h >= 2000 and or_w >= 1500){
                    auto interpolate_op = F::InterpolateFuncOptions().
                            scale_factor(std::vector<double>({0.5,  0.5})).
                            mode(torch::kArea);
                    score = F::interpolate(score, interpolate_op);
                    kernels = F::interpolate(kernels, interpolate_op);
                    _scale = scale / 0.5;
#ifdef DEBUG
                    printTensor(score, "score");
                    printTensorV(score, "score");
                    printTensor(kernels, "kernels");
                    printTensorV(kernels, "kernels");
#endif
                }
                score = score.cpu().to(at::kFloat);
                kernels = kernels.cpu().to(torch::kU8);
#ifdef DEBUG_TIME
                test_end = std::chrono::steady_clock::now();
                time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
                showTime(time_cost, "detection inference");
#endif
#ifdef DEBUG
                printTensor(output, "detection output");
                printTensor(score, "score");
                printTensor(kernels, "kernels");
#endif
            }
#ifdef DEBUG_TIME
            test_begin = std::chrono::steady_clock::now();
#endif
            assert(kernels.dim() == 4 && score.dim() == 4);
            int height = kernels.size(2);
            int width = kernels.size(3);
            at::Tensor pred_tensor = torch::zeros({height, width}, torch::kInt32);
            int *pPred = pred_tensor.data_ptr<int>();
            auto *pKernel = kernels.data_ptr<unsigned char>();
            float filter_kernel_area = min_kernel_area / std::pow(_scale, 2);
            // get label num;
            opti_pred(pKernel, kernel_nums, height, width, filter_kernel_area, pPred, false);
#ifdef DEBUG_TIME
            test_end = std::chrono::steady_clock::now();
            time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
            showTime(time_cost, "detection get pred label_num");
            test_begin = std::chrono::steady_clock::now();
#endif
            int label_num = torch::max(pred_tensor).item<int>() + 1;

            // debug
#ifdef DEBUG
            std::cout << "pred sum:" << torch::sum(pred_tensor).item<int>() << "\tmax:" << torch::max(pred_tensor).item<int>()
                      << std::endl;
            std::cout << "label_num:" << label_num << std::endl;
            std::cout << "ratio:" << ratio << std::endl;
#endif
            auto *pScore = score.data_ptr<float>();
            opti_pse(pPred, pScore, width, height, label_num, min_area,
                     _scale, min_score, ratio, rect_vec, return_points);
#ifdef DEBUG_TIME
            test_end = std::chrono::steady_clock::now();
            time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
            showTime(time_cost, "detection get rotated rect");
#endif
        }

        void PseModel::predict(const cv::Mat &img, at::Tensor & box_tensor){
            std::vector<sRotatedRect> s_rect_vec;
            predict(img, s_rect_vec);
            get_rect_tensor(s_rect_vec, box_tensor);
#ifdef DEBUG
            std::cout << "box points x mean:" << torch::mean(box_tensor.index({Slice(), Slice(), 0})).item<float>() << std::endl;
            std::cout << "box points y mean:" << torch::mean(box_tensor.index({Slice(), Slice(), 1})).item<float>() << std::endl;
#endif
        }

        at::Tensor PseModel::predict(const cv::Mat &img) {
            std::vector<sRotatedRect> s_rect_vec;
            predict(img, s_rect_vec);
            return get_rect_box_tensor(s_rect_vec);
        }

        void PseModel::predict(const std::string & img_path, at::Tensor & box_tensor){
            cv::Mat img;
            img = cv::imread(img_path, cv::IMREAD_COLOR);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            predict(img, box_tensor);
        }

        at::Tensor PseModel::predict(const std::string & img_path) {
            cv::Mat img;
            img = cv::imread(img_path, cv::IMREAD_COLOR);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            return predict(img);
        }
}

}



