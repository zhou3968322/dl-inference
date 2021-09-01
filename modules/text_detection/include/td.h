//
// Created by 周炳诚 on 2021/3/26.
//

#ifndef __TD_H__
#define __TD_H__

#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include "util.h"
#include "td_config.h"



namespace dl_model {
    // text detection
    namespace td {

        using namespace dl_model;

        struct PseModel: public BaseModel, public PseConfig {

            using BaseModel::BaseModel;
            using PseConfig::PseConfig;

            PseModel():BaseModel(),PseConfig(){
              mt = PSE;
            };

            PseModel(const std::string & model_path, const std::string & mode="gpu", int index=DEFAULT_DEVICE_ID, bool _half=false)
            :BaseModel(model_path, mode, index, _half),
            PseConfig(){
                mt = PSE;
            };

            PseModel(torch::jit::script::Module & _module, const std::string & mode="gpu", int index=DEFAULT_DEVICE_ID, bool _half=false)
                    :BaseModel(_module, mode, index, _half),
                     PseConfig(){
                mt = PSE;
            };

            /*!
             *
             * @param img  rgb img
             * @param tensor output tensor;
             * @return ratio for post process;
             */
            float loadDetectionImage(const cv::Mat & img, at::Tensor & tensor);

            float loadDetectionImage(const std::string & img_path, at::Tensor & tensor);

            void predict(const cv::Mat &img, std::vector<sRotatedRect> & rect_vec,bool return_points=false);

            void predict(const cv::Mat &img, at::Tensor & box_tensor);

            at::Tensor predict(const cv::Mat &img);

            void predict(const std::string & img_path, at::Tensor & box_tensor);

            at::Tensor predict(const std::string & img_path);
        };
    }
}


#endif //__TD_H__

