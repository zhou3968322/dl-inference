//
// Created by 周炳诚 on 2021/3/26.
//

#ifndef __TR_H__
#define __TR_H__

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "base_core.h"
#include "tr_config.h"




namespace dl_model{
    namespace tr{

        using namespace dl_model;
        struct stImgBatch{

            std::vector<cv::Mat> imgs;
            std::vector<cv::Rect> rects;
            int max_width;
            int batch_size;
            int width;
            int height;

            stImgBatch();

            stImgBatch(int i_max_width, int i_batch_size, int i_height=48, int i_width=0);

            stImgBatch(const stImgBatch & img_batch);

            stImgBatch operator= (const stImgBatch & img_batch);

            int size();

            void push_img(cv::Mat & img, cv::Rect & rect);

            at::Tensor convert_to_tensor();

            ~stImgBatch(){

                imgs.clear();
            };
        };

        struct CrnnModel: public BaseModel, public CrnnConfig{
            using BaseModel::BaseModel;
            using CrnnConfig::CrnnConfig;

            CrnnModel():BaseModel(),CrnnConfig(){
                mt = CRNN;
            };

            CrnnModel(const std::string & model_path, const std::string & vocab_path, int _max_height=48, int _nc=1,
                      const std::string & mode = "gpu", int index = DEFAULT_DEVICE_ID):
                      BaseModel(model_path, mode, index),CrnnConfig(vocab_path, _max_height, _nc){
                mt = CRNN;
            };

            CrnnModel(const std::string & model_path, const std::string & vocab_path, const std::string & mode, int index):
                    BaseModel(model_path, mode, index),CrnnConfig(vocab_path, 48, 1){
                mt = CRNN;
            };

            CrnnModel(torch::jit::script::Module & _module, const std::string & vocab_path, int _max_height=48, int _nc=1,
                      const std::string & mode = "gpu", int index = DEFAULT_DEVICE_ID):
                    BaseModel(_module, mode, index),CrnnConfig(vocab_path, _max_height, _nc){
                mt = CRNN;
            };

            CrnnModel(torch::jit::script::Module & _module, const std::string & vocab_path, const std::string & mode, int index):
                    BaseModel(_module, mode, index),CrnnConfig(vocab_path, 48, 1){
                mt = CRNN;
            };

            void loadImage(const std::string & img_path, cv::Mat & img);

            inline void getImgBatch(const cv::Mat & img, std::vector<cv::Rect> & rect_vec, std::vector<stImgBatch> & img_batches);

            std::vector<stImgBatch> getImgBatch(const cv::Mat & img, std::vector<cv::Rect> & rect_vec);

            std::wstring index2char(at::Tensor & index_tensor);

            std::wstring decodeText(at::Tensor & output);

            inline void decodeBatchText(at::Tensor & batch_tensor, std::vector<std::wstring> & texts);

            std::vector<std::wstring> decodeBatchText(at::Tensor & batch_tensor);

            // rect_vec会变化
            void predict(cv::Mat & img, std::vector<cv::Rect> & rect_vec, std::vector<TextRect> & text_rect_vec);

            void predict(cv::Mat & img, const std::vector<std::vector<int>> & text_coors, std::vector<TextRect> & text_rect_vec);

            void predict(cv::Mat & img, const nlohmann::json & j, std::vector<TextRect> & text_rect_vec);

            void predict(cv::Mat & img, const std::string & json_path, std::vector<TextRect> & text_rect_vec);

            void predict(const std::string & img_path, std::vector<cv::Rect> & rect_vec, std::vector<TextRect> & text_rect_vec);

            void predict(const std::string & img_path, const std::vector<std::vector<int>> & text_coors, std::vector<TextRect> & text_rect_vec);

            void predict(const std::string & img_path, const nlohmann::json & j, std::vector<TextRect> & text_rect_vec);

            void predict(const std::string & img_path, const std::string & json_path, std::vector<TextRect> & text_rect_vec);

        };


    }
}


#endif //__TR_H__
