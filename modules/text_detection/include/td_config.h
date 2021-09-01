/**
* @program: ocr
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-26 15:25
**/
//
// Created by 周炳诚 on 2021/3/26.
//

#ifndef __TD_CONFIG_H__
#define __TD_CONFIG_H__

#include <nlohmann/json.hpp>
#include <iostream>
#include "base_core.h"


namespace dl_model {

    // text detection
    namespace td{
        using namespace dl_model;
        struct PseConfig : public BaseConfig {

            int long_size;
            float scale;
            int kernel_nums;
            int min_kernel_area;
            int min_area;
            float min_score;
            float mag_ratio;
            int nc;


            inline void create(int _long_size, float _scale, int _kernel_nums=4, int _min_kernel_area=10,
                                int _min_area=10, float _min_score=0.8, float _mag_ratio=1.5, int _nc=3);

            inline void create(const nlohmann::json & j);

            PseConfig();

            /*!
             * 参数初始化
             * @param _long_size 长度
             * @param _scale 缩放比例
             * @param _kernel_nums 使用的kernel数
             * @param _min_kernel_area 最小kernel面积
             * @param _min_area 最小box面积
             * @param _min_score pixel score阈值
             */
            PseConfig(int _long_size, float _scale, int _kernel_nums=4, int _min_kernel_area=10,
                      int _min_area=10, float _min_score=0.8, float _mag_ratio=1.5, int _nc=3);


            /*!
             * 通过json初始化
             * @param j nlohmann json format
             */
            PseConfig(const nlohmann::json & j);

            /*!
             * 通过json路径初始化
             * @param json_path
             */
            PseConfig(const std::string & json_path);

        };

    }
}

#endif //__TD_CONFIG_H__
