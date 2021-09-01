/**
* @program: dl_core
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-29 14:35
**/
//
// Created by 周炳诚 on 2021/3/29.
//

// 注意这个__TR_CONFIG_H__主要是为了防止import多次的问题
#ifndef __TR_CONFIG_H__
#define __TR_CONFIG_H__

#include <iostream>
#include <codecvt>
#include "base_core.h"

namespace dl_model {

    // text recognition
    namespace tr{

        using namespace dl_model;

        struct CrnnConfig : public BaseConfig {

            int max_height;
            int nc=1;
            int n_class;
            std::wstring vocab;
            std::string padding;
            std::vector<int> width_vec;
            std::vector<int> batch_size_vec;


            void create(int _max_height=48, int _nc=1);

            void loadVocab(const std::string & vocab_path);

            CrnnConfig();

            CrnnConfig(const std::string & vocab_path, int _max_height=48, int _nc=1);

        };

    }
}


#endif //__TR_CONFIG_H__
