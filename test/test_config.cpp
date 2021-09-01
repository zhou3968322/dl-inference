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
* @create: 2021-03-24 17:45
**/
//
// Created by 周炳诚 on 2021/3/24.
//

#include <unordered_map>
#include <opencv2/core/utility.hpp>
#include "td_config.h"
#include "tr_config.h"


//#include <type_traits>
//using common_type_fi_t = typename std::common_type<int, float>::type ;
using namespace dl_model;
using namespace cv;


const char* keys =
        {
                "{help h usage?| | test/data/pse_config.json test/data/charset_document.txt}"
                "{@pse_config | json path|test pse config json path}"
                "{@charset_path | crnn charset path|test crnn charset path}"
        };

int main(int argc, const char *argv[]) {
    // ./test_config test/data/pse_config.json
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    td::PseConfig config0;
    assert(config0.long_size == 2240);
    assert(config0.scale == 1);
    assert(config0.kernel_nums == 4);
    assert(config0.min_kernel_area == 10);
    assert(config0.min_area == 10);
    assert(config0.min_score == (float) 0.8);
    assert(config0.mag_ratio == (float) 1.5);

    std::string config_path = parser.get<String>(0);
    td::PseConfig config1(config_path);
    assert(config1.long_size == 2240);
    assert(config1.scale == 1);
    assert(config1.kernel_nums == 4);
    assert(config1.min_kernel_area == 10);
    assert(config1.min_score == (float) 0.8);

    nlohmann::json j = {{"long_size", 4480},
                        {"scale", 1},
                        {"min_score", 0.7},
                        {"min_kernel_area", 20},
                        {"min_area", 20},
                        {"mag_ratio", 1.5}};
    td::PseConfig config2(j);
    assert(config2.long_size == 4480);
    assert(config2.scale == 1);
    assert(config2.kernel_nums == 4);
    assert(config2.min_score == (float) 0.7);
    assert(config2.min_kernel_area == 20);
    assert(config2.min_area == 20);
    assert(config2.mag_ratio == (float) 1.5);
    ;
    td::PseConfig config3(4480, 1, 3, 30, 30, 0.7, 1.5);
    assert(config3.long_size == 4480);
    assert(config3.scale == 1);
    assert(config3.kernel_nums == 3);
    assert(config3.min_kernel_area == 30);
    assert(config3.min_area == 30);
    assert(config3.min_score == (float) 0.7);
    assert(config3.mag_ratio == (float) 1.5);

    std::string vocab_path = parser.get<String>(1);;
    tr::CrnnConfig crnn_config1(vocab_path, 48);
    assert(crnn_config1.n_class == 7500 && crnn_config1.max_height == 48 && crnn_config1.padding == "adaptive_soft_padding");
    assert(crnn_config1.vocab[1] == 916 && crnn_config1.width_vec[0] == 200 && crnn_config1.batch_size_vec[0] == 128);
    std::cout << "success test config" << std::endl;
    return 0;
}


