/**
* @program: dl_core
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-28 19:34
**/
//
// Created by 周炳诚 on 2021/3/28.
//
#include <torch/script.h>
#include <opencv2/core/utility.hpp>
#include "td.h"
#include "tr.h"

using namespace cv;

const char* keys =
        {
                "{help h usage?| | jit_pse_normal.pt jit_crnn_document.pt charset_crnn_document.txt}"
                "{@pse_model_path | pse model path|test pse model json path}"
                "{@crnn_model_path | crnn model path path|test crnn model path}"
                "{@crnn_charset_path | crnn charset path path|test crnn charset path}"
        };

int main(int argc, const char *argv[]) {

//    std::string modelPath= "/data/duser/models/ysocr_models/detection_models/pse/jit_pse_normal.pt";
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    std::string modelPath = parser.get<String>(0);

    torch::jit::script::Module test_module = torch::jit::load(modelPath);
    dl_model::BaseModel bm1(modelPath);
    assert(bm1.device.type() == at::kCUDA && bm1.device.type() == at::kCUDA && bm1.device.index() == 0);
    dl_model::BaseModel bm2(test_module);
    assert(bm2.device.type() == at::kCUDA && bm2.device.type() == at::kCUDA && bm2.device.index() == 0);
    dl_model::BaseModel bm3(modelPath, "cpu");
    assert(bm3.device.type() == at::kCPU && bm3.device.type() == at::kCPU);
    dl_model::BaseModel bm4(test_module, "cpu");
    assert(bm4.device.type() == at::kCPU && bm4.device.type() == at::kCPU);
    dl_model::BaseModel bm5(modelPath, "gpu", 0);
    assert(bm5.device.type() == at::kCUDA && bm5.device.type() == at::kCUDA && bm5.device.index() == 0);
    dl_model::BaseModel bm6(test_module, "gpu", 0);
    assert(bm6.device.type() == at::kCUDA && bm6.device.type() == at::kCUDA && bm6.device.index() == 0);

    dl_model::td::PseModel pm1(modelPath, "gpu", 0);
    assert(pm1.device.type() == at::kCUDA && pm1.device.type() == at::kCUDA && pm1.device.index() == 0);
    assert(pm1.long_size == 2240 && pm1.scale == 1 && pm1.kernel_nums == 4 &&  pm1.min_kernel_area == 10
        && pm1.min_area == 10 && pm1.min_score == (float) 0.8 && pm1.mag_ratio == (float) 1.5 && pm1.half == false);
    dl_model::td::PseModel pm2;
    pm2 = pm1;
    assert(pm2.long_size == 2240 && pm2.scale == 1 && pm2.kernel_nums == 4 &&  pm2.min_kernel_area == 10
           && pm2.min_area == 10 && pm2.min_score == (float) 0.8 && pm2.mag_ratio == (float) 1.5 && pm2.half == false);

    dl_model::td::PseModel pm3(modelPath, "gpu", 0, true);
    assert(pm3.half);
//    const std::string crnn_model_path= "/data/duser/models/ysocr_models/recognition_models/document/jit_document.pt";
//    const std::string vocab_path = "/data/duser/models/ysocr_models/recognition_models/document/charset_document.txt";
    const std::string crnn_model_path = parser.get<String>(1);
    const std::string vocab_path = parser.get<String>(2);
    dl_model::tr::CrnnModel cm1(crnn_model_path, vocab_path);
    assert(cm1.n_class == 7500 && cm1.max_height == 48 && cm1.padding == "adaptive_soft_padding");
    assert(cm1.vocab[1] == 916 && cm1.width_vec[0] == 200 && cm1.batch_size_vec[0] == 128);

    dl_model::tr::CrnnModel cm2;
    cm2 = cm1;
    assert(cm2.n_class == 7500 && cm2.max_height == 48 && cm2.padding == "adaptive_soft_padding");
    assert(cm2.vocab[1] == 916 && cm2.width_vec[0] == 200 && cm2.batch_size_vec[0] == 128);
    std::cout << "success test load model" << std::endl;
    return 0;
}
