/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-30 19:34
**/
//
// Created by 周炳诚 on 2021/3/30.
//
#include <random>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include "ocr.h"

using namespace dl_model;
using namespace pipeline;
using namespace cv;

const char* keys =
        {
                "{help h usage?| | jit_pse_normal.pt jit_crnn_document.pt charset_crnn_document.txt document.jpeg}"
                "{@pse_model_path | pse model path|test pse model json path}"
                "{@crnn_model_path | crnn model path path|test crnn model path}"
                "{@crnn_charset_path | crnn charset path path|test crnn charset path}"
                "{@img_path | test img path | test img path}"
        };


int main(int argc, const char *argv[]) {

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    std::string td_model_path = parser.get<String>(0);
    std::string tr_model_path = parser.get<String>(1);
    std::string vocab_path = parser.get<String>(2);
    std::string img_path = parser.get<String>(3);
//    std::string td_model_path = "/data/duser/models/ysocr_models/detection_models/pse/jit_pse_normal.pt";
//    std::string img_path = "/data/duser/bczhou_test/pse_test_data/3b470dc4-91a3-43a8-addf-3145393c6d7c_page1.jpg";
//    std::string tr_model_path = "/data/duser/models/ysocr_models/recognition_models/document/jit_document.pt";
//    std::string vocab_path = "/data/duser/models/ysocr_models/recognition_models/document/charset_document.txt";
    td::PseModel pse_model(td_model_path, "gpu", 0);
    tr::CrnnModel crnn_model(tr_model_path, vocab_path);

    OcrPipeLine ocr_pipeline1(pse_model, crnn_model);

    OcrPipeLine ocr_pipeline2(td_model_path, tr_model_path, vocab_path);

    Mat img;
    img = imread(img_path, IMREAD_COLOR);
    cvtColor(img, img, COLOR_BGR2RGB);

    std::vector<TextRect> text_rect_vec;
    std::chrono::steady_clock::time_point test_begin = std::chrono::steady_clock::now();
    text_rect_vec = ocr_pipeline1.predict(img);
    std::chrono::steady_clock::time_point test_end = std::chrono::steady_clock::now();
    int time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
    std::cout << "pipe1 img cost:" << time_cost << "[ms]" << std::endl;
    text_rect_vec.clear();
    test_begin = std::chrono::steady_clock::now();
    text_rect_vec = ocr_pipeline1.predict(img_path);
    test_end = std::chrono::steady_clock::now();
    time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
    std::cout << "pipe1 img path cost:" << time_cost << "[ms]" << std::endl;
    text_rect_vec.clear();
    std::cout << "success test ocr" << std::endl;
    
// test for batches
//    std::vector<std::string> file_names;
//    std::string path1 = "/data/duser/bczhou_test/c-inference-test/*.jpg";
//    std::string path2 = "/data/duser/bczhou_test/c-inference-test/*.png";
//    std::string path3 = "/data/duser/bczhou_test/c-inference-test/*.jpeg";
//    get_glob_files(path1, file_names);
//    get_glob_files(path2, file_names);
//    get_glob_files(path3, file_names);
//    get_glob_files(path4, file_names);
//    std::random_device rd;
//    std::mt19937 g(rd());
//    std::shuffle(file_names.begin(), file_names.end(), g);
//
//    int sum_cost = 0;
//    for (int i = 0; i < file_names.size(); ++i) {
//        img_path = file_names.at(i);
//        test_begin = std::chrono::steady_clock::now();
//        ocr_pipeline1.predict(img_path, text_rect_vec);
//        test_end = std::chrono::steady_clock::now();
//        time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
//        std::cout << "ocr pipe img path cost:" << time_cost << "[ms]" << std::endl;
//        sum_cost += time_cost;
//        text_rect_vec.clear();
//        if ((i + 1) % 50 == 0)
//            std::cout << "ocr pipe run " << i + 1 << " times, average cost:" << (float) sum_cost / (i + 1) << "[ms]" << std::endl;
//    }
//    std::cout << "ocr pipe img path cost:" << (float)sum_cost / file_names.size() << "[ms]" << std::endl;
    return 0;
}

