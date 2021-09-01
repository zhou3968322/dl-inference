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
#include <opencv2/core/utility.hpp>
#include <nlohmann/json.hpp>
#include "tr.h"

using namespace cv;


void loadJson(std::string & jsonPath, nlohmann::json & j){
    std::ifstream i(jsonPath);
    i >> j;
}


const char* keys =
        {
                "{help h usage?| | jit_crnn_document.pt  document1.json document1.png}"
                "{@crnn_model_path | crnn model path|test crnn model model path}"
                "{@vocab_path | crnn vocab path|test crnn vocab model path}"
                "{@json_path | box info json path | box info json path}"
                "{@img_path | test img path | test img path}"
        };

int main(int argc, const char *argv[]){

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    std::string modelPath = parser.get<String>(0);
    std::string vocabPath = parser.get<String>(1);
    std::string jsonPath = parser.get<String>(2);
    std::string fullImgPath = parser.get<String>(3);
    //    std::string modelPath = "/data/duser/models/ysocr_models/recognition_models/document/jit_document.pt";
//    std::string vocabPath = "/data/duser/models/ysocr_models/recognition_models/document/charset_document.txt";
//    std::string jsonPath = "/data/duser/bczhou_test/img_with_npys/0.json";
//    std::string fullImgPath = "/data/duser/bczhou_test/img_with_npys/0.png";

    nlohmann::json j;
    loadJson(jsonPath, j);
    std::vector<std::vector<int>> rectArray = j.get<std::vector<std::vector<int>>>();
    dl_model::tr::CrnnModel crnn_model(modelPath, vocabPath);
    Mat img;
    img = imread(fullImgPath, IMREAD_GRAYSCALE);
    std::vector<Rect> rect_vec;
    for (int i = 0; i < rectArray.size(); ++i) {
        Rect rect(rectArray[i][0], rectArray[i][1],
                      rectArray[i][2] - rectArray[i][0],
                      rectArray[i][3] - rectArray[i][1]);
        rect_vec.push_back(rect);
    }
    std::vector<TextRect> text_rect_vec;
    std::vector<Rect> i_rect_vec1(rect_vec);
    crnn_model.predict(img, i_rect_vec1, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(img, rectArray, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(img, j, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(img, jsonPath, text_rect_vec);
    text_rect_vec.clear();
    std::vector<Rect> i_rect_vec2(rect_vec);
    crnn_model.predict(fullImgPath, i_rect_vec2, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(fullImgPath, rectArray, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(img, j, text_rect_vec);
    text_rect_vec.clear();
    crnn_model.predict(img, jsonPath, text_rect_vec);
    text_rect_vec.clear();
    std::cout << "success test text recognition" << std::endl;
    return 0;
}

