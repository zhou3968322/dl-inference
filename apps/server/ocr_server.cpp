/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-20 10:29
**/
//
// Created by 周炳诚 on 2021/5/20.
//

#include <opencv2/core/utility.hpp>
#include <nlohmann/json.hpp>
#include "boost/filesystem.hpp"
#include "ocr.h"
#include "base64.h"
#include "crow.h"
#include "crow_log.h"


#ifndef DATA_ROOT
#define DATA_ROOT "/usr/local/cpp_libs/inference/data/"
#endif

using namespace cv;
using namespace dl_model;
using namespace pipeline;

const char* keys =
        {
                "{ help h| | jit_pse_normal.pt jit_crnn_document.pt charset_crnn_document.txt 51000}"
                "{ pse_model_path pmp| jit_pse_document.pt |test pse model json path}"
                "{ crnn_model_path cmp | jit_crnn_document_v1.pt |test crnn model path}"
                "{ crnn_charset_path | charset_document_v1.txt |test crnn charset path}"
                "{ port | 51000 | server port}"
                "{ device | 0 | gpu index}"
        };

int main(int argc, const char *argv[]) {

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string td_model_path = parser.get<String>("pse_model_path");
    std::string tr_model_path = parser.get<String>("crnn_model_path");
    std::string vocab_path = parser.get<String>("crnn_charset_path");
    if (! boost::filesystem::is_regular_file(td_model_path)){
        td_model_path = "online_models/" + td_model_path;
        td_model_path = DATA_ROOT + td_model_path;
    }
    if (! boost::filesystem::is_regular_file(tr_model_path)){
        tr_model_path = "online_models/" + tr_model_path;
        tr_model_path = DATA_ROOT + tr_model_path;
    }
    if (! boost::filesystem::is_regular_file(vocab_path)){
        vocab_path = "online_models/" + vocab_path;
        vocab_path = DATA_ROOT + vocab_path;
    }
    static int port = parser.get<int>("port");
    static int device_index = parser.get<int>("device");
    torch::jit::script::Module td_module = torch::jit::load(td_model_path);
    td::PseModel pm(td_module, "gpu", device_index, true);
    torch::jit::script::Module tr_module = torch::jit::load(tr_model_path);
    tr::CrnnModel cm(tr_module, vocab_path, "gpu", device_index);
    OcrPipeLine ocr_pipe(pm, cm);
    std::cout << "server starting" << std::endl;

    //App
    crow::SimpleApp app;
    CROW_ROUTE(app, "/predict").methods("POST"_method, "GET"_method)([& ocr_pipe](const crow::request & req){
        CROW_LOG_INFO << "Received Request";
        nlohmann::json result = {{"msg", "Failed"}, {"code", 500}};
        std::ostringstream os;
        try{
            //Get Image
            auto args = crow::json::load(req.body);

            // Get Image
            std::string base64_image = args["image"].s();
            std::string decoded_image = base64_decode(base64_image);
            std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
            cv::Mat img = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

            //predict
            std::vector<TextRect> text_rect_vec;
            text_rect_vec = ocr_pipe.predict(img);
            result["msg"] = "Success";
            result["code"] = 200;
            nlohmann::json res_boxes;
            size_t box_size = text_rect_vec.size();
            for (int i = 0; i < box_size; ++i) {
                res_boxes.push_back({
                    {"x", text_rect_vec.at(i).x},
                    {"y", text_rect_vec.at(i).y},
                    {"width", text_rect_vec.at(i).width},
                    {"height", text_rect_vec.at(i).height},
                    {"text", text_rect_vec.at(i).text}
                });
            }
            result["box_list"] = res_boxes;
            text_rect_vec.clear();
            os << result.dump(-1, ' ', true);
            CROW_LOG_INFO << "Success Process Request,box_nums:" << box_size;
            return crow::response(200, os.str());
        } catch (std::exception& e) {
            result = {{"msg", "Failed"}, {"code", 500}};
            os << result.dump(-1, ' ', true);
            CROW_LOG_INFO << "Failed Process Request";
            return crow::response(500, os.str());
        }

    }
    );
    app.port(port).multithreaded().run();
    return 0;
}


