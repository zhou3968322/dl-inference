/**
* @program: ocr
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-26 11:09
**/
//
// Created by 周炳诚 on 2021/3/26.
//
# include "util.h"
# include "tr.h"


namespace dl_model {

    // text recognition
    namespace tr {
        using namespace dl_model;
        using namespace torch::indexing;

        inline void CrnnConfig::create(int _max_height, int _nc) {
            nc = _nc;
            max_height = _max_height;
            padding = "adaptive_soft_padding";
            width_vec = {200, 400, 800, 1600, 3200, 10000};
            batch_size_vec = {256, 128, 64, 32, 16, 4};
        }

        inline void CrnnConfig::loadVocab(const std::string & vocab_path) {
            std::wifstream vocFile(vocab_path);
            vocFile.imbue(std::locale(vocFile.getloc(),
                                      new std::codecvt_utf8<wchar_t, 0x10ffff, std::consume_header>()));
            std::wstringstream wss;
            wss << vocFile.rdbuf();
            vocab = wss.str();
            vocFile.close();
            n_class = vocab.size() + 1;
        }

        CrnnConfig::CrnnConfig(){
            vocab = L"0123456789abcdefghijklmnopqrstuvwxyz";
            n_class = vocab.size() + 1;
            create();
        }

        CrnnConfig::CrnnConfig(const std::string & vocab_path, int _max_height, int _nc){
            create(_max_height, _nc);
            loadVocab(vocab_path);
        }

        stImgBatch::stImgBatch() {
            max_width=200;
            batch_size=64;
            width=0;
            height=48;
        }

        stImgBatch::stImgBatch(int i_max_width, int i_batch_size, int i_height, int i_width){
            width = i_width;
            batch_size = i_batch_size;
            max_width = i_max_width;
            height = i_height;
        }

        stImgBatch::stImgBatch(const stImgBatch & img_batch){
            imgs = img_batch.imgs;
            max_width = img_batch.max_width;
            batch_size = img_batch.batch_size;
            width = img_batch.width;
            height = img_batch.height;
            rects = img_batch.rects;
        }

        stImgBatch stImgBatch::operator= (const stImgBatch & img_batch){
            imgs = img_batch.imgs;
            max_width = img_batch.max_width;
            batch_size = img_batch.batch_size;
            width = img_batch.width;
            height = img_batch.height;
            rects = img_batch.rects;
            return *this;
        }

        int stImgBatch::size(){
            return imgs.size();
        }

        void stImgBatch::push_img(cv::Mat & img, cv::Rect & rect){
            imgs.push_back(img);
            rects.push_back(rect);
            width = std::max(width, (int)(img.cols / ((double) img.rows) * height));
            assert(imgs.size() <= batch_size);
        }

        at::Tensor stImgBatch::convert_to_tensor(){
            int batch = imgs.size();
            auto tensor = torch::full({batch, height, width, 1}, 1.0, torch::kFloat);

            for (int i = 0; i < imgs.size(); ++i) {
                int c_width = (int) ((double) height / imgs.at(i).rows * imgs.at(i).cols);
                cv::Size size(c_width, height);
                cv::resize(imgs.at(i), imgs.at(i), size,  0, 0,cv::INTER_AREA);
#ifdef DEBUG
                cv::imwrite("./debug.jpeg", imgs.at(i));
#endif
                imgs.at(i).convertTo(imgs.at(i), CV_32FC3, 1.0f / 255.0f);
                auto img_tensor = torch::from_blob(imgs.at(i).data, {height, c_width});
#ifdef DEBUG
                showTensor(img_tensor);
#endif
                tensor.index_put_({i, Slice() , Slice(0, c_width), 0}, img_tensor);
#ifdef DEBUG
                at::Tensor test_tensor = tensor.index({i, Slice() , Slice(0, c_width)});
                showTensor(test_tensor);
#endif

            }
#ifdef DEBUG
            for (int i = 0; i < imgs.size(); ++i) {
                auto img_tensor = tensor[i].clone();
                showTensor(img_tensor);
            }
#endif
            tensor = tensor.permute({0, 3, 1, 2}).sub_(0.5).div_(0.5);
            return tensor;
        }

        void CrnnModel::loadImage(const std::string & img_path, cv::Mat & img){
            if (nc == 1){
                img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            } else{
                img = cv::imread(img_path, cv::IMREAD_COLOR);
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }
        };

        inline void CrnnModel::getImgBatch(const cv::Mat & img, std::vector<cv::Rect> & rect_vec,
                                           std::vector<stImgBatch> & img_batches){
            // initialize
            std::vector<stImgBatch> d_img_batches; // default img batches
            for (int i = 0; i < width_vec.size(); ++i) {
                d_img_batches.push_back(stImgBatch(width_vec[i], batch_size_vec[i]));
            }

            std::sort(rect_vec.begin(), rect_vec.end(), rect_sort_by_ratio);

            for (int i = 0; i < rect_vec.size(); ++i){
                double ratio = rect_vec[i].width / ((double)rect_vec[i].height);
                cv::Mat box_img = img(rect_vec[i]);
                for (int j = 0; j < width_vec.size(); ++j) {
                    if (d_img_batches[j].size() == d_img_batches[j].batch_size){
                        img_batches.push_back(stImgBatch(d_img_batches[j]));
                        d_img_batches.at(j) = stImgBatch(width_vec[j], batch_size_vec[j], max_height);
                    }
                    if (j >= 1 && d_img_batches[j - 1].size() > 0){
                        img_batches.push_back(stImgBatch(d_img_batches[j - 1]));
                        d_img_batches.at(j - 1) = stImgBatch(width_vec[j], batch_size_vec[j], max_height);
                    }
                    if (ratio <= d_img_batches[j].max_width / max_height){
                        d_img_batches[j].push_img(box_img, rect_vec.at(i));
                        break;
                    }
                }
            }

            for (int i = 0; i < width_vec.size(); ++i) {
                if (d_img_batches[i].size() > 0){
                    img_batches.push_back(stImgBatch(d_img_batches[i]));
                    d_img_batches.at(i) = stImgBatch(width_vec[i], batch_size_vec[i], max_height);
                }
            }
        }

        std::vector<stImgBatch> CrnnModel::getImgBatch(const cv::Mat & img, std::vector<cv::Rect> & rect_vec){
            std::vector<stImgBatch> img_batches;
            getImgBatch(img, rect_vec, img_batches);
            return img_batches;
        };

        std::wstring CrnnModel::index2char(at::Tensor &index_tensor) {
            std::wstring text = L"";
            for (int i = 0; i < index_tensor.size(0); ++i) {
                auto idx = index_tensor[i].item<int>();
                text += vocab[idx - 1];
            }
            return text;
        }

        std::wstring CrnnModel::decodeText(at::Tensor & output){
            assert(output.dim() == 1);
            std::wstring text;
            for (int i = 0; i < output.size(0); ++i) {
                auto idx = output[i].item<int>();
                int prev_idx = 0;
                if (i > 0){
                    prev_idx = output[i - 1].item<int>();
                }
                if (idx > 0 && idx != prev_idx) {
                    text += vocab[idx - 1];
                }
            }
            return text;
        };

        inline void CrnnModel::decodeBatchText(at::Tensor & batch_tensor, std::vector<std::wstring> & texts){
            assert(batch_tensor.dim() == 2);
            for (int j = 0; j < batch_tensor.size(0); ++j) {
                at::Tensor slice_tensor = batch_tensor.index({j, Slice()});
                texts.push_back(decodeText(slice_tensor));
            }
        };

        std::vector<std::wstring> CrnnModel::decodeBatchText(at::Tensor & batch_tensor){
            std::vector<std::wstring> texts;
            decodeBatchText(batch_tensor, texts);
            return texts;
        };

        void CrnnModel::predict(cv::Mat & img, std::vector<cv::Rect> & rect_vec, std::vector<TextRect> & text_rect_vec){
#ifdef DEBUG_TIME
            std::chrono::steady_clock::time_point test_begin = std::chrono::steady_clock::now();
#endif
            if (nc == 1 and img.type() == CV_8UC3){
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            } else if (nc == 3 and img.dims == CV_8UC1){
                cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
            }
            std::vector<stImgBatch> img_batches;
            getImgBatch(img, rect_vec, img_batches);
#ifdef DEBUG_TIME
            std::chrono::steady_clock::time_point test_end = std::chrono::steady_clock::now();
            int time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
            showTime(time_cost, "recognition preprocess");
#endif
            at::Tensor out_tensor;
            std::vector<std::wstring> texts;
#ifdef DEBUG_TIME
            test_begin = std::chrono::steady_clock::now();
            int inference_cost = 0;
            int decode_cost = 0;
            int load_data_cost = 0;
#endif
            for (int i = 0; i < img_batches.size(); ++i) {
                torch::NoGradGuard no_grad;
#ifdef DEBUG_TIME
                std::chrono::steady_clock::time_point load_data_begin = std::chrono::steady_clock::now();
#endif
                at::Tensor input_tensor = img_batches.at(i).convert_to_tensor();
#ifdef DEBUG_TIME
                std::chrono::steady_clock::time_point load_data_end = std::chrono::steady_clock::now();
                load_data_cost += std::chrono::duration_cast<std::chrono::milliseconds>(load_data_end - load_data_begin).count();
                std::chrono::steady_clock::time_point inference_begin = std::chrono::steady_clock::now();
#endif
                input_tensor = input_tensor.to(device);
                out_tensor = module.forward({input_tensor}).toTensor();
                at::Tensor result = std::get<1>(out_tensor.max(2)).transpose(1, 0).contiguous().cpu();
                at::Tensor res_tensor1 =  result.index({Slice(), Slice(0, 1)}).not_equal(0).detach().cpu();
                at::Tensor out_index_tensor;
                if (result.size(1) == 1){
                    out_index_tensor = res_tensor1;
                }else{
                    at::Tensor res_tensor2 = torch::logical_and(result.index({Slice(), Slice(1)}).
                            not_equal(result.index({Slice(), Slice(0, -1)})),
                            result.index({Slice(), Slice(1)}).not_equal(0)).detach().cpu();
                    out_index_tensor = torch::cat({res_tensor1, res_tensor2}, 1);
                }
                at::Tensor out_length = torch::sum(out_index_tensor, 1);
                at::Tensor out_res_tensor = result.index({out_index_tensor});
#ifdef DEBUG_TIME
//                std::cout << "out_tensor device:" << out_tensor.device() << std::endl;
                std::chrono::steady_clock::time_point inference_end = std::chrono::steady_clock::now();
                inference_cost += std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_begin).count();
#endif
#ifdef DEBUG_TIME
                std::chrono::steady_clock::time_point decode_begin = std::chrono::steady_clock::now();
#endif
//                decodeBatchText(result, texts);
#ifdef DEBUG_TIME
                std::chrono::steady_clock::time_point decode_end = std::chrono::steady_clock::now();
                decode_cost += std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_begin).count();
#endif
                for (int j = 0; j < img_batches.at(i).size(); ++j) {
                    int bi = torch::sum(out_length.index({Slice(0, j)})).item<int>();
                    int ei = torch::sum(out_length.index({Slice(0, j + 1)})).item<int>();
                    auto res_index_tensor = out_res_tensor.index({Slice(bi, ei)});
                    std::wstring text = index2char(res_index_tensor);
                    text_rect_vec.push_back(TextRect(img_batches.at(i).rects.at(j), text));
                }
                texts.clear();
            }
#ifdef DEBUG_TIME
            test_end = std::chrono::steady_clock::now();
            time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_begin).count();
            std::cout << "current recognition batch_cout:" << img_batches.size() << std::endl;
            showTime(load_data_cost, "load data");
            showTime(inference_cost, "recognition inference");
            showTime(decode_cost, "recognition decode");
            showTime(time_cost, "recognition inference and decode");

#endif
#ifdef DEBUG
            std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> convert;
            for (int i = 0; i < text_rect_vec.size(); ++i) {
                std::cout << "text:" << convert.to_bytes(text_rect_vec.at(i).text) <<
                " rect:" << rect_vec.at(i) << std::endl;
            }
#endif
        };

        void CrnnModel::predict(cv::Mat & img, const std::vector<std::vector<int>> & text_coors,
                                std::vector<TextRect> & text_rect_vec){
            std::vector<cv::Rect> rect_vec;
            for (int i = 0; i < text_coors.size(); ++i) {
                cv::Rect rect(text_coors[i][0], text_coors[i][1],
                              text_coors[i][2] - text_coors[i][0],
                              text_coors[i][3] - text_coors[i][1]);
                rect_vec.push_back(rect);
            }
            assert(text_rect_vec.size() == 0);
            predict(img, rect_vec, text_rect_vec);
        };

        void CrnnModel::predict(cv::Mat & img, const nlohmann::json & j, std::vector<TextRect> & text_rect_vec){
            std::vector<std::vector<int>> text_coors = j.get<std::vector<std::vector<int>>>();
            predict(img, text_coors, text_rect_vec);
        };

        void CrnnModel::predict(cv::Mat & img, const std::string & json_path, std::vector<TextRect> & text_rect_vec){
            std::ifstream ifs(json_path);
            nlohmann::json j;
            ifs >> j;
            ifs.close();
            predict(img, j, text_rect_vec);
        };

        void CrnnModel::predict(const std::string & img_path, std::vector<cv::Rect> & rect_vec,
                                std::vector<TextRect> & text_rect_vec){
            cv::Mat img;
            loadImage(img_path, img);
            predict(img, rect_vec, text_rect_vec);

        };

        void CrnnModel::predict(const std::string & img_path, const std::vector<std::vector<int>> & text_coors,
                                std::vector<TextRect> & text_rect_vec){
            cv::Mat img;
            loadImage(img_path, img);
            predict(img, text_coors, text_rect_vec);
        };

        void CrnnModel::predict(const std::string & img_path, const nlohmann::json & j,
                                std::vector<TextRect> & text_rect_vec){
            cv::Mat img;
            loadImage(img_path, img);
            predict(img, j, text_rect_vec);
        };

        void CrnnModel::predict(const std::string & img_path, const std::string & json_path,
                                std::vector<TextRect> & text_rect_vec){
            cv::Mat img;
            loadImage(img_path, img);
            predict(img, json_path, text_rect_vec);
        };
    }
}
