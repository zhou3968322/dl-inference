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
* @create: 2021-03-23 18:55
**/
//
// Created by 周炳诚 on 2021/3/23.
//



//#include <iostream>
//#include <algorithm>
//#include <opencv2/opencv.hpp>
//#include <torch/script.h>

//int main()
//{
//    try
//    {
//        torch::jit::script::Module model = torch::jit::load("traced_facelearner_model_new.pt");
//        model.to(torch::kCUDA);
//
//        cv::Mat visibleFrame = cv::imread("example.jpg");
//
//        cv::resize(visibleFrame, visibleFrame, cv::Size(112, 112));
//
//# slice images
//        cv::Mat subImg = visibleFrame(cv::Range(0, 100), cv::Range(0, 100));
//
//        at::Tensor tensor_image = torch::from_blob(visibleFrame.data, {  visibleFrame.rows,
//                                                                         visibleFrame.cols, 3 }, at::kByte);
//
//        tensor_image = tensor_image.to(at::kFloat).div(255).unsqueeze(0);
//        tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
//        ensor_image.sub_(0.5).div_(0.5);
//        cout << tensor[0][0][0].item<float>() << endl;
//        cout << torch::max(tensor).item<int>() << endl;
//        cout << torch::sum(tensor).item<int>() << endl;
//
//        //torch mean can only calculate  tensor with dtype float
//        cout << torch::mean(tensor).item<float>() << endl;
//        cout << tensor
//        tensor_image = tensor_image.to(torch::kCUDA);
//        // Execute the model and turn its output into a tensor.
//        auto output = model.forward({tensor_image}).toTensor();
////  有多个结果需要转tuple，默认返回类型是torch::jit::IValue
////        auto outputs = module->forward(inputs).toTuple();
////        torch::Tensor out1 = outputs->elements()[0].toTensor();
////        torch::Tensor out2 = outputs->elements()[1].toTensor();
//        output = output.cpu();
//        std::cout << "Embds: " << output << std::endl;
//
//        std::cout << "Done!\n";
//    }
//    catch (std::exception e)
//    {
//        std::cout << "exception" << e.what() << std::endl;
//    }



//    auto p = p_rotated_rect.begin();
//    int point_index = 0;
//    int sum_rects = p_rotated_rect.size();
//    while (p_rotated_rect.size() > 0 && p != p_rotated_rect.end()){
//        if ((sum_rects - p_rotated_rect.size()) == 10){
//            std::cout << "debugging" << std::endl;
//        }
//        int pre_i= point_index;
//        int next_i = point_index;
//        if (std::abs(p->angle) < 45){
//            pre_i = point_index + 1;
//            next_i = std::min(point_index + search_nums + 1, (int) p_rotated_rect.size());
//        }else{
//            pre_i = std::max(point_index - search_nums, 0);
//            next_i = point_index + search_nums + 1;
//        }
//        debug_img_rect(img, *p, true, "rotated_rect.jpeg");
//        std::vector<float> proj_d_vec;
//        std::vector<int> ppi_vec;
//        for (int pi = pre_i; pi < next_i; ++pi){
//            if (pi != point_index){
//                float proj_d = p->proj_dist(*(p + pi - point_index));
//                std::vector<sRotatedRect> debug_rotated_rects;
//                debug_rotated_rects.push_back(*(p + pi - point_index));
//                debug_rotated_rects.push_back(*(p));
//                debug_img_rect_line(img, debug_rotated_rects, true, "debug_dist_rect.jpeg");
//                proj_d_vec.push_back(proj_d);
//                ppi_vec.push_back(pi);
//            }
//        }
//        float half_min_side = std::min(p->size.height, p->size.width) / 2;
//        if (proj_d_vec.size() > 0){
//            auto min_proj_p = std::min_element(proj_d_vec.begin(), proj_d_vec.end());
//            int min_proj_p_index = min_proj_p - proj_d_vec.begin();
//            if (*min_proj_p < half_min_side){
//                line_rotated_rect.push_back(*p);
//                debug_img_rect_line(img, line_rotated_rect, true, "line_rect.jpeg");
//                s_rotated_rect.erase(p);
//                int tmp_point_index = ppi_vec[min_proj_p_index];
//                if (tmp_point_index > point_index){
//                    point_index = tmp_point_index - 1;
//                }else{
//                    point_index = tmp_point_index;
//                }
//                p = s_rotated_rect.begin() + point_index;
//            }else{
//                line_rotated_rect.push_back(*p);
//                s_rotated_rect.erase(p);
//                line_rotated_rects.push_back(line_rotated_rect);
//                debug_img_rect_line(img, line_rotated_rect, true, "line_rect_res.jpeg");
//                line_rotated_rect.clear();
//                p = s_rotated_rect.begin();
//                point_index = 0;
//            }
//        }else{
//            break;
//        }
//    }
//    std::memcpy((void *) t_m.data, X.data_ptr(), sizeof(torch::kFloat)*X.numel());
//    double m[3][3] = {{1.0001, -0.0146, 0}, {0.0189, 0.9928, 0 }, {0, 0, 1}};
//    cv::Mat t_m(3, 3, CV_64F, m);
//    img = cv::imread(imgPath);
//    std::cout << "transform mat" << t_m << std::endl;
//    cv::warpPerspective(img, img , t_m, cv::Size(img.cols, img.rows), cv::INTER_AREA, cv::BORDER_CONSTANT, (255, 255, 255));
//    cv::imwrite("./debug.jpeg", img);