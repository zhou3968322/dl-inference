/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-04-01 11:33
**/
//
// Created by 周炳诚 on 2021/4/1.
//

#include "base_core.h"
#include "util.h"


using namespace torch::indexing;


int main(){
    cv::Point2f center1(1578.7, 142.223);
    cv::Size2f size1(531.995, 43.5925);
    float angle1 = -3.20522;
    float score1 = 0.992058;
    sRotatedRect s_rect1(center1, size1, angle1, score1);
    cv::Point2f vertices1[4];
    s_rect1.points(vertices1);
    cv::Point2f center2(2374, 202.503);
    cv::Size2f size2(517.613, 36.6417);
    float angle2 = -4.13467;
    float score2 = 0.983481;
    sRotatedRect s_rect2(center2, size2, angle2, score2);
    cv::Point2f vertices2[4];
    s_rect2.points(vertices2);
    cv::Point2f center3(580.68, 377.64);
    cv::Size2f size3(346.056, 34.4023);
    float angle3 = -2.48955;
    float score3 = 0.986858;
    sRotatedRect s_rect3(center3, size3, angle3, score3);
    cv::Point2f vertices3[4];
    s_rect3.points(vertices3);
    // points are lb, lt, rt, rb
    int equation_nums = 3;
    torch::Tensor A = torch::zeros({equation_nums * 3, 6});
    torch::Tensor B = torch::zeros({equation_nums * 3, 1});
    rect_to_transform_tensor(A, B, s_rect1, 0, equation_nums);
    rect_to_transform_tensor(A, B, s_rect2, 1, equation_nums);
    rect_to_transform_tensor(A, B, s_rect3, 2, equation_nums);
    printTensor(A, "A");
    printTensorV(A, "A");
    printTensor(B, "B");
    printTensorV(B, "B");
    auto res = torch::svd(A);
    at::Tensor U = std::get<0>(res);
    at::Tensor S = std::get<1>(res);
    at::Tensor V = std::get<2>(res);
    printTensor(U, "U");
    printTensorV(U, "U");
    printTensor(S, "S");
    printTensorV(S, "S");
    printTensor(V, "V");
    printTensorV(V, "V");
    std::cout << "U is :" << U << std::endl;
    std::cout << "S is:" << S << std::endl;
    std::cout << "V is:" << V << std::endl;
    at::Tensor S_invert = get_diag_invert(S);
    at::Tensor X;
    X = torch::mm(torch::mm(torch::mm(V, S_invert), U.t()), B);

    printTensor(X, "X");
    printTensorV(X, "X");

    std::cout << "X" << X << std::endl;
    return 0;
}