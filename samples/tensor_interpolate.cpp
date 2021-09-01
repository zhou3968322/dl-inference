/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-31 16:30
**/
//
// Created by 周炳诚 on 2021/3/31.
//

#include <torch/torch.h>
#include <util.h>

namespace F = torch::nn::functional;

int main(){
    auto option = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    auto input1 = torch::rand({1, 1,  2240, 1600}, option);
    auto input2 = torch::rand({1, 4, 2240, 1600}, option);
    auto options = F::InterpolateFuncOptions().scale_factor(std::vector<double>({0.5, 0.5})).mode(torch::kArea);
    input1 = F::interpolate(input1, options);
    input2 = F::interpolate(input2, options);
    printTensor(input1, "output1");
    printTensor(input2, "output2");
    return 0;
}


