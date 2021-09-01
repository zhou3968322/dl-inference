/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-04-02 14:30
**/
//
// Created by 周炳诚 on 2021/4/2.
//


#include <algorithm>

int main(){
    std::vector<int> v = {5, 2, 8, 10, 9};
    int maxElementIndex = std::max_element(v.begin(),v.end()) - v.begin();
    int maxElement = *std::max_element(v.begin(), v.end());

    int minElementIndex = std::min_element(v.begin(),v.end()) - v.begin();
    int minElement = *std::min_element(v.begin(), v.end());

    std::cout << "maxElementIndex:" << maxElementIndex << ", maxElement:" << maxElement << '\n';
    std::cout << "minElementIndex:" << minElementIndex << ", minElement:" << minElement << '\n';
}