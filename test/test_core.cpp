/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-04-01 11:49
**/
//
// Created by 周炳诚 on 2021/4/1.
//

#include "base_core.h"

using namespace cv;
using namespace std;

int main(){
    Point2f center(1578.7, 142.223);
    Size2f size(531.995, 43.5925);
    float angle = -3.20522;
    float score = 0.992058;
    sRotatedRect s_rect1(center, size, angle, score);
    cout << s_rect1 << endl;
    Point2f vertices[4];
    s_rect1.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cout << vertices[i] << endl;
    }
    sRotatedRect s_rect2(vertices[0], vertices[1], vertices[2], score);
    cout << s_rect2 << std::endl;
    std::cout << "success test core" << std::endl;
    return 0;
}