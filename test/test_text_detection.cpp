/**
* @program: dl_core
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-28 21:42
**/
//
// Created by 周炳诚 on 2021/3/28.
//
#include <opencv2/core/utility.hpp>
#include "td.h"


using namespace cv;
using namespace std;

const char* keys =
        {
                "{help h usage?| | jit_pse_normal.pt document.jpeg}"
                "{@pse_model_path | pse model path | test pse model json path}"
                "{@img_path | test img path | test img path}"
        };

int main(int argc, const char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    string modelPath= parser.get<String>(0);;
    string imgPath = parser.get<String>(1);;
    dl_model::td::PseModel pse_model(modelPath, "gpu", 0);
    at::Tensor input_tensor;
    float ratio1, ratio2;
    float default_ratio = 0.952786;
    ratio1 = pse_model.loadDetectionImage(imgPath, input_tensor);
    Mat img;
    img = imread(imgPath, IMREAD_COLOR);
    cvtColor(img, img, COLOR_BGR2RGB);
    int img_width = img.cols;
    ratio2 = pse_model.loadDetectionImage(img, input_tensor);
    cout << "ratio1:" << ratio1 << ",ratio2:" << ratio2 << endl;
    at::Tensor box_tensor;
    box_tensor = pse_model.predict(imgPath);
    cout << "box_tensor size:" << box_tensor.size(0) << endl;
    img = imread(imgPath, IMREAD_COLOR);
    cvtColor(img, img, COLOR_BGR2RGB);
    box_tensor = pse_model.predict(img);
    cout << "box_tensor size:" << box_tensor.size(0) << endl;
    vector<sRotatedRect> s_rotated_rect;
    pse_model.predict(img, s_rotated_rect, true);
    cout << "rot vec size:" << s_rotated_rect.size() << endl;
    float angle = 1.0;
    angle = get_mean_angle(s_rotated_rect, img_width);
    cout << "mean angle is:" << angle << endl;
    angle = -1.0;
    rotate_img_box_points(img, s_rotated_rect, angle);
    rotate_img_box_points(img, s_rotated_rect);
    vector<Rect> rect_vec;
    rect_vec = rot_vec2rect(s_rotated_rect);
    cout << "rot vec size:" << rect_vec.size() << endl;
    dl_model::td::PseModel pse_half_model(modelPath, "gpu", 0, true);
    box_tensor = pse_half_model.predict(imgPath);
    printTensor(box_tensor);
    cout << "box_tensor size:" << box_tensor.size(0) << endl;
    std::cout << "success test text detection" << std::endl;
    return 0;
}
