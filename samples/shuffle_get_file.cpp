/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-31 12:34
**/
//
// Created by 周炳诚 on 2021/3/31.
//


#include <random>
#include <algorithm>
#include "util.h"



int main()
{
    std::string path1 = "/data/duser/bczhou_test/c-inference-test/*.jpg";
    std::string path2 = "/data/duser/bczhou_test/c-inference-test/*.png";
    std::string path3 = "/data/duser/bczhou_test/c-inference-test/*.jpeg";

    std::vector<std::string> file_names;
    get_glob_files(path1, file_names);
    get_glob_files(path2, file_names);
    get_glob_files(path3, file_names);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(file_names.begin(), file_names.end(), g);
    for (auto p = file_names.begin(); p != file_names.end(); ++p) {
        std::cout << *p << std::endl;
    }
    return 0;
}
