/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-31 12:22
**/
//
// Created by 周炳诚 on 2021/3/31.
//
#include <glob.h>
#include <vector>
#include <stdexcept>
#include <string.h>
#include <sstream>
#include "util.h"

void get_glob_files(const std::string& pattern, std::vector<std::string> & file_names){
    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        file_names.push_back(std::string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);
}

std::vector<std::string> get_glob_files(const std::string& pattern) {
    std::vector<std::string> file_names;

    get_glob_files(pattern, file_names);
    // done
    return file_names;
}


std::vector<std::string> split_path(const std::string& str, const std::set<char> delimiters)
{
    std::vector<std::string> result;

    char const* pch = str.c_str();
    char const* start = pch;
    for(; *pch; ++pch)
    {
        if (delimiters.find(*pch) != delimiters.end())
        {
            if (start != pch)
            {
                std::string str(start, pch);
                result.push_back(str);
            }
            else
            {
                result.push_back("");
            }
            start = pch + 1;
        }
    }
    result.push_back(start);

    return result;
}