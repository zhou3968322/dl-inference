/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-20 10:07
**/
//
// Created by 周炳诚 on 2021/5/20.
//

#ifndef __DL_INFERENCE_BASE64_H__
#define __DL_INFERENCE_BASE64_H__

#include <string>

std::string base64_encode(unsigned char const* , unsigned int len);
std::string base64_decode(std::string const& s);

#endif //__DL_INFERENCE_BASE64_H__
