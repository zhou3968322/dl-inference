/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-21 10:43
**/
//
// Created by 周炳诚 on 2021/5/21.
//

#ifndef __DL_INFERENCE_CROW_CONFIG_H__
#define __DL_INFERENCE_CROW_CONFIG_H__

// settings for crow
// TODO - replace with runtime config. libucl?

/* #ifdef - enables debug mode */
//#define CROW_ENABLE_DEBUG

/* #ifdef - enables logging */
#define CROW_ENABLE_LOGGING

/* #ifdef - enables ssl */
//#define CROW_ENABLE_SSL

/* #define - specifies log level */
/*
    Debug       = 0
    Info        = 1
    Warning     = 2
    Error       = 3
    Critical    = 4

    default to INFO
*/
#ifndef CROW_LOG_LEVEL
#define CROW_LOG_LEVEL 1
#endif


// compiler flags
#if __cplusplus >= 201402L
#define CROW_CAN_USE_CPP14
#endif

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#define CROW_MSVC_WORKAROUND
#define constexpr const
#define noexcept throw()
#endif
#endif

#endif //__DL_INFERENCE_CROW_CONFIG_H__
