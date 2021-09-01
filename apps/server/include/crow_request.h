/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-21 10:53
**/
//
// Created by 周炳诚 on 2021/5/21.
//

#ifndef __DL_INFERENCE_CROW_REQUEST_H__
#define __DL_INFERENCE_CROW_REQUEST_H__

#include <boost/asio.hpp>
#include "http_method.h"
#include "qs_parse.h"
#include "crow_hash.h"



namespace crow
{
    struct DetachHelper;

    struct request
    {
        HTTPMethod method;
        std::string raw_url;
        std::string url;
        query_string url_params;
        ci_map headers;
        std::string body;

        void* middleware_context{};
        boost::asio::io_service* io_service{};

        request()
                : method(HTTPMethod::Get)
        {
        }

        request(HTTPMethod method, std::string raw_url, std::string url, query_string url_params, ci_map headers, std::string body)
                : method(method), raw_url(std::move(raw_url)), url(std::move(url)), url_params(std::move(url_params)), headers(std::move(headers)), body(std::move(body))
        {
        }

        void add_header(std::string key, std::string value)
        {
            headers.emplace(std::move(key), std::move(value));
        }

        const std::string& get_header_value(const std::string& key) const
        {
            return crow::get_header_value(headers, key);
        }

        template<typename CompletionHandler>
        void post(CompletionHandler handler)
        {
            io_service->post(handler);
        }

        template<typename CompletionHandler>
        void dispatch(CompletionHandler handler)
        {
            io_service->dispatch(handler);
        }

    };
}

#endif //__DL_INFERENCE_CROW_REQUEST_H__
