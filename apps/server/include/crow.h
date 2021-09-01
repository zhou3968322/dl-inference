/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-21 10:44
**/
//
// Created by 周炳诚 on 2021/5/21.
//
// all this is shamelessly copy from https://github.com/ipkn/crow

#ifndef __DL_INFERENCE_CROW_H__
#define __DL_INFERENCE_CROW_H__

#include <chrono>
#include <string>
#include <functional>
#include <memory>
#include <future>
#include <cstdint>
#include <type_traits>
#include <thread>
#include <condition_variable>
#include <boost/asio.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "crow_request.h"
#include "crow_response.h"
#include "crow_server.h"
#include "crow_base_rule.h"

#ifdef CROW_ENABLE_SSL
#include <boost/asio/ssl.hpp>
#endif


namespace crow
{
    // Any middleware requires following 3 members:

    // struct context;
    //      storing data for the middleware; can be read from another middleware or handlers

    // before_handle
    //      called before handling the request.
    //      if res.end() is called, the operation is halted.
    //      (still call after_handle of this middleware)
    //      2 signatures:
    //      void before_handle(request& req, response& res, context& ctx)
    //          if you only need to access this middlewares context.
    //      template <typename AllContext>
    //      void before_handle(request& req, response& res, context& ctx, AllContext& all_ctx)
    //          you can access another middlewares' context by calling `all_ctx.template get<MW>()'
    //          ctx == all_ctx.template get<CurrentMiddleware>()

    // after_handle
    //      called after handling the request.
    //      void after_handle(request& req, response& res, context& ctx)
    //      template <typename AllContext>
    //      void after_handle(request& req, response& res, context& ctx, AllContext& all_ctx)

    struct CookieParser
    {
        struct context
        {
            std::unordered_map<std::string, std::string> jar;
            std::unordered_map<std::string, std::string> cookies_to_add;

            std::string get_cookie(const std::string& key) const
            {
                auto cookie = jar.find(key);
                if (cookie != jar.end())
                    return cookie->second;
                return {};
            }

            void set_cookie(const std::string& key, const std::string& value)
            {
                cookies_to_add.emplace(key, value);
            }
        };

        void before_handle(request& req, response& res, context& ctx)
        {
            int count = req.headers.count("Cookie");
            if (!count)
                return;
            if (count > 1)
            {
                res.code = 400;
                res.end();
                return;
            }
            std::string cookies = req.get_header_value("Cookie");
            size_t pos = 0;
            while(pos < cookies.size())
            {
                size_t pos_equal = cookies.find('=', pos);
                if (pos_equal == cookies.npos)
                    break;
                std::string name = cookies.substr(pos, pos_equal-pos);
                boost::trim(name);
                pos = pos_equal+1;
                while(pos < cookies.size() && cookies[pos] == ' ') pos++;
                if (pos == cookies.size())
                    break;

                size_t pos_semicolon = cookies.find(';', pos);
                std::string value = cookies.substr(pos, pos_semicolon-pos);

                boost::trim(value);
                if (value[0] == '"' && value[value.size()-1] == '"')
                {
                    value = value.substr(1, value.size()-2);
                }

                ctx.jar.emplace(std::move(name), std::move(value));

                pos = pos_semicolon;
                if (pos == cookies.npos)
                    break;
                pos++;
                while(pos < cookies.size() && cookies[pos] == ' ') pos++;
            }
        }

        void after_handle(request& /*req*/, response& res, context& ctx)
        {
            for(auto& cookie:ctx.cookies_to_add)
            {
                if (cookie.second.empty())
                    res.add_header("Set-Cookie", cookie.first + "=\"\"");
                else
                    res.add_header("Set-Cookie", cookie.first + "=" + cookie.second);
            }
        }
    };

    /*
    App<CookieParser, AnotherJarMW> app;
    A B C
    A::context
        int aa;

    ctx1 : public A::context
    ctx2 : public ctx1, public B::context
    ctx3 : public ctx2, public C::context

    C depends on A

    C::handle
        context.aaa

    App::context : private CookieParser::contetx, ...
    {
        jar

    }

    SimpleApp
    */
}


#ifdef CROW_MSVC_WORKAROUND
#define CROW_ROUTE(app, url) app.route_dynamic(url)
#else
#define CROW_ROUTE(app, url) app.route<crow::black_magic::get_parameter_tag(url)>(url)
#endif

namespace crow
{
#ifdef CROW_ENABLE_SSL
    using ssl_context_t = boost::asio::ssl::context;
#endif
    template <typename ... Middlewares>
    class Crow
    {
    public:
        using self_t = Crow;
        using server_t = Server<Crow, SocketAdaptor, Middlewares...>;
#ifdef CROW_ENABLE_SSL
        using ssl_server_t = Server<Crow, SSLAdaptor, Middlewares...>;
#endif
        Crow()
        {
        }

        template <typename Adaptor>
        void handle_upgrade(const request& req, response& res, Adaptor&& adaptor)
        {
            router_.handle_upgrade(req, res, adaptor);
        }

        void handle(const request& req, response& res)
        {
            router_.handle(req, res);
        }

        DynamicRule& route_dynamic(std::string&& rule)
        {
            return router_.new_rule_dynamic(std::move(rule));
        }

        template <uint64_t Tag>
        auto route(std::string&& rule)
        -> typename std::result_of<decltype(&Router::new_rule_tagged<Tag>)(Router, std::string&&)>::type
        {
            return router_.new_rule_tagged<Tag>(std::move(rule));
        }

        self_t& port(std::uint16_t port)
        {
            port_ = port;
            return *this;
        }

        self_t& bindaddr(std::string bindaddr)
        {
            bindaddr_ = bindaddr;
            return *this;
        }

        self_t& multithreaded()
        {
            return concurrency(std::thread::hardware_concurrency());
        }

        self_t& concurrency(std::uint16_t concurrency)
        {
            if (concurrency < 1)
                concurrency = 1;
            concurrency_ = concurrency;
            return *this;
        }

        void validate()
        {
            router_.validate();
        }

        void notify_server_start()
        {
            std::unique_lock<std::mutex> lock(start_mutex_);
            server_started_ = true;
            cv_started_.notify_all();
        }

        void run()
        {
            validate();
#ifdef CROW_ENABLE_SSL
            if (use_ssl_)
            {
                ssl_server_ = std::move(std::unique_ptr<ssl_server_t>(new ssl_server_t(this, bindaddr_, port_, &middlewares_, concurrency_, &ssl_context_)));
                ssl_server_->set_tick_function(tick_interval_, tick_function_);
                notify_server_start();
                ssl_server_->run();
            }
            else
#endif
            {
                server_ = std::move(std::unique_ptr<server_t>(new server_t(this, bindaddr_, port_, &middlewares_, concurrency_, nullptr)));
                server_->set_tick_function(tick_interval_, tick_function_);
                notify_server_start();
                server_->run();
            }
        }

        void stop()
        {
#ifdef CROW_ENABLE_SSL
            if (use_ssl_)
            {
                ssl_server_->stop();
            }
            else
#endif
            {
                server_->stop();
            }
        }

        void debug_print()
        {
            CROW_LOG_DEBUG << "Routing:";
            router_.debug_print();
        }

        self_t& loglevel(crow::LogLevel level)
        {
            crow::logger::setLogLevel(level);
            return *this;
        }

#ifdef CROW_ENABLE_SSL
        self_t& ssl_file(const std::string& crt_filename, const std::string& key_filename)
        {
            use_ssl_ = true;
            ssl_context_.set_verify_mode(boost::asio::ssl::verify_peer);
            ssl_context_.use_certificate_file(crt_filename, ssl_context_t::pem);
            ssl_context_.use_private_key_file(key_filename, ssl_context_t::pem);
            ssl_context_.set_options(
                    boost::asio::ssl::context::default_workarounds
                          | boost::asio::ssl::context::no_sslv2
                          | boost::asio::ssl::context::no_sslv3
                    );
            return *this;
        }

        self_t& ssl_file(const std::string& pem_filename)
        {
            use_ssl_ = true;
            ssl_context_.set_verify_mode(boost::asio::ssl::verify_peer);
            ssl_context_.load_verify_file(pem_filename);
            ssl_context_.set_options(
                    boost::asio::ssl::context::default_workarounds
                          | boost::asio::ssl::context::no_sslv2
                          | boost::asio::ssl::context::no_sslv3
                    );
            return *this;
        }

        self_t& ssl(boost::asio::ssl::context&& ctx)
        {
            use_ssl_ = true;
            ssl_context_ = std::move(ctx);
            return *this;
        }


        bool use_ssl_{false};
        ssl_context_t ssl_context_{boost::asio::ssl::context::sslv23};

#else
        template <typename T, typename ... Remain>
        self_t& ssl_file(T&&, Remain&&...)
        {
            // We can't call .ssl() member function unless CROW_ENABLE_SSL is defined.
            static_assert(
                    // make static_assert dependent to T; always false
                    std::is_base_of<T, void>::value,
                    "Define CROW_ENABLE_SSL to enable ssl support.");
            return *this;
        }

        template <typename T>
        self_t& ssl(T&&)
        {
            // We can't call .ssl() member function unless CROW_ENABLE_SSL is defined.
            static_assert(
                    // make static_assert dependent to T; always false
                    std::is_base_of<T, void>::value,
                    "Define CROW_ENABLE_SSL to enable ssl support.");
            return *this;
        }
#endif

        // middleware
        using context_t = detail::context<Middlewares...>;
        template <typename T>
        typename T::context& get_context(const request& req)
        {
            static_assert(black_magic::contains<T, Middlewares...>::value, "App doesn't have the specified middleware type.");
            auto& ctx = *reinterpret_cast<context_t*>(req.middleware_context);
            return ctx.template get<T>();
        }

        template <typename T>
        T& get_middleware()
        {
            return utility::get_element_by_type<T, Middlewares...>(middlewares_);
        }

        template <typename Duration, typename Func>
        self_t& tick(Duration d, Func f) {
            tick_interval_ = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            tick_function_ = f;
            return *this;
        }

        void wait_for_server_start()
        {
            std::unique_lock<std::mutex> lock(start_mutex_);
            if (server_started_)
                return;
            cv_started_.wait(lock);
        }

    private:
        uint16_t port_ = 80;
        uint16_t concurrency_ = 1;
        std::string bindaddr_ = "0.0.0.0";
        Router router_;

        std::chrono::milliseconds tick_interval_;
        std::function<void()> tick_function_;

        std::tuple<Middlewares...> middlewares_;

#ifdef CROW_ENABLE_SSL
        std::unique_ptr<ssl_server_t> ssl_server_;
#endif
        std::unique_ptr<server_t> server_;

        bool server_started_{false};
        std::condition_variable cv_started_;
        std::mutex start_mutex_;
    };
    template <typename ... Middlewares>
    using App = Crow<Middlewares...>;
    using SimpleApp = Crow<>;
}



#endif //__DL_INFERENCE_CROW_H__
