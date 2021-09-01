/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-05-21 10:52
**/
//
// Created by 周炳诚 on 2021/5/21.
//

#ifndef __DL_INFERENCE_CROW_HASH_H__
#define __DL_INFERENCE_CROW_HASH_H__


#include <boost/algorithm/string/predicate.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace crow
{
    struct ci_hash
    {
        size_t operator()(const std::string& key) const
        {
            std::size_t seed = 0;
            std::locale locale;

            for(auto c : key)
            {
                boost::hash_combine(seed, std::toupper(c, locale));
            }

            return seed;
        }
    };

    struct ci_key_eq
    {
        bool operator()(const std::string& l, const std::string& r) const
        {
            return boost::iequals(l, r);
        }
    };

    using ci_map = std::unordered_multimap<std::string, std::string, ci_hash, ci_key_eq>;
}


#endif //__DL_INFERENCE_CROW_HASH_H__
