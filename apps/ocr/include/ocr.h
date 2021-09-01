//
// Created by 周炳诚 on 2021/3/26.
//

#ifndef __OCR_H__
#define __OCR_H__

# include "tr.h"
# include "td.h"

namespace pipeline {
    using namespace dl_model;

    class OcrPipeLine {
    public:

        OcrPipeLine(td::PseModel & _td_model, tr::CrnnModel & _tr_model);

        OcrPipeLine(const std::string & td_model_path, const std::string & tr_model_path, const std::string & vocab_path,
                    int _max_height = 48, int _tr_nc = 1, const std::string & mode = "gpu", int index = DEFAULT_DEVICE_ID,
                    bool td_half = false);

        void predict(cv::Mat & img, std::vector<TextRect> & text_rect_vec);

        std::vector<TextRect> predict(cv::Mat & img);

        void predict(const std::string & img_path, std::vector<TextRect> & text_rect_vec);

        std::vector<TextRect> predict(const std::string & img_path);



    private:
        td::PseModel td_model;
        tr::CrnnModel tr_model;

    };
}

#endif //__OCR_H__
