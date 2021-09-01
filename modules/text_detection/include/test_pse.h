/**
* @program: ocr
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-03-26 11:09
**/
#ifndef __TEST_PSE_H__
#define __TEST_PSE_H__
#include<vector>
#include "base_core.h"

void opti_pse(int* pLabel, float* pScore, int nCols, int nRows, int label_num,
              int min_area, float scale, float min_score, float ratio,
              std::vector<sRotatedRect> & rot_vec,bool return_points=true, bool debug_mode=false);

void opti_pred(unsigned char* pKernel, int nKernelNum, int nKernelHeight, int nKernelWidth, float fMinArea, int* pPred, bool debug_mode);

int opti_pred_new(unsigned char* pKernel,
                  int nKernelNum,
                  int nKernelHeight,
                  int nKernelWidth,
                  float min_kernel_area,
                  int scale, //
                  double* pScore,
                  int min_area,
                  float min_score,
                  float ratio,
                  float* pBoxArray,
                  int* pBoxMask);

int connected_self(uchar* pImg, int* pLabel, int nHeight, int nWidth);

#endif