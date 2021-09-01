#include<iostream>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<deque>
#include<algorithm>
#include "test_pse.h"
using namespace std;
using namespace cv;


void opti_pse(int* pLabel, float* pScore, int nCols, int nRows, int label_num,
              int min_area, float scale, float min_score, float ratio,
              vector<sRotatedRect> & rot_vec, bool return_points, bool debug_mode){
    if(debug_mode)
    {
        cout << "nCols: " << nCols << endl;
        cout << "nRows: " << nRows << endl;
        cout << "label_num: " << label_num << endl;
        cout << "min_area: " << min_area << endl;
        cout << "scale: " << scale << endl;
        cout << "min_score: " << min_score << endl;
        cout << "ratio: " << ratio << endl;

        for(int i=0; i<1; ++i)
        {
            cout << "label row: " << i <<endl;
            for(int j=0; j<100; ++j)
            {
                cout << pLabel[nCols*i + j] << " ";
            }
            cout <<endl;

            cout << "score row: " << i <<endl;
            for(int j=0; j<100; ++j)
            {
                cout << pScore[nCols*i + j] << " ";
            }
            cout <<endl;
        }
    }

    vector<vector<Point>> split_points(label_num, vector<Point>());
    vector<float> vec_sum_scores(label_num, 0.0f);

    int label_char = 0;
    size_t index = 0;
    for(int i=0; i<nRows; ++i)
    {
        for(int j=0; j<nCols; ++j)
        {
            index = nCols*i + j;
            label_char = pLabel[index];
            if(0 == label_char)
            {
                continue;
            }
            split_points[label_char].push_back(Point(j, i));
            vec_sum_scores[label_char] += pScore[index];
        }
    }

    size_t point_num = 0;
    int min_point_num = (int) min_area/(scale*scale);
    float ratio_f = ratio / scale;
    for (int i=1; i<label_num; ++i)
    {
        point_num = split_points[i].size();
        if (point_num < min_point_num)
        {
            continue;
        }
        if (vec_sum_scores[i] / point_num < min_score)
        {
            continue;
        }
        if (return_points){
            sRotatedRect srotated_rect(split_points[i], vec_sum_scores[i] / point_num, ratio_f);
            rot_vec.push_back(srotated_rect);
            if(debug_mode)
            {
                cout << "rotated rect:" << srotated_rect << endl;
            }
        }else{
            RotatedRect rotated_rect = minAreaRect(split_points[i]);
            sRotatedRect srotated_rect(rotated_rect, vec_sum_scores[i] / point_num, ratio_f);
            rot_vec.push_back(srotated_rect);
            if(debug_mode)
            {
                cout << "rotated rect:" << srotated_rect << endl;
            }
        }
    }
}


struct stLabelPoint
{
    int x;
    int y;
    int label;

    stLabelPoint(int xx, int yy, int llabel)
    {
        x = xx;
        y = yy;
        label = llabel;
    }
};

bool comp(const stLabelPoint& a, const stLabelPoint& b)
{
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

void opti_pred(unsigned char* pKernel, int nKernelNum, int nKernelHeight, int nKernelWidth, float fMinArea, int* pPred, bool debug_mode)
{
    if (debug_mode)
    {
        cout << "nKernelNum: " << nKernelNum << endl;
        cout << "nKernelHeight: " << nKernelHeight << endl;
        cout << "nKernelWidth: " << nKernelWidth << endl;
        cout << "fMinArea: " << fMinArea << endl;

        cout << "pKernel: ";
        for (int i=0; i <nKernelWidth; ++i)
        {
            cout << pKernel[i] << " ";
        }
        cout << endl;

        cout << "pPred: ";
        for (int i=0; i <nKernelWidth; ++i)
        {
            cout << pPred[i] << " ";
        }
        cout << endl;
    }
    size_t nSize = nKernelHeight * nKernelWidth;
    Mat oriImg(nKernelHeight, nKernelWidth, CV_8U, pKernel+nSize*(nKernelNum-1));
    Mat labelImg(nKernelHeight, nKernelWidth, CV_32S, pPred);
    int nLabelNum = connectedComponents(oriImg, labelImg, 4, CV_32S);

    vector<vector<size_t>> vLabelIndex(nLabelNum, vector<size_t>());
    for (size_t i = 0; i<nSize; ++i)
    {
        if (0 == pPred[i])
        {
            continue;
        }
        vLabelIndex[pPred[i]].push_back(i);
    }

    vector<size_t>::iterator iter;
    for (int i=1; i<nLabelNum; ++i)
    {
        if (vLabelIndex[i].size() < fMinArea)
        {
            for (iter=vLabelIndex[i].begin(); iter!=vLabelIndex[i].end(); ++iter)
            {
                pPred[*iter] = 0;
            }
        }
    }

    deque<stLabelPoint> deqLabelPoint;
    int i = 0, j = 0, label = 0;
    size_t offsetIndex = 0, realIndex = 0;
    for (i =0; i<nKernelHeight; ++i)
    {
        offsetIndex = i*nKernelWidth;
        for (j=0; j<nKernelWidth; ++j)
        {
            realIndex = offsetIndex + j;
            label = pPred[realIndex];
            if (0 == label)
            {
                continue;
            }
            deqLabelPoint.push_back(stLabelPoint(i, j, label));
        }
    }

    int nOffsetX[4] = {-1, 1, 0, 0};
    int nOffsetY[4] = {0, 0, -1, 1};
    unsigned char* pTmpKernel = nullptr;
    stLabelPoint stTmpLabelPoint(0,0,0);
    int nTmpX = 0, nTmpY = 0;
    offsetIndex = 0;
    bool bIsEdge = true;
    deque<stLabelPoint> deqHelpLabelPoint;
    for (int i=nKernelNum-2; i>-1; --i)
    {
        pTmpKernel = pKernel+nSize*i;

        while(!deqLabelPoint.empty())
        {
            stTmpLabelPoint = deqLabelPoint.front();
            deqLabelPoint.pop_front();
            bIsEdge = true;

            for (int j=0; j<4; ++j)
            {
                nTmpX = stTmpLabelPoint.x + nOffsetX[j];
                nTmpY = stTmpLabelPoint.y + nOffsetY[j];
                if (nTmpX<0 || nTmpX>=nKernelHeight || nTmpY<0 || nTmpY>=nKernelWidth)
                {
                    continue;
                }
                offsetIndex = nTmpX*nKernelWidth + nTmpY;
                if (0 == pTmpKernel[offsetIndex] || pPred[offsetIndex] > 0)
                {
                    continue;
                }

                deqLabelPoint.push_back(stLabelPoint(nTmpX, nTmpY, stTmpLabelPoint.label));
                pPred[offsetIndex] = stTmpLabelPoint.label;
                bIsEdge = false;
            }

            if (bIsEdge)
            {
                deqHelpLabelPoint.push_back(stTmpLabelPoint);
            }
        }
        swap(deqLabelPoint, deqHelpLabelPoint);
    }
}

namespace FinalBox
{
    static int min_point_num;
    static float min_score;
    static float ratio_f;
    static int nLabelNum;
    static vector<vector<Point>>* split_points;
    static vector<float>* vec_sum_scores;
    static float* pBoxArray;
    static int* pBoxMask;

    void* find_loop(void* ptr)
    {
        int id = *(int*)ptr;

        size_t point_num = (*split_points)[id].size();

        if (point_num < min_point_num)
        {
            return nullptr;
        }
        if ((*vec_sum_scores)[id] / point_num < min_score)
        {
            return nullptr;
        }
        pBoxMask[id] = 1;

        RotatedRect restPoint = minAreaRect((*split_points)[id]);
        Mat boxPoints2f;
        boxPoints(restPoint, boxPoints2f);

        for (int j=0; j<4; ++j)
        {
            float* data = boxPoints2f.ptr<float>(j);
            pBoxArray[id*8 + j*2] = data[0]/ratio_f;
            pBoxArray[id*8 + j*2 + 1] = data[1]/ratio_f;
        }
    }

    void get_final_box()
    {
        pthread_t* threads = new pthread_t[nLabelNum];
        int* ids = new int[nLabelNum];

        int i = 0;
        for (; i < nLabelNum; ++i)
        {
            ids[i] = i;
            pthread_create(&threads[i], 0, find_loop, &ids[i]);
        }

        for (i = 0; i < nLabelNum; ++i)
        {
            pthread_join(threads[i], 0);
        }

        delete[] threads;
        delete[] ids;
    }
}

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
                      int* pBoxMask)
{
    size_t nSize = nKernelHeight * nKernelWidth;
    int* pPred = new int[nSize];

    int nLabelNum = connected_self(pKernel+nSize*(nKernelNum-1), pPred, nKernelHeight, nKernelWidth);
    vector<vector<size_t>> vLabelIndex(nLabelNum, vector<size_t>());
    for (size_t i = 0; i<nSize; ++i)
    {
        if (0 == pPred[i])
        {
            continue;
        }
        vLabelIndex[pPred[i]].push_back(i);
    }

    {
        vector<size_t>::iterator iter;
        float fMinArea = min_kernel_area / scale;
        for (int i=1; i<nLabelNum; ++i)
        {
            if (vLabelIndex[i].size() < fMinArea)
            {
                for (iter=vLabelIndex[i].begin(); iter!=vLabelIndex[i].end(); ++iter)
                {
                    pPred[*iter] = 0;
                }
            }
        }
    }

    deque<stLabelPoint> deqLabelPoint;
    int i = 0, j = 0, label = 0;
    size_t offsetIndex = 0, realIndex = 0;
    for (i =0; i<nKernelHeight; ++i)
    {
        offsetIndex = i*nKernelWidth;
        for (j=0; j<nKernelWidth; ++j)
        {
            realIndex = offsetIndex + j;
            label = pPred[realIndex];
            if (0 == label)
            {
                continue;
            }
            deqLabelPoint.push_back(stLabelPoint(i, j, label));
        }
    }

    int nOffsetX[4] = {-1, 1, 0, 0};
    int nOffsetY[4] = {0, 0, -1, 1};
    unsigned char* pTmpKernel = nullptr;
    stLabelPoint stTmpLabelPoint(0,0,0);
    int nTmpX = 0, nTmpY = 0;
    offsetIndex = 0;
    bool bIsEdge = true;
    deque<stLabelPoint> deqHelpLabelPoint;
    for (int i=nKernelNum-2; i>-1; --i)
    {
        pTmpKernel = pKernel+nSize*i;

        while(!deqLabelPoint.empty())
        {
            stTmpLabelPoint = deqLabelPoint.front();
            deqLabelPoint.pop_front();
            bIsEdge = true;

            for (int j=0; j<4; ++j)
            {
                nTmpX = stTmpLabelPoint.x + nOffsetX[j];
                nTmpY = stTmpLabelPoint.y + nOffsetY[j];
                if (nTmpX<0 || nTmpX>=nKernelHeight || nTmpY<0 || nTmpY>=nKernelWidth)
                {
                    continue;
                }
                offsetIndex = nTmpX*nKernelWidth + nTmpY;
                if (0 == pTmpKernel[offsetIndex] || pPred[offsetIndex] > 0)
                {
                    continue;
                }

                deqLabelPoint.push_back(stLabelPoint(nTmpX, nTmpY, stTmpLabelPoint.label));
                pPred[offsetIndex] = stTmpLabelPoint.label;
                bIsEdge = false;
            }

            if (bIsEdge)
            {
                deqHelpLabelPoint.push_back(stTmpLabelPoint);
            }
        }
        swap(deqLabelPoint, deqHelpLabelPoint);
    }

    //end bfs

    vector<vector<Point>> split_points(nLabelNum, vector<Point>());
    vector<float> vec_sum_scores(nLabelNum, 0.0f);
    int label_char = 0;
    {
        int* ppPred = pPred;
        for (i = 0; i < nKernelHeight; ++i)
        {
            for (j = 0; j < nKernelWidth; ++j, ++ppPred, ++pScore)
            {
                label_char = *ppPred;
                if(0 == label_char)
                {
                    continue;
                }
                split_points[label_char].push_back(Point(j, i));
                vec_sum_scores[label_char] += (*pScore);
            }
        }
    }

    FinalBox::min_point_num = min_area/(scale*scale);
    FinalBox::min_score = min_score;
    FinalBox::ratio_f = ratio/scale;
    FinalBox::nLabelNum = nLabelNum;
    FinalBox::split_points = &split_points;
    FinalBox::vec_sum_scores = &vec_sum_scores;
    FinalBox::pBoxMask = pBoxMask;
    FinalBox::pBoxArray = pBoxArray;
    FinalBox::get_final_box();
    std::cout << "PSE-PostProcessing Done" << std::endl;
    delete[] pPred;
    return 0;
}
