#include <opencv2/opencv.hpp>
#include<algorithm>
#include<vector>
#include<unordered_map>
#include<deque>

typedef struct stLabelPoint
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
} stLabelPoint;

template<typename LabelT>
inline static LabelT findRoot(const LabelT *P, LabelT i)
{
    LabelT root = i;
    while (P[root] < root)
    {
        root = P[root];
    }
    return root;
}

template<typename LabelT>
inline static void setRoot(LabelT *P, LabelT i, LabelT root)
{
    while (P[i] < i)
    {
        LabelT j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

template<typename LabelT>
inline static LabelT set_union(LabelT *P, LabelT i, LabelT j)
{
    LabelT root = findRoot(P, i);
    if (i != j)
    {
        LabelT rootj = findRoot(P, j);
        if (root > rootj)
        {
            root = rootj;
        }
        setRoot(P, j, root);
    }
    setRoot(P, i, root);
    return root;
}

template<typename LabelT>
inline static void flattenL(LabelT *P, const int start, const int nElem, LabelT& k)
{
    size_t end = start + nElem;
    for (int i = start; i < end; ++i)
    {
        if (P[i] < i)
        {//node that point to root
            P[i] = P[P[i]];
        }
        else
        { //for root node
            P[i] = k;
            k = k + 1;
        }
    }
}

class FirstScan4Connectivity : public cv::ParallelLoopBody
{
public:
    FirstScan4Connectivity(const uchar* img, int* imgLabels, int *P, int *chunksSizeAndLabels, int width)
        : pImg(img), pLabel(imgLabels), P_(P), chunksSizeAndLabels_(chunksSizeAndLabels), nWidth(width){}

    FirstScan4Connectivity&  operator=(const FirstScan4Connectivity& ) { return *this; }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int r = range.start;
        chunksSizeAndLabels_[r] = range.end;

        int label = int((r * nWidth + 1) / 2 + 1);

        const int firstLabel = label;
        const int w = nWidth;
        const int limitLine = r, startR = r;

        // Rosenfeld Mask
        // +-+-+-+
        // |-|q|-|
        // +-+-+-+
        // |s|x|
        // +-+-+
        size_t offset_cur = 0, offset_prev = 0;
        for (; r != range.end; ++r){
            offset_cur = r * nWidth;
            offset_prev = (r - 1) * nWidth;
            uchar const * const img_row = pImg + offset_cur;
            uchar const * const img_row_prev = pImg + offset_prev;
            int * const imgLabels_row = pLabel + offset_cur;
            int * const imgLabels_row_prev = pLabel + offset_prev;
            for (int c = 0; c < w; ++c) {

#define condition_q r > limitLine && img_row_prev[c] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

                if (condition_x){
                    if (condition_q){
                        if (condition_s){
                            //step s->x->q. Merge
                            imgLabels_row[c] = set_union(P_, imgLabels_row[c - 1], imgLabels_row_prev[c]);
                        }
                        else{
                            //copy q
                            imgLabels_row[c] = imgLabels_row_prev[c];
                        }
                    }
                    else{
                        if (condition_s){ // copy s
                            imgLabels_row[c] = imgLabels_row[c - 1];
                        }
                        else{
                            //new label
                            imgLabels_row[c] = label;
                            P_[label] = label;
                            label = label + 1;
                        }
                    }
                }
                else{
                    //x is a background pixel
                    imgLabels_row[c] = 0;
                }
            }
        }
        //write in the following memory location
        chunksSizeAndLabels_[startR + 1] = label - firstLabel;
    }
#undef condition_q
#undef condition_s
#undef condition_x

private:
    const uchar* pImg;
    int* pLabel;
    int* P_;
    int* chunksSizeAndLabels_;
    int nWidth;
};

class SecondScan : public cv::ParallelLoopBody{
    int* imgLabels_;
    const int *P_;
    int nWidth;

public:
    SecondScan(int* imgLabels, const int *P, int width)
        : imgLabels_(imgLabels), P_(P), nWidth(width){}

    SecondScan&  operator=(const SecondScan& ) { return *this; }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int* img_row_start = imgLabels_ + nWidth * range.start;
        int* img_row_end = imgLabels_ + nWidth * range.end;

        for (; img_row_start != img_row_end; ++img_row_start)
        {
            *img_row_start = P_[*img_row_start];
        }

        // int rowEnd = range.end;

        // int i = range.start, j = 0, label = 0;
        // size_t offsetIndex = 0, realIndex = 0;
        // std::vector<stLabelPoint> vBfs;
        // for (; i < rowEnd; ++i)
        // {
        //     offsetIndex = i * nWidth;
        //     for (j = 0; j < nWidth; ++j)
        //     {
        //         realIndex = offsetIndex + j;
        //         label = P_[imgLabels_[realIndex]];
        //         imgLabels_[realIndex] = label;
        //         if (0 == label)
        //         {
        //             continue;
        //         }
        //     }
        // }
    }
};

inline static void mergeLabels4Connectivity(int* pLabel, int* P, const int* chunksSizeAndLabels, int nHeight, int nWidth)
{
    int c = 0;
    int* imgLabels_row = nullptr;
    int* imgLabels_row_prev = nullptr;
    for (int r = chunksSizeAndLabels[0]; r < nHeight; r = chunksSizeAndLabels[r])
    {
        imgLabels_row = pLabel + r * nWidth;
        imgLabels_row_prev = imgLabels_row - nWidth;

        for (c = 0; c < nWidth; ++c)
        {
            if (imgLabels_row[c] > 0 && imgLabels_row_prev[c] > 0)
            {
                imgLabels_row[c] = set_union(P, imgLabels_row_prev[c], imgLabels_row[c]);
            }
        }
    }
}


namespace secondscan
{
    static int* pLabel;
    static int* P;
    static int nHeight;
    static int nWidth;
    static int nThreads;
    static int nPatchHeight;

    void set_common_value(int* pLabel, int* P, int nHeight, int nWidth, int nThreads)
    {
        secondscan::pLabel = pLabel;
        secondscan::P = P;
        secondscan::nHeight = nHeight;
        secondscan::nWidth = nWidth;
        secondscan::nThreads = nThreads;
        secondscan::nPatchHeight = int((nHeight - 1)/nThreads + 1);
    }

    void* run_thread_second_scan_loop(void* ptr)
    {
        int id = *(int*)ptr;
        int rowBegin = id * nPatchHeight;
        rowBegin = rowBegin < nHeight ? rowBegin : nHeight;
        int rowEnd = rowBegin + nPatchHeight;
        rowEnd = rowEnd < nHeight ? rowEnd : nHeight;

        int* img_row_start = pLabel + nWidth * rowBegin;
        int* img_row_end = pLabel + nWidth * rowEnd;
        for (; img_row_start != img_row_end; ++img_row_start)
        {
            *img_row_start = P[*img_row_start];
        }
    }

    void second_scan_threads()
    {
        pthread_t* threads = new pthread_t[nThreads];
        //pthread_t* threads = (pthread_t*)malloc(nThreads * sizeof(pthread_t));
        int* ids = new int[nThreads];
        for (int i = 0; i < nThreads; ++i)
        {
            ids[i] = i;
            pthread_create(&threads[i], 0, run_thread_second_scan_loop, &ids[i]);
        }
        for (int i = 0; i < nThreads; ++i)
        {
            pthread_join(threads[i], 0);
        }

        delete[] threads;
        delete[] ids;
    }
}


int connected_self(uchar* pImg, int* pLabel, int nHeight, int nWidth)
{

    int* chunksSizeAndLabels = (int*)cv::fastMalloc(nHeight * sizeof(int));

    size_t Plength = (nHeight * nWidth + 1) / 2 + 1;
    int* P = (int*)cv::fastMalloc(Plength * sizeof(int));
    P[0] = 0;

    cv::Range range(0, nHeight);
    double nParallelStripes = std::max(1, std::min(nHeight / 2, cv::getNumThreads()*4));

    cv::parallel_for_(range, FirstScan4Connectivity(pImg, pLabel, P, chunksSizeAndLabels, nWidth), nParallelStripes);

    mergeLabels4Connectivity(pLabel, P, chunksSizeAndLabels, nHeight, nWidth);

    int nLabels = 1;
    for (int i = 0; i < nHeight; i = chunksSizeAndLabels[i]){
        flattenL(P, int(i * nWidth + 1) / 2 + 1, chunksSizeAndLabels[i + 1], nLabels);
    }

    // std::vector<std::vector<int, std::vector<int>>> vLabelForClear(nHeight, std::vector<int, std::vector<int>>(nLabels));
    // std::vector<std::vector<int, std::vector<int>>> vLabelForBfsX(nHeight, std::vector<int, std::vector<int>>(nLabels));
    // std::vector<std::vector<int, std::vector<int>>> vLabelForBfsY(nHeight, std::vector<int, std::vector<int>>(nLabels));
    //change
    //cv::parallel_for_(range, SecondScan(pLabel, P, nWidth, &vLabelForClear, &vLabelForBfsX, &vLabelForBfsY), nParallelStripes);
    cv::parallel_for_(range, SecondScan(pLabel, P, nWidth), nParallelStripes);

    // secondscan::set_common_value(pLabel, P, nHeight, nWidth, nParallelStripes);
    // secondscan::second_scan_threads();

    cv::fastFree(chunksSizeAndLabels);
    cv::fastFree(P);
    return nLabels;
}