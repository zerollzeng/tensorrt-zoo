#ifndef YOLOV3_HPP
#define YOLOV3_HPP

#include "Trt.h"

struct YoloInDataSt{
    std::vector<float> data;
    int originalWidth;
    int originalHeight;
};

struct Bbox {
    int left,right,top,bottom;
    float score;
};

struct YoloOutDataSt{
    std::vector<Bbox> result;
};

class YoloV3 {
public:    
    YoloV3(const std::string& prototxt, 
            const std::string& caffeModel,
            const std::string& engineFile,
            const std::vector<std::string>& outputBlobName,
            const std::vector<std::vector<float>>& calibratorData,
            int maxBatchSize,
            int mode,
            int yoloClassNum,
            int netSize);

    ~YoloV3();

    void DoInference(YoloInDataSt* input, std::vector<Bbox>& output);

protected:
    Trt* mNet;
    
    int mYoloClassNum;

    const int mNetWidth = 416;
    const int mNetHeight = 416;

    std::vector<float> mpDetCpu;
};

#endif // YOLOV3_HPP