#include "Trt.h"
#include <vector>
#include <string>
#include <ctime>

#include "opencv2/opencv.hpp"

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

int main(int argc, char ** argv) {
    TrtPluginParams params = new TrtPluginParams();
    Trt* trt = new Trt(params);

    std::cout << "usage: sh /path/to/vis -prototxt path/to/prototxt -caffemodel path/to/caffemodel/ -save_engine path/to/save_engin -input path/to/input/img" << std::endl;
    InputParser cmdparams(argc, argv);
    const std::string& prototxt = cmdparams.getCmdOption("-prototxt");
    const std::string& caffemodel = cmdparams.getCmdOption("-caffemodel");
    const std::string& save_engine = cmdparams.getCmdOption("-save_engine");
    const std::string& img_name = cmdparams.getCmdOption("-input");
    std::vector<std::string> outputBlobName;
    outputBlobName.push_back("layer1-act");
    int maxBatchSize = 1;
    int mode=0;
    trt->CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, calibratorData, maxBatchSize, mode);

    cv::Mat img = cv::imread(img_name);
    int c = 3;
    int h = 416;   //net h
    int w = 416;   //net w

    float scale = std::min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,cv::INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    cv::Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height);
    resized.copyTo(cropped(rect));

    // cv::imwrite("cropped.jpg",cropped);

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(c);
    cv::split(img_float, input_channels);

    YoloInDataSt* input = new YoloInDataSt();
    input->originalWidth = img.cols;
    input->originalHeight = img.rows;
    float* data = input->data;
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    
}