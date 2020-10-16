/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 09:45:10
 * @LastEditTime: 2019-12-04 19:27:27
 * @LastEditors: zerollzeng
 */
#include "OpenPose.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <string>
#include "time.h"

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

int main(int argc, char** argv) {
    std::cout << "usage: path/to/testopenpose --prototxt path/to/prototxt --caffemodel path/to/caffemodel/ --save_engine path/to/save_engin --input path/to/input/img --run_mode 0/1/2" << std::endl;
    InputParser cmdparams(argc, argv);
    const std::string& prototxt = cmdparams.getCmdOption("--prototxt");
    const std::string& caffemodel = cmdparams.getCmdOption("--caffemodel");
    const std::string& save_engine = cmdparams.getCmdOption("--save_engine");
    const std::string& img_name = cmdparams.getCmdOption("--input");
    int run_mode = std::stoi(cmdparams.getCmdOption("--run_mode"));
    int H = std::stoi(cmdparams.getCmdOption("--h"));
    int W = std::stoi(cmdparams.getCmdOption("--w"));

    cv::Mat img = cv::imread(img_name);
    if(img.empty()) {
        std::cout << "error: can not read image" << std::endl;
        return -1;
    }
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    cv::resize(img,img,cv::Size(W,H));

    int N = 1;
    int C = 3;
    std::vector<float> inputData;
    inputData.resize(N * C * H * W);

    unsigned char* data = img.data;
    for(int n=0; n<N;n++) {
        for(int c=0;c<3;c++) {
            for(int i=0;i<W*H;i++) {
                inputData[i+c*W*H+n*3*H*W] = (float)data[i*3+c];
            }
        }
    }
    std::vector<float> result;
    

    std::vector<std::string> outputBlobname{"net_output"};
    std::vector<std::vector<float>> calibratorData;
    calibratorData.resize(3);
    for(size_t i = 0;i<calibratorData.size();i++) {
        calibratorData[i].resize(3*H*W);
        for(size_t j=0;j<calibratorData[i].size();j++) {
            calibratorData[i][j] = 0.05;
        }
    }
    int maxBatchSize = N;

    OpenPose openpose(prototxt,
                        caffemodel,
                        save_engine,
                        outputBlobname,
                        calibratorData,
                        maxBatchSize,
                        run_mode);

    int i=0;
    while(i<1) {
        clock_t start = clock();
        openpose.DoInference(inputData,result);
        clock_t end = clock();
        std::cout << "inference Time : " <<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " ms" << std::endl;
        i++;
    }
    
    cv::cvtColor(img,img,cv::COLOR_RGB2BGR);
    for(size_t i=0;i<result.size()/3;i++) {
        cv::circle(img,cv::Point(result[i*3],result[i*3+1]),2,cv::Scalar(0,255,0),-1);
    }
    cv::imwrite("result.jpg",img);
    
    return 0;

}
