/*
 * @Author: zerollzeng
 * @Date: 2019-10-18 09:40:30
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-18 16:52:29
 */
#include "Trt.h"

#include <string>

int main() {
    std::string onnxModelpath = "./models/bvlc_googlenet/model.onnx";
    std::string engineFile = "";
    const std::vector<std::string> customOutput{"conv2/3x3_2"};
    int maxBatchSize = 1;
    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine(onnxModelpath, engineFile, customOutput, maxBatchSize);
    onnx_net->Forward();
    return 0;
}