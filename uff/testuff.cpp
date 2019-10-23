/*
 * @Author: zerollzeng
 * @Date: 2019-10-23 11:24:58
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-23 13:52:23
 */
#include "Trt.h"

#include <string>

int main() {
    std::string uffModel = "./models/faster_rcnn.pb";
    std::string engineFile = "";
    std::vector<std::string> input{"input_1"};
    std::vector<std::string> output{"dense_class/Softmax","dense_regress/BiasAdd","proposal"};
    int maxBatchSize = 1;
    Trt* uff_net = new Trt();
    uff_net->CreateEngine(uffModel, engineFile, input, output, maxBatchSize);
    uff_net->Forward();
    return 0;
}