/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 14:50:04
 * @LastEditTime: 2020-11-5 10:30:43
 * @LastEditors: https://github.com/ZHEQIUSHUI
 */
#ifndef YOLOV3_HPP
#define YOLOV3_HPP

#include "Trt.h"

struct Bbox {
	int left, right, top, bottom;
	int clsId;
	float score;
};
struct YoloInDataSt{
    std::vector<float> data;
    std::vector<int> originalWidths;
    std::vector<int> originalHeights;
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
		int device,
		int yoloClassNum,
		int netSize);

	~YoloV3();

	void DoInference(YoloInDataSt* input, int batchsize, std::vector<std::vector<Bbox>>& output);

protected:
	Trt* mNet;

	int mYoloClassNum;

	int mNetWidth = 608;
	int mNetHeight = 608;
	int factor = 63883;

	std::vector<float> mpDetCpu;
};

#endif // YOLOV3_HPP
