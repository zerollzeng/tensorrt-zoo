/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 14:50:04
 * @LastEditTime: 2020-11-5 10:30:43
 * @LastEditors: https://github.com/ZHEQIUSHUI
 */
#include "YoloV3.h"
#include "utils.h"
//#include "spdlog/spdlog.h"

#include <NvInfer.h>
#include <NvCaffeParser.h>

#include <chrono>
#include <vector>
#include <cstring>

YoloV3::YoloV3(const std::string& prototxt, 
                const std::string& caffeModel,
                const std::string& engineFile,
                const std::vector<std::string>& outputBlobName,
                const std::vector<std::vector<float>>& calibratorData,
                int maxBatchSize,
                int mode,
				int device,
                int yoloClassNum,
                int netSize) {
    TrtPluginParams params;
    params.yoloClassNum = yoloClassNum;
    params.yolo3NetSize = netSize;
	mNetHeight = netSize;
	mNetWidth = netSize;
    mNet = new Trt(params);
	mNet->SetDevice(device);
    mNet->CreateEngine(prototxt, caffeModel, engineFile, outputBlobName, calibratorData, maxBatchSize, mode);
    mYoloClassNum = yoloClassNum;
	switch (netSize)
	{
	case 416:
		factor = 63883;
		break;
	case 608:
		factor = 136459;
		break;
	default:
		break;
	}
	mpDetCpu.resize(factor* maxBatchSize);
    //
	
}

YoloV3::~YoloV3() {
    if(mNet != nullptr) {
        delete mNet;
        mNet = nullptr;
    }

}


void DoNms(std::vector<Detection>& detections,int classes ,float nmsThresh)
{
    using namespace std;
    // auto t_start = chrono::high_resolution_clock::now();

    std::vector<std::vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };
        
        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    std::vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    // auto t_end = chrono::high_resolution_clock::now();
    // float total = chrono::duration<float, milli>(t_end - t_start).count();
    // cout << "Time taken for nms is " << total << " ms." << endl;
}

void YoloV3::DoInference(YoloInDataSt* input,int batchsize, std::vector<std::vector<Bbox>>& output) {

    mNet->CopyFromHostToDevice(input->data, 0);
    mNet->Forward();
    mNet->CopyFromDeviceToHost(mpDetCpu, 1);

	output.resize(batchsize);
	for (size_t i = 0; i < batchsize; i++)
	{
		int detCount = (int)mpDetCpu[factor*i];

		// std::cout << "detCount: " << detCount << std::endl;
		// for(int i=1;i<71;i++) {
		//     if((i-1)%6 == 0) {
		//         std::cout << std::endl;
		//     }
		//     std::cout << mpDetCpu[i] << " ";
		// }

		std::vector<Detection> result;
		result.resize(detCount);
		memcpy(result.data(), &mpDetCpu[factor * i+1], detCount * sizeof(Detection));

		//scale bbox to img
		int width = input->originalWidths[i];
		int height = input->originalHeights[i];
		float scale = std::min(float(mNetWidth) / width, float(mNetHeight) / height);
		float scaleSize[] = { width * scale,height * scale };

		//correct box
		for (auto& item : result)
		{
			auto& bbox = item.bbox;
			bbox[0] = (bbox[0] * mNetWidth - (mNetWidth - scaleSize[0]) / 2.f) / scaleSize[0];
			bbox[1] = (bbox[1] * mNetHeight - (mNetHeight - scaleSize[1]) / 2.f) / scaleSize[1];
			bbox[2] /= scaleSize[0];
			bbox[3] /= scaleSize[1];
		}
		DoNms(result,mYoloClassNum,0.5);
		output[i].resize(result.size());
		for (size_t j = 0; j < result.size(); j++)
		{
			auto& item = result[j];
			Bbox bbox;
			auto& b = item.bbox;
			bbox.left = std::max(int((b[0] - b[2] / 2.)*width), 0);
			bbox.right = std::min(int((b[0] + b[2] / 2.)*width), width);
			bbox.top = std::max(int((b[1] - b[3] / 2.)*height), 0);
			bbox.bottom = std::min(int((b[1] + b[3] / 2.)*height), height);
			bbox.score = item.prob;
			bbox.clsId = item.classId;
			output[i][j] = bbox;
		}
	}
}
