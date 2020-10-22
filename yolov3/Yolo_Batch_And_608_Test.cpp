/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 14:50:04
 * @LastEditTime: 2020-11-5 10:30:43
 * @LastEditors: https://github.com/ZHEQIUSHUI
 */
 
#include <iostream>
#include <opencv2/opencv.hpp>
#include "YoloV3.h"
#ifdef _DEBUG
#pragma comment(lib,"libtinytrtd.lib")
#pragma comment(lib,"opencv_world410d.lib")
#else
#pragma comment(lib,"libtinytrt.lib")
#pragma comment(lib,"opencv_world410.lib")
#endif
#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cudart.lib")
using namespace cv;
using namespace std;

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

int main(int argc, char** argv)
{
	std::cout << "usage: path/to/testyolov3 --prototxt path/to/prototxt --caffemodel path/to/caffemodel/ --save_engine path/to/save_engin --input path/to/input/img" << std::endl;
    InputParser cmdparams(argc, argv);
    const std::string& prototxt = cmdparams.getCmdOption("--prototxt");
    const std::string& caffemodel = cmdparams.getCmdOption("--caffemodel");
    const std::string& save_engine = cmdparams.getCmdOption("--save_engine");
    const std::string& img_name = cmdparams.getCmdOption("--input");

	int yoloClassNum = 80;
	int yolo_netSize = 416;//or 608
	std::vector<std::string> yolo_outputBlobname{ "yolo-det" };
	std::vector<std::vector<float>> yolo_calibratorData;
	yolo_calibratorData.resize(3);
	for (size_t i = 0; i < yolo_calibratorData.size(); i++) {
		yolo_calibratorData[i].resize(3 * yolo_netSize * yolo_netSize);
		for (size_t j = 0; j < yolo_calibratorData[i].size(); j++) {
			yolo_calibratorData[i][j] = 0.05;
		}
	}
	int batchsize = 2;
	YoloV3 yolo(prototxt, caffemodel, save_engine, yolo_outputBlobname, yolo_calibratorData, batchsize, /*mode*/1,/*device*/ 0, yoloClassNum, yolo_netSize);
	Mat src = imread(img_name);
	Mat src1 = imread(img_name);
	vector<Mat> imgs = { src,src1 };
	YoloInDataSt yolo_input;
	yolo_input.data.resize(batchsize * 3 * yolo_netSize*yolo_netSize);
	yolo_input.originalHeights.resize(batchsize);
	yolo_input.originalWidths.resize(batchsize);

	float* data = yolo_input.data.data();
	int channelLength = yolo_netSize * yolo_netSize;
	for (size_t j = 0; j < imgs.size(); j++)
	{
		Mat img = imgs[j];
		cv::Mat yolo_img;
		float scale = std::min(float(yolo_netSize) / img.cols, float(yolo_netSize) / img.rows);
		auto scaleSize = cv::Size(int(img.cols * scale), int(img.rows * scale));
		cv::resize(img, yolo_img, scaleSize, 0, 0, cv::INTER_CUBIC);
		switch (yolo_img.channels())
		{
		case 1:
			cv::cvtColor(yolo_img, yolo_img, cv::COLOR_GRAY2RGB);
			break;
		case 4:
			cv::cvtColor(yolo_img, yolo_img, cv::COLOR_BGRA2RGB);
			break;
		case 3:
			cv::cvtColor(yolo_img, yolo_img, cv::COLOR_BGR2RGB);
			break;
		default:
			break;
		}
		cv::Mat cropped(yolo_netSize, yolo_netSize, CV_8UC3, 127);
		cv::Rect rect((yolo_netSize - scaleSize.width) / 2, (yolo_netSize - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
		yolo_img.copyTo(cropped(rect));
		//imshow("crop", cropped);
		cropped.convertTo(cropped, CV_32FC3, 1 / 255.0);
		std::vector<cv::Mat> input_channels(3);
		cv::split(cropped, input_channels);
		yolo_input.originalWidths[j] = img.cols;
		yolo_input.originalHeights[j] = img.rows;

		for (int i = 0; i < 3; ++i) {
			memcpy(data, input_channels[i].data, channelLength * sizeof(float));
			data += channelLength;
		}
	}

	std::vector<std::vector<Bbox>> yolo_output;

	while (1)
	{
		chrono::steady_clock::time_point time_start = chrono::steady_clock::now();
		yolo.DoInference(&yolo_input, batchsize, yolo_output);
		chrono::steady_clock::time_point time_end = chrono::steady_clock::now();
		chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(time_end - time_start);
		cout << "time use:" << time_used.count() << "s" << endl;
	}
	
	for (size_t i = 0; i < yolo_output.size(); i++)
	{
		auto sub_result = yolo_output[i];
		for (size_t j = 0; j < sub_result.size(); j++)
		{
			Bbox bbox = sub_result[j];
			cv::rectangle(imgs[i], cv::Point(bbox.left, bbox.top), cv::Point(bbox.right, bbox.bottom), cv::Scalar(0, 255, 0), 2);
		}
	}
	imwrite("./result.jpg",imgs[0]);
}
