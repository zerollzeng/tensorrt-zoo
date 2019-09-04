<!--
 * @Author: zerollzeng
 * @Date: 2019-09-04 17:18:42
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-09-04 17:18:43
 -->
# Quick start with yolov3
## data prepare

Download the caffe model converted by official model:

+ Baidu Cloud [here](https://pan.baidu.com/s/1VBqEmUPN33XrAol3ScrVQA) pwd: gbue
+ Google Drive [here](https://drive.google.com/open?id=18OxNcRrDrCUmoAMgngJlhEglQ1Hqk_NJ)


If run model trained by yourself, comment the "upsample_param" blocks, and modify the prototxt the last layer as:
```
layer {
    #the bottoms are the yolo input layers
    bottom: "layer82-conv"
    bottom: "layer94-conv"
    bottom: "layer106-conv"
    top: "yolo-det"
    name: "yolo-det"
    type: "Yolo"
}
```
the model comes from [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3), if you want to canvert your own yolov3 model, you can use [BingzheWu/object_detetction_tools](https://github.com/BingzheWu/object_detetction_tools) with nn_model_transform, note that you need to compile a caffe with upsample layer, see [here](https://github.com/BVLC/caffe/pull/6384), if you have any question you can creat an issue :D

## build demo

```bash
git clone --recursive git@github.com:zerollzeng/tensorrt-zoo.git
cd tensorrt-zoo
mkdir build && cd build && cmake .. && make && cd ..
./bin/testyolov3 --prototxt=path_to_prototxt --caffemodel=path_to_caffemodel --save_engine=save_engine_name --input=test.jpg
# and now you can see result in result.jpg
```