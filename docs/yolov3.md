<!--
 * @Author: zerollzeng
 * @Date: 2019-09-04 17:18:42
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-15 14:40:47
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

sample ouput
```
(base) root@70063c05b121:/tensorrt-zoo# ./bin/testyolov3 --prototxt models/yolov3/yolov3_416_trt.prototxt --caffemodel models/yolov3/yolov3_416.caffemodel --save_engine models/yolov3/yolov3_416_2080ti.engine --input test.jpg 
usage: ./testyolov3 --prototxt path/to/prototxt --caffemodel path/to/caffemodel/ --save_engine path/to/save_engin --input path/to/input/img
[2019-10-15 06:38:27.115] [info] create plugin factory
[2019-10-15 06:38:27.115] [info] yolo3 params: class: 80, netSize: 416 
[2019-10-15 06:38:27.115] [info] upsample params: scale: 2
[2019-10-15 06:38:27.115] [info] prototxt: ./models/yolov3/yolov3_416_trt.prototxt
[2019-10-15 06:38:27.115] [info] caffeModel: ./models/yolov3/yolov3_416.caffemodel
[2019-10-15 06:38:27.115] [info] engineFile: ./models/yolov3/yolov3_416_2080ti.engine
[2019-10-15 06:38:27.115] [info] outputBlobName: 
yolo-det 
[2019-10-15 06:38:27.115] [info] deserialize engine from ./models/yolov3/yolov3_416_2080ti.engine
TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.2
[2019-10-15 06:38:36.413] [info] max batch size of deserialized engine: 1
[2019-10-15 06:38:36.433] [info] create execute context and malloc device memory...
[2019-10-15 06:38:36.433] [info] init engine...
TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.2
[2019-10-15 06:38:36.435] [info] malloc device memory
nbBingdings: 2
[2019-10-15 06:38:36.435] [info] input: 
[2019-10-15 06:38:36.435] [info] binding bindIndex: 0, name: data, size in byte: 2076672
[2019-10-15 06:38:36.435] [info] binding dims with 3 dimemsion
3 x 416 x 416   
[2019-10-15 06:38:36.436] [info] output: 
[2019-10-15 06:38:36.436] [info] binding bindIndex: 1, name: yolo-det, size in byte: 255532
[2019-10-15 06:38:36.436] [info] binding dims with 3 dimemsion
63883 x 1 x 1   
Time over all layers: 0.000
[2019-10-15 06:38:36.480] [info] net forward takes 12.5075 ms
inference Time : 9.658 ms
[2019-10-15 06:38:36.481] [info] ------------------------
[2019-10-15 06:38:36.481] [info] object in 271,180,378,423
[2019-10-15 06:38:36.481] [info] object in 435,217,516,450
[2019-10-15 06:38:36.481] [info] object in 352,265,478,477
[2019-10-15 06:38:36.481] [info] object in 0,275,44,480
```