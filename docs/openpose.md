<!--
 * @Author: zerollzeng
 * @Date: 2019-10-16 16:45:46
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-17 11:36:11
 -->
# QUICK START

## preparetions

At first you need to download offcial body-25 openpose caffemodel from this [link](http://posefs1.perception.cs.cmu.edu/OpenPose/models/), and download prototxt from this [link](https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt)

then modify prototxt's inputdims to 1x3x480x640 for run test sample since tensorrt need a fix input dimemsion. you can use other w and h value too. but make sure it's multiple of 16.
```
name: "OpenPose - BODY_25"
input: "image"
input_dim: 1 # This value will be defined at runtime
input_dim: 3
input_dim: 480 # modify this for run sample
input_dim: 640 # modify this for run sample
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "image"
  top: "conv1_1"
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
```

after build successfully you can run openpose sample like
```bash
./bin/testopenpose --prototxt /path/to/prototxt --caffemodel /path/to/caffemodel --save_engine path/to/save_engine --input /path/to/input_image
```

and you should see output looks like below, and result in result.jpg
```
(base) root@70063c05b121:/tensorrt-zoo# ./bin/testopenpose --prototxt models/openpose/pose_deploy.prototxt --caffemodel models/openpose/pose_iter_584000.caffemodel --save_engine models/openpose/sample.engine --input ./test.jpg 
usage: path/to/testopenpose --prototxt path/to/prototxt --caffemodel path/to/caffemodel/ --save_engine path/to/save_engin --input path/to/input/img
[2019-10-16 09:39:38.481] [info] create plugin factory
[2019-10-16 09:39:38.481] [info] yolo3 params: class: 1, netSize: 416 
[2019-10-16 09:39:38.481] [info] upsample params: scale: 2
[2019-10-16 09:39:38.481] [info] prototxt: models/openpose/pose_deploy.prototxt
[2019-10-16 09:39:38.481] [info] caffeModel: models/openpose/pose_iter_584000.caffemodel
[2019-10-16 09:39:38.481] [info] engineFile: models/openpose/sample.engine
[2019-10-16 09:39:38.481] [info] outputBlobName: 
net_output 
[2019-10-16 09:39:38.481] [info] build caffe engine with models/openpose/pose_deploy.prototxt and models/openpose/pose_iter_584000.caffemodel
[2019-10-16 09:39:38.942] [info] Number of network layers: 261
[2019-10-16 09:39:38.942] [info] Number of input: 
Input layer: 
image : 3x480x640 
[2019-10-16 09:39:38.942] [info] Number of output: 1
Output layer: 
net_output : 78x60x80 
[2019-10-16 09:39:38.942] [info] parse network done
[2019-10-16 09:39:38.942] [info] fp16 support: true
[2019-10-16 09:39:38.942] [info] int8 support: true
[2019-10-16 09:39:38.942] [info] Max batchsize: 1
[2019-10-16 09:39:38.942] [info] Max workspace size: 10485760
[2019-10-16 09:39:38.942] [info] Number of DLA core: 0
[2019-10-16 09:39:38.942] [info] Max DLA batchsize: 268435456
[2019-10-16 09:39:38.942] [info] Current use DLA core: 0
[2019-10-16 09:39:38.942] [info] build engine...
TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.2
Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
Detected 1 inputs and 3 output network tensors.
TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.2
[2019-10-16 09:40:30.143] [info] serialize engine to models/openpose/sample.engine
[2019-10-16 09:40:30.143] [info] save engine to models/openpose/sample.engine...
[2019-10-16 09:40:43.500] [info] create execute context and malloc device memory...
[2019-10-16 09:40:43.500] [info] init engine...
TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.2
[2019-10-16 09:40:43.502] [info] malloc device memory
nbBingdings: 2
[2019-10-16 09:40:43.502] [info] input: 
[2019-10-16 09:40:43.503] [info] binding bindIndex: 0, name: image, size in byte: 3686400
[2019-10-16 09:40:43.503] [info] binding dims with 3 dimemsion
3 x 480 x 640   
[2019-10-16 09:40:43.504] [info] output: 
[2019-10-16 09:40:43.504] [info] binding bindIndex: 1, name: net_output, size in byte: 1497600
[2019-10-16 09:40:43.504] [info] binding dims with 3 dimemsion
78 x 60 x 80   
=====>malloc extra memory for openpose...
heatmap Dims3
heatmap size: 1 78 60 80
allocate heatmap host and divice memory done
resize map size: 1 78 240 320
kernel size: 1 78 240 320
allocate kernel host and device memory done
peaks size: 1 25 128 3
allocate peaks host and device memory done
=====> malloc extra memory done
[2019-10-16 09:40:43.547] [info] net forward takes 40.6057 ms
inference Time : 89.374 ms
```
