<!--
 * @Author: zerollzeng
 * @Date: 2019-09-02 16:45:43
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-10 17:41:21
 -->
# tensorrt-zoo
common computer vision models and some useful tools base on [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

since most of models have complicated pre-processing and post-processing phase, so new model fully supports is time-consuming. so if you have any suggestions, please create an issue :yum::yum::yum::yum::yum:

# Roadmap
- [x] openpose :fire::fire::fire: --- 2019.10.17
- [x] yolov3
- [ ] find some interesting things can done with tiny tensorrt :dancer::dancer::dancer:

# Quick Start
for run sample you need to install opencv and TensorRT

```bash
mkdir build && cd build && cmake -D PYTHON_API ON .. && make
```
for yolo3 sample see docs/yolov3.md
for openpose sample see docs/openpose.md


# System requirements
CUDA version >= 10.0 is fully test

TensorRT 6.0+, 5.1.5 is in other branch

OPENCV version >= 3.0, only use for read/write images and some basic image processing. so version 2.x might work but have not been tested.

# About License
most of the openpose post processing was copy and modify slightly from official openpose [repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose),maybe you should distribute it under openpose's [license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE), so as for some other code. for the other I wrote, licence is the same as [darknet's](https://github.com/pjreddie/darknet) [license](https://github.com/pjreddie/darknet/blob/master/LICENSE.fuck), and to make it simple and clear, You just DO WHAT THE FUCK YOU WANT TO.


