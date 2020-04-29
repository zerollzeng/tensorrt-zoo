<!--
 * @Author: zerollzeng
 * @Date: 2019-09-02 16:45:43
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-10 17:41:21
 -->
# tensorrt-zoo
common computer vision models and some useful tools base on [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

**抱歉各位我现在没有太多精力维护这个项目，希望各位在提issue的时候能够尽量根据模板把信息补充完整，只是简单说明或者贴个失败的图我也无法定位问题**

# Roadmap
- [x] openpose :fire::fire::fire: --- 2019.10.17
- [x] yolov3
- [ ] find some interesting things can done with tiny tensorrt :dancer::dancer::dancer:

# Quick Start
for run sample you need to install opencv and TensorRT

```bash
git clone --recursive https://github.com/zerollzeng/tensorrt-zoo.git
mkdir build && cd build && cmake -D PYTHON_API ON .. && make
```
for yolo3 sample see docs/yolov3.md
for openpose sample see docs/openpose.md


# System requirements
CUDA version >= 10.0 is fully test

TensorRT 6.0+, 5.x you need to reset old commit(not recommend)

OPENCV version >= 3.0, only use for read/write images and some basic image processing. so version 2.x might work but have not been tested.

# About License
most of the openpose post processing was copy and modify slightly from official openpose [repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose),maybe you should distribute it under openpose's [license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE), so as for some other code. for the other I wrote, You just DO WHAT YOU WANT TO.


