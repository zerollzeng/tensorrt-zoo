<!--
 * @Author: zerollzeng
 * @Date: 2019-09-02 16:45:43
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-09-04 17:35:23
 -->
# tensorrt-zoo
common computer vision models and some useful tools base on [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).
since most of models have complicated pre-processing and post-processing phase, so new model fully supports is time-consuming. so if you have any suggestions, please create an issue :yum::yum::yum::yum::yum:

# Roadmap
- [x] yolov3
- [x] network visualization - it's fascinating :clap::clap::clap:
- [ ] find some interesting things can done with tiny tensorrt :dancer::dancer::dancer:

# System requirements
CUDA version >= 9.0 is fully test. 8.0 might also work but didn't test.

TensorRT 5.1.5.0

OPENCV version >= 3.0, only use for read/write images and some basic image processing. so version 2.x might work but didn't test'.

# Quick Start with python and openpose
I am gonna show you how to use tiny-tensorrt to visualize a model layer activation, hope it will give you an intuitive understanding of tiny-tensorrt and convolutional neural network.

you need to compile the project at first
```bash
# make sure you install cuda and tensorrt 5.x, opencv is not necessary for this sample.
# you can just comment relative line in CMakeLists.txt if you don't want to run yolov3
git clone --recursive https://github.com/zerollzeng/tensorrt-zoo
mkdir build
cd build && cmake .. && make && cd ..
```

then you need to prepare caffe model for openpose and install some python packages
```bash
# download model, you can also download it via browser
wget -P ./models/openpose http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
wget -P ./models/openpose https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
cd activation-visualization
pip install -r requirements.txt
```

we can run sample now
```bash
python vis.py
```

and you can get output activation in activation folder, it look like this

![image](https://user-images.githubusercontent.com/38289304/64239641-299dcd00-cf33-11e9-9c75-051fa0c5c13f.png)

you can see details activation images in those sub-folders. like this feature map in first convolution layer

![0](https://user-images.githubusercontent.com/38289304/64239906-a761d880-cf33-11e9-8005-542dee105dbc.jpg)

and it's just a channel of the fisrt convolution layer, while the other looks like

![image](https://user-images.githubusercontent.com/38289304/64240061-dd06c180-cf33-11e9-9e27-49a369ad62cd.png)

if you see other activation map, you might see some activation in the middle of the model layers like Mconv3_stage0_L2_1, you know this layer get some keypoints and pose informaion.

![image](https://user-images.githubusercontent.com/38289304/64241470-64553480-cf36-11e9-9142-b3199fa8e7d9.png)

or activation near the end of the model which looks like Mconv6_stage1_L2, which output activation is very close to pose and keypoints.

![image](https://user-images.githubusercontent.com/38289304/64241656-bac27300-cf36-11e9-886e-5687136e1e73.png)

you can test with your own model, maybe need some change in vis.py because of different pre-processing. see
```bash
python vis.py --help
```

# yolov2 sample
see README.md in yolov3


