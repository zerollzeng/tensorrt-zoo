'''
@Date: 2019-08-29 10:28:33
@LastEditors: zerollzeng
@LastEditTime: 2019-08-29 15:09:56
'''
import sys
sys.path.append("../lib")
import pytrt
import numpy as np
from PIL import Image

img = Image.open("../test.jpg")

img_arr = list(img.getdata())

img_rgb = []

for i in range(3):
    for pixel in img_arr:
        img_rgb.append(pixel[i])

img_rgb = [float(i)/255-0.5 for i in img_rgb]
heatmap = []

trt = pytrt.Trt()
prototxt = "../models/openpose/pose_deploy.prototxt"
caffeModel = "../models/openpose/pose_iter_584000.caffemodel"
engineFile = "../models/openpose/openpose.trt"
outputBlobName = ["net_output"]
calibratorData = [[]]
maxBatchSize = 1
mode = 0
# trt.CreateEngine("../models/mobilenetv2-1.0/mobilenetv2-1.0.onnx","../models/mobilenetv2-1.0/mobilenetv2-1.0.trt",1)
trt.CreateEngine(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize,mode)
trt.DataTransfer(img_rgb,0,True)
trt.Forward()
trt.DataTransfer(heatmap,1,False)

print(heatmap)
print(len(heatmap))
# for x in range(10):
#     trt.Forward()