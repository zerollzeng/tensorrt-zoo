'''
@Date: 2019-08-29 10:28:33
@LastEditors: zerollzeng
@LastEditTime: 2019-09-02 19:36:08
'''
import sys
sys.path.append("../lib")
import pytrt
import numpy as np
from PIL import Image

img = Image.open("../test.jpg")

img_arr = list(img.getdata())

np_hwc = np.array(img,dtype=np.float32)
np_chw = np.transpose(np_hwc,(2,0,1))

np_chw = np_chw/255 - 0.5;

# print(np_chw)



# for i in range(3):
#     for pixel in img_arr:
#         img_rgb.append(pixel[i])

# img_rgb = [float(i)/255-0.5 for i in img_rgb]
# heatmap = []

trt = pytrt.Trt()
pluginParams = pytrt.TrtPluginParams()
prototxt = "../models/openpose/pose_deploy.prototxt"
caffeModel = "../models/openpose/pose_iter_584000.caffemodel"
engineFile = "../models/openpose/openpose.trt"
outputBlobName = ["net_output"]
calibratorData = [[]]
maxBatchSize = 1
mode = 0
# trt.CreateEngine("../models/mobilenetv2-1.0/mobilenetv2-1.0.onnx","../models/mobilenetv2-1.0/mobilenetv2-1.0.trt",1)
trt.CreateEngine(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize,mode)
# trt.DataTransfer(img_rgb,0,True)
# trt.Forward()
# trt.DataTransfer(heatmap,1,False)
trt.DoInference(np_chw)
heatmap = trt.GetOutput(1)

for i in range(78):
    img = heatmap[i]
    img_min = img.min()
    img_max = img.max()
    img = (img-img_min)/(img_max-img_min)
    Image.fromarray(np.uint8(img*255),'L').save(str(i)+".jpg")

# print(len(heatmap))
# for x in range(10):
#     trt.Forward()