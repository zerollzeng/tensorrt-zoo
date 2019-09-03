'''
@Date: 2019-08-29 10:28:33
@LastEditors: zerollzeng
@LastEditTime: 2019-09-03 18:45:31
'''
import sys
sys.path.append("../lib")
import pytrt
import numpy as np
from PIL import Image



def parse(prototxt_path,layer_types):
    with open(prototxt_path, 'r') as prototxt:
        type_dict = {}
        for type in layer_types:
            prototxt.seek(0)
            while(1):
                line = prototxt.readline()
                if not line:
                    break
                if(''.join(line.split()) == "layer{"):
                    layer_type = ''
                    layer_name = ''
                    while(1):
                        sub_line = prototxt.readline()
                        if(''.join(sub_line.split()) == "}"):
                            break
                        sub_line = ''.join(sub_line.split()).strip('"').replace('"','')
                        sub_list = sub_line.split(':')
                        if sub_list[0].lower() == 'type':
                            layer_type = sub_list[1]
                        if sub_list[0].lower() == 'name':
                            layer_name = sub_list[1]
                    if layer_type.lower() == type.lower():
                        type_dict.setdefault(type,[]).append(layer_name)
    return type_dict
                            

prototxt = "../models/openpose/pose_deploy.prototxt"
parse(prototxt,['prelu'])  




# img = Image.open("../test.jpg")

# img_arr = list(img.getdata())

# np_hwc = np.array(img,dtype=np.float32)
# np_chw = np.transpose(np_hwc,(2,0,1))

# np_chw = np_chw/255 - 0.5;


# trt = pytrt.Trt()
# pluginParams = pytrt.TrtPluginParams()
# prototxt = "../models/openpose/pose_deploy.prototxt"
# caffeModel = "../models/openpose/pose_iter_584000.caffemodel"
# engineFile = "../models/openpose/openpose.trt"
# outputBlobName = ["net_output"]
# calibratorData = [[]]
# maxBatchSize = 1
# mode = 0
# # trt.CreateEngine("../models/mobilenetv2-1.0/mobilenetv2-1.0.onnx","../models/mobilenetv2-1.0/mobilenetv2-1.0.trt",1)
# trt.CreateEngine(prototxt,caffeModel,engineFile,outputBlobName,calibratorData,maxBatchSize,mode)
# trt.DoInference(np_chw)
# heatmap = trt.GetOutput(1)

# for i in range(78):
#     img = heatmap[i]
#     img_min = img.min()
#     img_max = img.max()
#     img = (img-img_min)/(img_max-img_min)
#     Image.fromarray(np.uint8(img*255),'L').save(str(i)+".jpg")
