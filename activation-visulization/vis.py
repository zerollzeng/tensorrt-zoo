'''
@Date: 2019-08-29 10:28:33
@LastEditors: zerollzeng
@LastEditTime: 2019-09-04 14:45:58
'''
import sys
import os
import shutil
from argparse import ArgumentParser
sys.path.append("../lib")
import pytrt
import numpy as np
from PIL import Image


'''
@description: get all layer name of some specific layer type in prototxt
@prototxt_path path to prototxt file
@layer_type layer tpye in prototxt such as 'convolution'
@return: list of all layer name with specific type
'''
def parse(prototxt_path,layer_types):
    with open(prototxt_path, 'r') as prototxt:
        blob_list = []
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
                        blob_list.append(layer_name)
    return blob_list



if __name__ == '__main__':
    command_parser = ArgumentParser()
    command_parser.add_argument('-i','--input',help='path to input image',default='../test.jpg')
    command_parser.add_argument('--prototxt',help='caffe prototxt', default='../models/openpose/pose_deploy.prototxt')
    command_parser.add_argument('--caffemodel',help='caffe weights file', default='../models/openpose/pose_iter_584000.caffemodel')
    command_parser.add_argument('--engine_file',help='save engine file path',default='../models/openpose/openpose.trt')
    command_parser.add_argument('--extra_blob',type=list,help='other blob you want to do visualization')
    command_parser.add_argument('--mark_type',type=list,default=['convolution'],help='mark output type')
    command_parser.add_argument('--normalize_factor',default=255,help='network_input = input/normalize_factor + normalize_bias')
    command_parser.add_argument('--normalize_bias',default=-0.5,help='network_input = input/normalize_factor + normalize_bias')
    command_parser.add_argument('--activation_save_path',default='./activation',help='activation save dir')

    args = command_parser.parse_args()

    img = Image.open(args.input)
    np_hwc = np.array(img,dtype=np.float32)
    np_chw = np.transpose(np_hwc,(2,0,1))
    np_chw = np_chw/args.normalize_factor - args.normalize_bias;

    trt = pytrt.Trt()
    pluginParams = pytrt.TrtPluginParams()
    outputBlobName = parse(args.prototxt,args.mark_type)
    calibratorData = [[]]
    maxBatchSize = 1
    mode = 0
    # trt.CreateEngine("../models/mobilenetv2-1.0/mobilenetv2-1.0.onnx","../models/mobilenetv2-1.0/mobilenetv2-1.0.trt",1)
    trt.CreateEngine(args.prototxt,args.caffemodel,args.engine_file,outputBlobName,calibratorData,maxBatchSize,mode)
    trt.DoInference(np_chw)

    activation_save_path = args.activation_save_path
    if os.path.exists(activation_save_path):
        shutil.rmtree(activation_save_path)
    os.mkdir(activation_save_path)

    for blob in outputBlobName:
        print("save blob: {}",blob)
        output = trt.GetOutput(blob)
        if output.size == 0:
            continue
        num_channel = output.shape[0]
        os.mkdir(os.path.join(activation_save_path,blob))
        for i in range(num_channel):
            img = output[i]
            img_min = img.min()
            img_max = img.max()
            img = (img-img_min)/(img_max-img_min+0.000001)
            path = os.path.join(activation_save_path,blob,str(i)+'.jpg')
            Image.fromarray(np.uint8(img*255),'L').save(path)
