'''
@Author: zerollzeng
@Date: 2019-08-29 17:20:51
@LastEditors: zerollzeng
@LastEditTime: 2019-08-29 17:23:36
'''
import sys
sys.path.append("../lib")
import pytrt


trt = pytrt.Trt()
trt.CreateEngine("../models/bvlc_alexnet/model.onnx","../models/bvlc_alexnet/alexnet.trt",1)
for i in range(100):
    trt.Forward()