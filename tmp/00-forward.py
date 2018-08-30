#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
execute network and dump result(fp32 binary) and statistics of network parameters/activations
"""
import os
import numpy as np
import sys
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import csslib
import caffe

np.set_printoptions(linewidth=200)

##### main #####
if __name__ == '__main__':
    #caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    model     = sys.argv[1]
    param     = sys.argv[2]
    image     = sys.argv[3]
    imgwidth  = int(sys.argv[4])
    imgheight = int(sys.argv[5])
    out_layer = sys.argv[6]
    scale     = sys.argv[7]
    nchannels = int(sys.argv[8])
    #scale_in  = float(sys.argv[8])
    print "\n----- 00-forward.py ----- model:", model, "param:", param, "image:", image, "imgwidth:", imgwidth, "imgheight", imgheight
    #
    net = csslib.load_network(model, param)                 # model, parameterからネットワークを作成
    net.blobs['data'].reshape(1, nchannels, imgheight, imgwidth)    # reshape to image size
    scale_dict = csslib.load_dict(scale)
    #
    transformer = csslib.gen_transformer(net)
    output = csslib.forward(net, image, transformer)        # 入力画像ファイルをnetで処理
    #output = csslib.forward_nt(net, image) #, transformer)        # 入力画像データ(binary)をnetで処理
    # 結果、統計値をファイルに出力
    #out_fp32 = output[out_layer].flatten() * (scale_dict[out_layer] * scale_in)
    out_fp32 = output[out_layer].flatten() * scale_dict[out_layer]
    csslib.save_ndarray_bin(out_fp32, 'out.bin')
    #print output[out_layer].flatten() * scale_dict[out_layer]
    print out_fp32
    csslib.net_param_statitics(net)             # network parameter統計値を stat_param.txt に出力
    csslib.net_act_statitics(net)               # activation統計値を stat_out.txt に出力
