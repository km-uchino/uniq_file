#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Power層による入力の補正に合わせてconv層のbiasを補正する
"""
import os
import numpy as np
import argparse

# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import csslib

# conv層のbiasをscaling
def scale_bias(ptxt, net, net_scale, scale):
    for layer in ptxt.layer:
        if layer.type == 'Convolution':
            #print layer.name
            param = net.params[layer.name]                    # 変更前ネットワークパラメータ
            prm_norm = net_scale.params[layer.name]            # 変更後ネットワークパラメータ領域
            prm_norm[1].data[...] = param[1].data / scale
    return net_scale

##### main #####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "model_file",
        help="prototxt file"
    )
    parser.add_argument(
        "param_file",
        help="caffemodel file"
    )
    # Optional arguments.
    parser.add_argument(
        "--scale",
        default='128',
        help="scale"
    )
    args = parser.parse_args()
    #
    model = args.model_file #sys.argv[1]
    param = args.param_file #sys.argv[2]
    scale  = int(args.scale)  #int(sys.argv[3])
    print '\n----- 00-scale_bias.py -----', "model:", model, "param:", param, "scale:", scale
    #
    tmp = param.split('.')
    param_out = tmp[0] + 'b.' + tmp[1]
    print "param:", param, "->", param_out
    ptxt     = csslib.load_pb2(model)                  # 変換前prototxt
    net      = csslib.load_network(model, param)       # 変換前ネットワーク
    net_scale = csslib.load_network(model, param)      # 変換後ネットワーク（変換前を初期値とする）
    
    net_scale = scale_bias(ptxt, net, net_scale, scale)   # scale_bias
    #print 'saving un-modified caffemodel to tmp.caffemodel'
    #net.save('tmp.caffemodel')             # 変更前ネットワークパラメータを保存
    print 'saving modified caffemodel to', param_out
    net_scale.save(param_out)             # scale_biasしたネットワークパラメータを保存
