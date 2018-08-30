#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
層名中の'/'を'_'に変更
"""
import os
import sys
import numpy as np
import argparse
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import csslib

#caffe_root = '../../../../caffe/'
#sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

# ネットワークパラメータ形状表示
def show_net_params(net):
    for layer_name, param in net.params.iteritems():
        elements = len(param)
        print ("%40s" % layer_name) + '\t' + str(elements) + '\t',
        for index in range(0, elements):
            print param[index].data.shape, '\t',
        print ""

# ネットワークからパラメータ名のリストを作成
def gen_param_namelist(net, list):
    for layer_name, param in net.params.iteritems():
        list.append(layer_name)

##### main #####
if __name__ == '__main__':
    print '----- 01_rename_slash.py -----'
    caffe.set_mode_cpu()
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "model_def",
        help="CBS separated prototxt file (input)"
    )
    parser.add_argument(
        "model_weights",
        help="CBS separated caffemodel (input)"
    )
    parser.add_argument(
        "model_def_ns",
        help="CBS integrated prototxt file (input)"
    )
    parser.add_argument(
        "model_weights_ns",
        help="CBS integrated caffemodel (output)"
    )
    args = parser.parse_args()
    model_def        = args.model_def
    model_weights    = args.model_weights
    model_def_ns     = args.model_def_ns
    model_weights_ns = args.model_weights_ns

    # 手術前ネットワークの読み込み
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    #show_net_params(net)
    src_param_names = []
    gen_param_namelist(net, src_param_names)
    print src_param_names
    
    # 手術後ネットワークを作成
    net_ns = caffe.Net(model_def_ns,      # defines the structure of the model
                       model_weights,      # contains the trained weights
                       caffe.TEST)         # use test mode (e.g., don't perform dropout)
    #show_net_params(net_ns)
    #dst_param_names = []
    #gen_param_namelist(net_ns, dst_param_names)
    #print dst_param_names

    # 層名の変更リスト
    old_names = {"rpn_conv/3x3": "rpn_conv_3x3" }

    # 層名が変更された部分のパラメータをコピー
    for layer in src_param_names:
        #print layer
        if layer in old_names:
            new_name = old_names[layer]
            print layer, '->', new_name
            org_param = net.params[layer]
            ns_param = net_ns.params[new_name]
            #print len(org_param), len(ns_param)
            ns_param[0].data[...] = org_param[0].data        # copy original data
            ns_param[1].data[...] = org_param[1].data
    
    # 手術結果を保存
    #net.save('tmp.caffemodel')
    net_ns.save(model_weights_ns)
