#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Conv/BN/Scaleパラメータを統合したcaffemodelを作成
"""
import os
import sys
import numpy as np
import argparse
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import csslib

caffe_root = '../../../../caffe/'
sys.path.insert(0, caffe_root + 'python')
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
    print '----- 02_cbs.py -----'
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
        "model_def_cbs",
        help="CBS integrated prototxt file (input)"
    )
    parser.add_argument(
        "model_weights_cbs",
        help="CBS integrated caffemodel (output)"
    )
    args = parser.parse_args()
    model_def     = args.model_def
    model_weights = args.model_weights
    model_def_cbs     = args.model_def_cbs
    model_weights_cbs = args.model_weights_cbs

    # 手術前ネットワークの読み込み
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    show_net_params(net)
    src_param_names = []
    gen_param_namelist(net, src_param_names)
    print src_param_names
    
    # 手術後ネットワークを作成
    net_cbs = caffe.Net(model_def_cbs,      # defines the structure of the model
                        model_weights,      # contains the trained weights
                        caffe.TEST)         # use test mode (e.g., don't perform dropout)
    show_net_params(net_cbs)
    dst_param_names = []
    gen_param_namelist(net_cbs, dst_param_names)
    print dst_param_names

    # BatchNormとScaleを統合したconvパラメータを計算
    for layer in src_param_names:
        if layer.startswith('conv'):
            print "found convolution layer:", layer
            #print "len:", len(net.params[layer])
            print "conv param shape weight:", net.params[layer][0].data.shape, "bias:", net.params[layer][1].data.shape
            conv_index = src_param_names.index(layer)
            conv_param  = net.params[layer]
            bn_param    = net.params[src_param_names[conv_index + 1]]
            scale_param = net.params[src_param_names[conv_index + 2]]
            epsilon = 0.00001
            #print "mean:", bn_param[0].data
            #print "variance:", bn_param[1].data
            #print "scale_factor:", bn_param[2].data
            #print "gamma:", scale_param[0].data
            #print "beta:", scale_param[1].data
            # a = γ / sqrt(Variance / scale + ε)
            a = scale_param[0].data / np.sqrt(bn_param[1].data / bn_param[2].data[0] + epsilon)
            b = scale_param[1].data - a * bn_param[0].data / bn_param[2].data[0]
            #print "a[:, np.newaxis].shape", a[np.newaxis, np.newaxis, np.newaxis, :].shape
            #print "a:", a[np.newaxis, np.newaxis, np.newaxis, :]
            #print "b:", b
            #print conv_param[0].data
            #
            cbs_index = dst_param_names.index(layer + '_cbs')
            print "cbs_index:", cbs_index
            cbs_param = net_cbs.params[layer + '_cbs']
            cbs_param[0].data[...] = conv_param[0].data        # copy original data
            cbs_param[1].data[...] = conv_param[1].data
            for och in range(0, cbs_param[0].data.shape[0]):
                cbs_param[0].data[och] *= a[och]                # K'[M][N] = K[M][N] * a[M]
            #print cbs_param[0].data
            cbs_param[1].data[...] = cbs_param[1].data * a + b  # Bias'[M] = Bias[M] * a[M] + b[M]
            #print cbs_param[1].data

    # 手術結果を保存
    net_cbs.save(model_weights_cbs)
