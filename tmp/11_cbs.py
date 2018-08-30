#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
in-place処理となっている BN,Scale,Relu層を独立した層に変更
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

##### main #####
if __name__ == '__main__':
    print '----- 01_cbs.py -----'
    caffe.set_mode_cpu()
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "model_in",
        help="input prototxt file"
    )
    parser.add_argument(
        "model_out",
        help="output prototxt file"
    )
    args = parser.parse_args()
     
    # 変換前ネットワークの読み込み
    model_in  = args.model_in
    model_out = args.model_out
    print "model_in:", model_in, "model_out:", model_out
    net = caffe_pb2.NetParameter()
    txtf.Merge(open(model_in).read(), net)
    
    # eltwise対応版
    #net.name = 'my new net'
    del_list = []
    layerNames = [l.name for l in net.layer]
    for lname in layerNames:
        if lname.startswith('conv'):
            idx = layerNames.index(lname)
            #print "found convolution:", lname, "index=", idx
            l = net.layer[idx]
            if layerNames[idx + 1].startswith('bn') and layerNames[idx + 2].startswith('scale'):
                conv = net.layer[idx]
                bn = net.layer[idx + 1]      # conv->bn->scaleは連続しているという前提
                scale = net.layer[idx + 2]
                #print conv
                print "found conv->bn->scale :", conv.name, bn.name, scale.name
                #print type(conv.name), type(conv.top), type(conv.bottom)
                conv.name = conv.name + '_cbs'            # conv層のlayer名に'_cbs'を付加
                del conv.top[:]                           # topはrepeatedなので一旦削除
                conv.top.append(conv.name)                # top名をlayer名と同じにする
                #print "modified :", conv.name, conv.top
                for lname2 in layerNames:                 # 各層をチェックして
                    idx2 = layerNames.index(lname2)
                    #print "idx2:", idx2, "net.layer[idx2].bottom:", net.layer[idx2].bottom
                    if len(net.layer[idx2].bottom) > 0:
                        #print net.layer[idx2].bottom
                        for bidx in range(len(net.layer[idx2].bottom)):
                            if (net.layer[idx2].bottom[bidx] == scale.top[0]):   # scale層をbottomとする層は
                                print " change layer", lname2, "bottom", bidx, "top :", scale.top[0], "->", conv.top[0]
                                del net.layer[idx2].bottom[bidx]
                                net.layer[idx2].bottom.append(conv.name)  # bottomをconv層に変更
                # あとでBN,Scale層を削除するため、インデックスを保存
                del_list.append(idx + 1)
                del_list.append(idx + 2)
                                
    #print "del_list:", del_list
    del_list.sort(reverse=True)
    print "del_list:", del_list
    for i in del_list:
        del net.layer[i]
    print [l.name for l in net.layer]

    print 'writing', model_out
    with open(model_out, 'w') as f:
        f.write(str(net))
