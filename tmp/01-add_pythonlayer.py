#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
add python layers
"""
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import os
import sys
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import caffe.proto.caffe_pb2 as caffe_pb2
import csslib

# add python layers before and after convolution layers
def add_python_layer(net):
    # 変更後用のネットワーク領域を作成
    net_dst = caffe_pb2.NetParameter()
    # 元ネットワークの各層を変換しながらコピー（元ネットワークは一部破壊される）
    layers = net.layer
    for layer in layers:
        if layer.type == 'Convolution':
            #print layer.name
            layer_top = layer.top[0]
            f2i_top = layer_top + '_f2i'
            i2f_top = layer_top + '_i2f'
            # conv層のbottomになる層をi2fに接続し直す
            for l in layers:
                #if (len(l.bottom) > 0) and (l.bottom[0] == layer_top):
                #   #print "reconnect", l.name
                #   l.bottom[0] = i2f_top
                if (len(l.bottom) > 0):
                    for index in range(len(l.bottom)):
                        if (l.bottom[index] == layer_top):
                            print "reconnect", l.name, "bottom[", index, "] : from", l.bottom[index], "to", i2f_top
                            l.bottom[index] = i2f_top
            # f2i層挿入
            f2i = net_dst.layer.add(name=f2i_top,     # f2i.name = f2i_topで直接代入でも可
                                    type='Python',    # f2i.type = 'Python'で直接代入でも可
                                    #bottom=layer.bottom[0],
                                    python_param={'module':'f2i', 
                                                  'layer':'F2I',
                                                  'param_str':'{"scale":1.0, "shift":0}'})
            f2i.bottom.append(layer.bottom[0])
            f2i.top.append(f2i_top)
            #print f2i 
            # conv層bottomをf2iに接続
            layer.bottom[0] = f2i_top
            net_dst.layer.extend([layer])
            # i2f層挿入
            i2f = net_dst.layer.add(name=i2f_top,
                                    type='Python', 
                                    python_param={'module':'i2f', 
                                                  'layer':'I2F', 
                                                  'param_str':'{"scale":1.0, "shift":0}'})
            i2f.bottom.append(layer_top)
            i2f.top.append(i2f_top)
        elif layer.type == 'Eltwise':
            # eltwise layerにeltwise_paramを追加
            weight = []
            for i in range(len(layer.bottom)):
                weight.append(1.0)                 # 各層のweight : とりあえず全部１
            ew_ws = net_dst.layer.add(name=layer.top[0],
                                      type='Eltwise',
                                      eltwise_param={'operation':'SUM',
                                                     'coeff':weight})  # weightはタプルで指定
            for i in range(len(layer.bottom)):
                ew_ws.bottom.append(layer.bottom[i])
            ew_ws.top.append(layer.top[0])
        else:
            net_dst.layer.extend([layer])
    return net_dst


##### main #####
if __name__ == '__main__':
    print '\n----- 01-add_python_layer.py -----', sys.argv[1], '->',sys.argv[2]
    model_in  = sys.argv[1]
    model_out = sys.argv[2]
    #
    net = csslib.load_pb2(model_in)
    net_out = add_python_layer(net)
    print 'saving modified prototxt to', model_out
    csslib.save_pb2(net_out, model_out)
