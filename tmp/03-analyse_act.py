#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
analyse activation and compute activation scale
"""
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import os
import numpy as np
import sys
import glob
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import json
#import matplotlib.pyplot as plt
import copy
import csslib

# 
def gen_layer_namelist(net, list):
    for layer_name, blob in net.blobs.iteritems():
        list.append(layer_name)

# get activation
def get_act(net, layer):
    return net.blobs[layer].data

# array中の、値がlimit未満の要素数を返す
def count_lt(array, limit):
    return np.count_nonzero((array < limit) * 1)

# old:各convolution層のactivation分布で割合ratioとなる位置を調べ、スケーリング係数を決める
# 各convolution層のactivation分布で割合ratioとなる位置を調べる
def calc_act_scale():
    n = len(conv_layers)
    for i in range(n):
        print i, ':', conv_layers[i]
        data = blobs[conv_layers[i]].flatten()
        #print data.shape, np.min(data), np.max(data)
        data_abs = np.fabs(data)
        n_elements = data.shape[0]
        max_abs = np.max(data_abs)
        if pvalid >= 1.0:
            ratio = pvalid
        else:
            ratio_min = 0.0
            ratio_max = 1.0
            ratio = 0.5
            n_dest = n_elements * pvalid
            for loop in range(10):
                n = count_lt(data_abs, max_abs * ratio)
                #print loop, ratio, n
                if n < n_dest:
                    ratio_min = ratio
                else:
                    ratio_max = ratio
                ratio = (ratio_min + ratio_max) / 2
        # 変更前ネットワークでのスケーリング位置。
        # この値と変更前・変更後ネットワークのスケーリング係数から変更後ネットワークでのスケーリング位置を求める
        a_scale[conv_layers[i]] = max_abs * ratio
        print "calc_act_scale n_elements:", n_elements, "max_abs:", max_abs, "ratio:", ratio

# activationのスケールを入れてprototxtとcaffemodelを変更        
def modify_net_and_param():
    ascale = 1                          # dummy
    for layer in ptxt.layer:
        if len(layer.bottom) > 0:
            scale_prev = scale_dict_a[layer.bottom[0]]     # activationスケール後の入力scale
            scale_prev_old = scale_dict[layer.bottom[0]]   # activationスケール前の入力scale
        else:
            #scale_prev = 1.0
            scale_prev = 1.0 * in_norm_scale               # 入力のスケール係数(Power layerで入力データを{-1,1}に変換した分)
            scale_prev_old = 1.0
        # eltwiseではscale_prev_oldはlayer.bottom[0]側とは限らないが、
        # ここのscale_newは結果に関係ないので一応設定しておく
        if len(layer.top) > 0:
            scale_cur_old = scale_dict[layer.top[0]]           # activationスケール前のこの層のスケール
        else:
            scale_cur_old = 1.0           # activationスケール前のこの層のスケール
        scale_ratio = scale_cur_old / scale_prev_old       # この層のscale比率
        scale_new = scale_prev * scale_ratio               # activationスケール後のこの層のスケール
        print layer.name, "ascale:", ascale, "scale_new:", scale_new, "= scale_prev:", scale_prev, "* scale_ratio:", scale_ratio
        if layer.type == 'Convolution':
            # activation scale前ネットワークでの分布から決めたスケーリング位置を、activation scale後での位置に変換
            #  scale_cur_oldでyという値 -> オリジナルでの値：y * scale_cur_old
            #  -> scale_newでは (y * scale_cur_old) / scale_new になる
            #tmp = a_scale[layer.name] * scale_cur_old / scale_new  
            tmp = a_scale[layer.top[0]] * scale_cur_old / scale_new  
            # 実行時は ±1 * tmp の範囲を±1にスケーリングする
            # → スケーリングの係数は 1/tmp
            # 実行時に割り算が不要になるよう、1/tmp = c_scale/iscale のc_scaleをスケーリングのパラメータとする
            #ascale = int(iscale / tmp + 0.5)
            print "iscale:", iscale, "tmp", tmp
            ascale = int(iscale / tmp)
            #print "ascale:", ascale, "tmp", tmp, "scale_cur_old:", scale_cur_old, "a_scale:", a_scale[layer.name], "scale_new:", scale_new
            print "ascale:", ascale, "tmp", tmp, "scale_cur_old:", scale_cur_old, "a_scale:", a_scale[layer.top[0]], "scale_new:", scale_new
            # net_asパラメータ変更
            prm_as = net_as.params[layer.name]
            #prm_norm[0].data[...] = param[0].data / scale # weightは変更なし
            # Biasのスケール係数をactivationスケール変更前後の入力スケール比で補正する
            bias_mod = scale_prev_old / scale_prev
            prm_as[1].data[...] = prm_as[1].data * bias_mod  # 本当はbiasは(<< shift)で割る
            print "Bias modification:", bias_mod, "prev_old", scale_prev_old, "prev", scale_prev
            # 使わないけど、一応f2i層のpython_paramのscaleとshiftを変更しておく
            # mulの指定は廃止。prototxtにPower層を追加して調整する。
            prm_dict_str = '{ "scale":' + str(scale_new) + ', "shift":0 }'
            #if layer.name == 'conv_1_cbs':
            #    prm_dict_str = '{ "scale":' + str(scale_new) + ', "shift":0, "mul":2.0 }'  # 入力のスケール変更指示
            #else:
            #    prm_dict_str = '{ "scale":' + str(scale_new) + ', "shift":0 }'
            f2i.python_param.param_str = prm_dict_str
        elif (layer.type == 'Python') and (layer.python_param.layer == 'F2I'):
            f2i = layer                           # conv層のパラメータを設定するためF2I層を保持
        elif (layer.type == 'Python') and (layer.python_param.layer == 'I2F'):
            # i2f層のpython_paramにscaleとshiftを設定
            #prm_dict_str = '{ "scale":' + str(ascale) + ', "shift":' + str(int(np.log2(iscale))) + ' }'
            prm_dict_str = '{ "scale":' + str(ascale) + ', "shift":' + str(int(np.log2(iscale))) + ', "valid":1 }'
            layer.python_param.param_str = prm_dict_str
            scale_new *= float(iscale) / ascale   # この層でのアクティベーションスケール変更分を補正
        elif (layer.type == 'Eltwise'):
            #scale0 = scale_dict_a[layer.bottom[0]]
            #scale1 = scale_dict_a[layer.bottom[1]]
            #print "Eltwise: bottom0:", scale0, "bottom1:", scale1
            #scale_new = np.max((scale0, scale1))  # scaleの大きい側に小さい側を合わせる
            #layer.eltwise_param.coeff[0] = scale0 / scale_new
            #layer.eltwise_param.coeff[1] = scale1 / scale_new
            scale0 = scale_dict_a[layer.bottom[0]]
            scale1 = scale_dict_a[layer.bottom[1]]
            print "Eltwise: bottom0:", scale0, "bottom1:", scale1
            scale_new = np.max((scale0, scale1))  # scaleの大きい側に小さい側を合わせる
            scale0 = scale0 / scale_new
            scale1 = scale1 / scale_new
            # 2つのパスの最大値が加算された場合にオーバーフローしないようスケールを調整
            scale2 = scale0 + scale1
            scale0 = scale0 / scale2
            scale1 = scale1 / scale2
            scale_new = scale_new * scale2
            layer.eltwise_param.coeff[0] = scale0
            layer.eltwise_param.coeff[1] = scale1
        # この層出力のスケールを辞書に設定
        #scale_dict_a[layer.top[0]] = scale_new
        if len(layer.top) > 0:          # Silenceは len(layer.top) == 0
            for i in range(len(layer.top)):
                #print layer.name, layer.top[i], "scale_cur=", scale_cur
                scale_dict_a[layer.top[i]] = scale_new
        print layer.name, "scale_prev", scale_prev, "scale_new", scale_new
    #
    print scale_dict
    print scale_dict_a

##### main #####
if __name__ == '__main__':
    print '\n----- 03-analyse_act.py -----', sys.argv[1], sys.argv[2]
    #caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    model = sys.argv[1]
    param = sys.argv[2]
    imagedir = sys.argv[3] 
    imgwidth = int(sys.argv[4])
    imgheight = int(sys.argv[5])
    out_layer = sys.argv[6]
    scale_dict_file = sys.argv[7]
    nimages = int(sys.argv[8])
    in_norm_scale = float(sys.argv[9])
    nchannels = int(sys.argv[10])
    iscale = 256
    #iscale = 32768
    #pvalid = 1.2        # pvalid>1.0 : activationスケールに余裕をもたせる
    pvalid = 1.5        # pvalid>1.0 : activationスケールに余裕をもたせる
    #
    tmp = model.split('.')
    model_out = tmp[0] + 'a.' + tmp[1]
    tmp = param.split('.')
    param_out = tmp[0] + 'a.' + tmp[1]
    tmp = scale_dict_file.split('.')
    scale_dict_file_out = tmp[0] + 'a.' + tmp[1]
    print "model     :", model, "->", model_out
    print "param     :", param, "->", param_out
    print "scale_dict:", scale_dict_file, "->", scale_dict_file_out
    print "in_norm_scale:", in_norm_scale
    #
    imagefile = []
    if nimages == 1:
        imagefile.append(imagedir)
    elif nimages == 0:
        #imagefile = glob.glob(imagedir + '/*.png')[:3]
        #imagefile = glob.glob(imagedir + '/*.png')      # 全画像ではメモリを食い尽くす...
        imagefile = glob.glob(imagedir + '/*.png')[:50]
    else:
        for i in range(nimages):
            imagefile.append('testdata/img_' + ('0000' + str(i))[-4:] + '.jpg')
    print 'found', len(imagefile), 'imagefiles :', imagefile[:3], '...'
    # scale辞書の読み込み
    scale_dict = json.load(file(scale_dict_file))  # activation scaleを入れる前のスケール辞書
    scale_dict_a = copy.deepcopy(scale_dict)       # activation scaleを入れたスケール辞書(初期値は変更前と同じ)
    
    ptxt   = csslib.load_pb2(model)                  # 変換前prototxt
    net    = csslib.load_network(model, param)       # 変換前ネットワーク
    net_as = csslib.load_network(model, param)       # 変換後ネットワーク（変換前を初期値とする）
    net.blobs['data'].reshape(1, nchannels, imgheight, imgwidth)    # reshape to image size
    transformer = csslib.gen_transformer(net)        # 入力データのtransformerを作成
    #
    conv_layers = []
    for layer in ptxt.layer:
        if layer.type == 'Convolution':
            #conv_layers.append(layer.name)          # nameではなく
            conv_layers.append(layer.top[0])         # topを登録する
    print "conv_layers:", conv_layers
    #
    layer_names = []
    gen_layer_namelist(net, layer_names)
    print 'layer_names:', layer_names

    # 各層のアクティベーションサイズを取得
    actsize = {}
    #out = csslib.forward_nt(net, imagefile[0]) #, transformer)
    out = csslib.forward(net, imagefile[0], transformer)
    for layer in layer_names:
        act = get_act(net, layer)
        actsize[layer] = act.flatten().shape[0]
    #print "activation size:", actsize
    # 全画像を処理してactivation値を取得
    blobs = {}
    nimages = len(imagefile)
    print "nimages:", nimages
    for layer in layer_names:
        blobs[layer] = np.zeros((nimages, actsize[layer]), dtype=float)
    print "execute", len(imagefile), "images"
    for i in range(nimages):
        print imagefile[i]
        #out = csslib.forward_nt(net, imagefile[i]) #, transformer)
        out = csslib.forward(net, imagefile[i], transformer)
        for layer in layer_names:
            blobs[layer][i][...] = get_act(net, layer).flatten()
    # activation scaleを計算
    a_scale = {}
    calc_act_scale()
    print "a_scale:", a_scale

    # アクティベーションのスケールを入れてネットワークスケールを更新する
    modify_net_and_param()
    # 結果を出力
    net_as.save(param_out)                                  # caffemodel
    csslib.save_pb2(ptxt, model_out)                        # prototxt
    json.dump(scale_dict_a, file(scale_dict_file_out, 'w')) # scale辞書
