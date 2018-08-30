#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
normalize and quantize kernel parameters
"""
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import os
import numpy as np
import argparse
# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2' 
import csslib

scale_dict = {}    # 各層出力のスケール値
scale_prev = 1.0   # 前層のscale

# a. Weight,Biasの絶対値の最大をscaleとする
# b. Biasのほうが大きい場合、精度が落ちるので、Weightの最大値をscaleとする
#    Weightで正規化し、Biasを足す前にWeightの演算結果をビットシフトで調整する
#    Biasの値がWeightに対して非常に大きいということはないはず
def filter_get_scale(net, name):
    # the parameters are a list of [weights, biases]
    weights = net.params[name][0].data
    biases  = net.params[name][1].data
    w_max = np.fabs(weights.flatten()).max()
    b_max = np.fabs(biases.flatten()).max()
    shift = np.fmax(np.ceil(np.log(b_max / w_max)), 0.0)
    #print "w_max:", w_max, "b_max:", b_max, "shift:", shift
    return w_max, shift

# arrayの各要素の+/-1の範囲を正負各qmax個に量子化し+/-1の範囲にクリップする
#  -1.0  -  0  -  1.0
#  -127  -  0  -  127
def quantize(array, qmax):
    tmp = np.ma.round(array * qmax) / qmax
    return tmp

def quantize_clip(array, qmax):
    tmp = quantize(array, qmax)
    return tmp.clip(-1.0, 1.0)

# conv層パラメータをnormalize
def normalize(ptxt, net, net_norm):
    global scale_prev
    for layer in ptxt.layer:
        print layer.name, "scale_prev=", scale_prev
        if len(layer.bottom) > 0:
            scale_prev = scale_dict[layer.bottom[0]]     # 入力のscale
        scale_cur = scale_prev                           # default:前層と同じ
        if layer.type == 'Convolution':
            # net_normパラメータを正規化、量子化。 Biasのスケール係数は前段スケールとの積
            param = net.params[layer.name]                    # 変更前ネットワークパラメータ
            prm_norm = net_norm.params[layer.name]            # 変更後ネットワークパラメータ領域
            scale, shift = filter_get_scale(net, layer.name)  # 当段のスケール係数
            if b_no_normalize:                                
                if b_no_quantize:                             # 正規化も量子化もなし
                    pass                                      
                else:                                         # 量子化のみ
                    # quantize()は定義域が+/-1なので、一旦正規化後、正規化分を戻す
                    tmp = scale_prev * scale
                    prm_norm[0].data[...] = quantize_clip(param[0].data / scale, qmax) * scale
                    prm_norm[1].data[...] = quantize(param[1].data / tmp, qmax) * tmp
            else:
                scale_cur = scale_prev * scale
                print layer.name, "scale:", scale, "shift:", shift, "scale_cur:", scale_cur
                if b_no_quantize:                             # 正規化のみ
                    prm_norm[0].data[...] = param[0].data / scale
                    prm_norm[1].data[...] = param[1].data / scale_cur  # 本当はbiasは(<< shift)で割る
                else:                                         # 正規化 -> 量子化
                    prm_norm[0].data[...] = quantize_clip(param[0].data / scale, qmax)
                    prm_norm[1].data[...] = quantize(param[1].data / scale_cur, qmax)  # 本当はbiasは(<< shift)で割る
            # f2i層のpython_paramにscaleとshiftを設定
            # param_str 中のkeyはダブルクオートでくくらないとjsonが読んでくれない
            prm_dict_str = '{ "scale":' + str(scale_cur) + ', "shift":' + str(int(shift)) + ' }'
            f2i.python_param.param_str = prm_dict_str
        elif (layer.type == 'Python') and (layer.python_param.layer == 'F2I'):
            f2i = layer     # conv層のパラメータを設定するためF2I層を保持
        elif (layer.type == 'Eltwise'):
            scale0 = scale_dict[layer.bottom[0]]
            scale1 = scale_dict[layer.bottom[1]]
            # scaleの大きい側に小さい側を合わせる
            scale_cur = np.max((scale0, scale1))
            print layer.name, "bottom0:", scale0, "bottom1:", scale1, "scale_cur:", scale_cur
            layer.eltwise_param.coeff[0] = scale0 / scale_cur
            layer.eltwise_param.coeff[1] = scale1 / scale_cur
        # この層出力のスケールを辞書に設定
        if len(layer.top) > 0:          # Silenceは len(layer.top) == 0
            for i in range(len(layer.top)):
                #print layer.name, layer.top[i], "scale_cur=", scale_cur
                scale_dict[layer.top[i]] = scale_cur
    return net_norm

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
    #parser.add_argument(
    #    "--scale",
    #    default='1',
    #    help="initial scale"
    #)
    parser.add_argument(
        "--qbit",
        default='16',
        help="quantization bit"
    )
    parser.add_argument(    # --no_quantize, --no_normalize時のネットワーク動作は未確認
        "--no_quantize",
        action='store_true',
        help="disable quantization"
    )
    parser.add_argument(
        "--no_normalize",
        action='store_true',
        help="disable normalization"
    )
    args = parser.parse_args()
    #
    model = args.model_file #sys.argv[1]
    param = args.param_file #sys.argv[2]
    #scale_prev = float(args.scale)  #int(sys.argv[3])
    qbit  = int(args.qbit)  #int(sys.argv[3])
    b_no_quantize = args.no_quantize
    b_no_normalize = args.no_normalize
    #print '\n----- 02-normalize.py -----', "model:", model, "param:", param, "scale", scale_prev, "qbit:", qbit, "b_no_quantize:", b_no_quantize, "b_no_normalize:", b_no_normalize
    print '\n----- 02-normalize.py -----', "model:", model, "param:", param, "qbit:", qbit, "b_no_quantize:", b_no_quantize, "b_no_normalize:", b_no_normalize
    #
    tmp = model.split('.')
    model_out = tmp[0] + 'n.' + tmp[1]
    tmp = param.split('.')
    param_out = tmp[0] + 'n.' + tmp[1]
    print "model:", model, "->", model_out
    print "param:", param, "->", param_out
    # global variables
    qmax = (1 << (qbit - 1)) - 1         # qbitで表現されるint型の最大の絶対値
    print "qbit:", qbit, "qmax:", qmax
    #
    ptxt     = csslib.load_pb2(model)                  # 変換前prototxt
    net      = csslib.load_network(model, param)       # 変換前ネットワーク
    net_norm = csslib.load_network(model, param)       # 変換後ネットワーク（変換前を初期値とする）
    
    net_norm = normalize(ptxt, net, net_norm)   # normalize
    
    # normalizeしたネットワークを保存
    print 'saving modified caffemodel to', param_out
    net_norm.save(param_out)                    # caffemodel
    print 'saving modified prototxt to', model_out
    csslib.save_pb2(ptxt, model_out)                   # prototxt
    ## scale辞書を表示
    #for key in scale_dict:
    #    print key, scale_dict[key]
    # scale辞書をscale_dict.txt に保存
    #json.dump(scale_dict, file('scale_dict.txt', 'w'))
    print 'saving scale dictionary to scale_dict.txt'
    csslib.save_dict(scale_dict, 'scale_dict.txt')
