#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
convert fp32 network parameter to int16 and dump
"""
import sys
import numpy as np
import argparse
import csslib

# conv層パラメータをint16でダンプ
# * weightは±1に正規化されている。qmaxをかけて整数化するだけ
# * biasのシフトには未対応
def prm_fp32toi16(net):
    i = 1
    for layer_name, param in net.params.iteritems():
	elements = len(param)
	for index in range(0, elements):
            prm_f32 = param[index].data
            prm_i16 = np.int16(np.ma.round(prm_f32 * qmax))
            prm_i16_clip = prm_i16.clip(qmax * -1, qmax)
            #diff = prm_i16 - prm_i16_clip
            #print "diff:", diff.flatten().max(), diff.flatten().min()
            #csslib.array_info('prm_f32', prm_f32)
            #csslib.array_info('prm_i16', prm_i16)
            #dif_f32 = prm_f32 - prm_i16 / float(qmax)
            #print "diff_max", dif_f32.flatten().max(), "delta:", 1.0/float(qmax)
            csslib.save_ndarray_bin_shape(prm_i16_clip, ("P%02d_" % i) + layer_name, '.bin')
	i += 1

##### main #####
if __name__ == '__main__':
    print '\n----- 04-prmFP32toI16.py -----'
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
        "--qbit",
        default='16',
        help="quantization bit"
    )
    args = parser.parse_args()
    #
    model = args.model_file
    param = args.param_file
    qbit  = int(args.qbit)
    if qbit > 16:
        print "ERROR qbit(%d) > 16" % qbit
        sys.exit()
    print "model:", model, "param:", param, "qbit:", qbit
    # global variables
    qmax = (1 << (qbit - 1)) - 1         # qbitで表現されるint型の最大の絶対値
    print "qbit:", qbit, "qmax:", qmax
    #
    net      = csslib.load_network(model, param)       # 変換前ネットワーク
    prm_fp32toi16(net)

    # test
    #a = np.random.rand(10)
    #csslib.array_info("a", a)
    #b = quantize(a, qmax)
    #csslib.array_info("b", b)
    #print a
    #print b
    #print b * qmax
    

    
