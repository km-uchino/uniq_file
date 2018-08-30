#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
prototxtよりconv,eltwise層のscaleパラメータを抽出
"""
import sys
import csslib
import ast

def extract_conv_scale():
    i = 0
    print 'const int k_scale' + layer_name + '[] = {'
    for layer in ptxt.layer:
        if (layer.type == 'Python') and (layer.python_param.layer == 'I2F'):
            pdict = ast.literal_eval(layer.python_param.param_str)
            print '    %4d' % pdict['scale'], ', //', i, layer.name
            i = i + 1
        #if (layer.name[:7] == 'upscore'):    # HC用
        #    # upscoreでは、次段に接続するためscaleを1.0に戻す
        #    print '    %4d' % int(sdict[layer.name] * 256), ', //', layer.name, ': output scale=1.0'
    print '};'
    print ''

def extract_eltwise_scale():
    print 'const DType k_ew' + layer_name + '[] = {'
    for layer in ptxt.layer:
        if (layer.type == 'Eltwise'):
            print '    EWCOEF(', '%14.12f' % layer.eltwise_param.coeff[0], '), EWCOEF(', '%14.12f' % layer.eltwise_param.coeff[1], '), //', layer.name
    print '};'
    print ''

# int16 -> fp32へ戻すためのスケールを抽出    
def extract_i2f_scale():
    layers = ( 'eltwise_stage2_block1_rl', 'rpn_cls_score_cv_i2f', 'rpn_bbox_pred_cv_i2f' )
    i = 0
    print 'const float i2f_scale' + layer_name + '[] = {'
    for layer in layers:
        print '   %14.12f' % sdict[layer], ',   //', i, layer
        i = i + 1
    print '};'
    print ''
    
##### main #####
if __name__ == '__main__':
    print '\n// ----- 05-extract_prm.py -----', sys.argv[1], sys.argv[2]
    ptxt  = csslib.load_pb2(sys.argv[1])
    sdict = csslib.load_dict(sys.argv[2])
    out_layer = sys.argv[3]
    if len(sys.argv) > 4:
        layer_name = sys.argv[4]
    else:
        layer_name = ''
    #for layer in ptxt.layer:
    #    print layer.name
    print '// ----- for css_pose.cpp -----'
    extract_conv_scale()
    extract_eltwise_scale()
    extract_i2f_scale()
    #print '// ----- for Makefile -----'
    #print '// SCALE_INTxx =', sdict[out_layer]
