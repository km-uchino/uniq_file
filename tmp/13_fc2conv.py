#!/usr/bin/env python
# -*- coding: utf-8 -*-
# caffemodel中の FC層パラメータを 1x1 convolutionに変換

# suppress caffe.Net() messages
# ref. https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
import os
import sys
os.environ['GLOG_minloglevel'] = '2' 
import caffe

def transplant(new_net, net, suffix=''):
    # from fcn.berkeleyvision.org
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

if __name__ == '__main__':
    fc_ptxt   = sys.argv[1]
    fc_cmodel = sys.argv[2]
    fcn_ptxt   = sys.argv[3]
    fcn_cmodel = sys.argv[4]
    print '\n----- fc2conv.py -----', fc_cmodel, '->', fcn_cmodel
    caffe.set_mode_cpu()
    net_fc  = caffe.Net(fc_ptxt, fc_cmodel, caffe.TEST)
    net_FCN = caffe.Net(fcn_ptxt, caffe.TEST)
    transplant(net_FCN, net_fc)
    net_FCN.save(fcn_cmodel)
