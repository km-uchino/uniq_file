# -*- coding: utf-8 -*-
# library for CSS

import numpy as np
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import json

np.set_printoptions(linewidth=800)

# binaryファイルのデータをtype型としてndarrayに読み込む
def load_ndarray_bin(filename, type):
    return np.fromfile(filename, type)

# arrayをbinaryファイルにダンプ
def save_ndarray_bin(array, filename):
    array.tofile(filename)

# arrayをbinaryファイルにダンプ
def save_ndarray_bin_shape(array, filename, suffix):
    fullfilename = filename + '_'
    nelem = len(array.shape)
    for i in range(nelem - 1, -1, -1):
        fullfilename = fullfilename + str(array.shape[i]) + 'x'
    fullfilename = fullfilename + suffix
    print "saving", fullfilename, array.shape, array.dtype
    array.tofile(fullfilename)
        
# json形式の辞書をロード
def load_dict(filename):
    return json.load(file(filename))

# 辞書をjson形式でダンプ
def save_dict(dict, filename):
    json.dump(dict, file(filename, 'w'))

# protocol buffer読み込み
def load_pb2(model):
    net = caffe_pb2.NetParameter()
    with open(model, 'r') as f:
        pb.text_format.Merge(f.read(), net)    
    return net

# save network prototxt
def save_pb2(net, outfile):
    with open(outfile, 'w') as f:
        f.write(pb.text_format.MessageToString(net))

def load_network(model, param):
    print 'load_network'
    return caffe.Net(model,      # defines the structure of the model
                     param,      # contains the trained weights
                     caffe.TEST) # use test mode (e.g., don't perform dropout)

# netのlayerのパラメータ統計値を表示
def layer_statistics(net, layer, fp):
    # the parameters are a list of [weights, biases]
    weights = net.params[layer][0].data
    biases  = net.params[layer][1].data
    w_flat = weights.flatten()
    print >> fp, ":layer:", layer, ":w.shape:", weights.shape, ":w.min:", w_flat.min(), ":w.max:", w_flat.max(), ":w.mean:", w_flat.mean(),
    print >> fp, ":b.shape:", biases.shape, ":b.min:", biases.min(), ":b.max:", biases.max(), ":b.mean:", biases.mean()

# network中のパラメータのある層のパラメータ統計値を表示
def net_param_statitics(net):
    with open('stat_param.txt', 'wt') as fout:
        for layer_name, param in net.params.iteritems():
            layer_statistics(net, layer_name, fout)    

# networkパラメータを PXX_xxxx.npy にダンプ
def dump_param(net):
    i = 0
    for layer_name, param in net.params.iteritems():
	elements = len(param)
	for index in range(0, elements):
	    np.save(("P%02d_" % i) + layer_name + '_' + str(index) + '.npy', param[index].data)
	i += 1

# create transformer for the input called 'data'
#  reshapeサイズはnet.blobs['data'].dataのサイズを使用するので、 net.blobs['data'].dataは予め所定サイズに変更しておくこと
def gen_transformer(net):        
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
    #transformer.set_mean('data', mu)              # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR : also necessary for grayscale?
    return transformer

def transform_image(net, imagefile, transformer):
    image = caffe.io.load_image(imagefile)
    #print 'image.shape:', image.shape
    #print 'image.average:', image.mean(0).mean(0)
    transformed_image = transformer.preprocess('data', image)
    print 'transformed_image.shape:', transformed_image.shape
    print 'transformed_image.average:', transformed_image.mean(1).mean(1)
    #plt.imshow(image)
    return transformed_image

def forward(net, image, transformer):
    net.blobs['data'].data[...] = transform_image(net, image, transformer)
    #print type(net.blobs['data'].data),  "shape:", net.blobs['data'].data.shape
    output = net.forward()                                    # run network
    return output

# activationの統計値出力
def act_statistics(net, layer, fp):
  data = net.blobs[layer].data
  flat = data.flatten()
  f_gt0 = flat[flat > 0]
  print >> fp, ":layer:", layer, ":shape:", data.shape, ":size:", flat.size, ":min:", flat.min(), ":max:", flat.max(),
  print >> fp, ":nzmean:", f_gt0.mean(), ":nz_std:", f_gt0.std(),  ":zero_ratio:", float(flat.size - f_gt0.size) * 100 / flat.size

# 全activationの統計値をstat_out.txtに出力
def net_act_statitics(net):
    with open('stat_out.txt', 'wt') as fout:
        for layer_name, blob in net.blobs.iteritems():
            act_statistics(net, layer_name, fout)

def array_info(name, array):
    print name, array.shape, array.dtype
