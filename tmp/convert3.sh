#!/bin/bash
# ==================== ========================== ============================ =======================
#                      prototxt                   caffemodel                   scale_dict             
# ==================== ========================== ============================ =======================
#  初期ネットワーク    faster_cbs03.prototxt      faster_cbs03.caffemodel                            
#  Power層追加         faster_cbs03.prototxt      ↑
#  Bias調整            faster_cbs03.prototxt      faster_cbs03b.caffemodel
#  Python Layer追加    faster_cbs03p.prototxt     ↑                                                  
#  normalization       faster_cbs03pn.prototxt    faster_cbs03bn.caffemodel    sdict_css_res36hn.txt  
#  activation scaling  faster_cbs03pna.prototxt   faster_cbs03bna.caffemodel   sdict_css_res36hna.txt 
# ==================== ========================== ============================ =======================

NBIT=16                 # parameter量子化ビット数
# activation量子化ビット数は i2f.py の nbit を変更する
# 各層fp32データの分布範囲： data:+/-8, dataA:+/-32, dataB:+/-28
IN_NORM_SCALE=16        # 入力データ(FP32)を{-1,1}に正規化した際のスケーリング係数(Power層scale の逆数)
BASE="faster_cbs03"     # FCN版のベース名
#BASE_FC="faster_cbs03"  # fc版のベース名
#BASE="faster_cbs03fcn"  # FCN版のベース名

# ../exp_cbs/P000__input_320x240x3x1x.bin と ../exp_cbs/P001_slice_320x240x1x1x.binでは値域が違うが
# どちらでも IN_NORM_SCALE=1/256.0 でよい
#../exp_cbs/P000__input_320x240x3x1x.bin : float32 (230400,) -122.772 ~ 152.02                                     
#../exp_cbs/P001_slice_320x240x1x1x.bin  : float32 (76800,)  -102.98  ~ 152.02                      
#IMAGE="../exp_cbs/P000__input_320x240x3x1x.bin"   # input image (float32)
#IMAGE="css_test.jpg"   # input image
IMAGE="../exp_cbs/P075_roi_pool5_6x6x128x5x.bin"   # input image (float32)
IMAGEWIDTH=6
IMAGEHEIGHT=6
NCHANNELS=128
NBATCH=5
IMAGEDIR='../data'   # activation分布測定用画像ディレクトリ
NIMAGES=0            # 0:read directory
#OUT_LAYER="cls_prob"
OUT_LAYER="bbox_pred_i2f"
#
PROTOTXT="${BASE}.prototxt"
PROTOTXT_P="${BASE}p.prototxt"
PROTOTXT_PN="${BASE}pn.prototxt"
PROTOTXT_PNA="${BASE}pna.prototxt"
CAFFEMODEL="${BASE}.caffemodel"
CAFFEMODEL_B="${BASE}b.caffemodel"    # bias scaling後
CAFFEMODEL_BN="${BASE}bn.caffemodel"
CAFFEMODEL_BNA="${BASE}bna.caffemodel"
SCALE_DICT="sdict_${BASE}.txt"        # OUT_LAYERのスケールを指定(1.0)
SCALE_DICT_N="sdict_${BASE}n.txt"
SCALE_DICT_NA="sdict_${BASE}na.txt"
STAT_PARAM="stat_param_${BASE}.txt"
STAT_PARAM_N="stat_param_${BASE}n.txt"
STAT_PARAM_NA="stat_param_${BASE}na.txt"
STAT_OUT="stat_out_${BASE}.txt"
STAT_OUT_N="stat_out_${BASE}n.txt"
STAT_OUT_NA="stat_out_${BASE}na.txt"

# 入力データ値域確認
./show_fp32.py ${IMAGE}
# !!! prototxtにPower層を追加し、入力データが{-1,1}の範囲になるようscale=1/IN_NORM_SCALE にすること !!!

## fc用caffemodelを FCN用caffemodelに変換
#./fc2conv.py

# prototxtのPower層でのスケール分、各conv層のbiasを補正
./00-scale_bias.py --scale ${IN_NORM_SCALE} ${PROTOTXT} ${CAFFEMODEL}
# add python layer
./01-add_pythonlayer.py ${PROTOTXT} ${PROTOTXT_P} 
# normalize and quantize
#./02-normalize.py --qbit ${NBIT} --scale ${IN_NORM_SCALE} ${PROTOTXT_P} ${CAFFEMODEL_B} 
./02-normalize.py --qbit ${NBIT} ${PROTOTXT_P} ${CAFFEMODEL_B} 
mv scale_dict.txt ${SCALE_DICT_N}
## analyse activation and modify network
./33-analyse_act.py ${PROTOTXT_PN} ${CAFFEMODEL_BN} ${IMAGEDIR} ${IMAGEWIDTH} ${IMAGEHEIGHT} ${OUT_LAYER} ${SCALE_DICT_N} ${NIMAGES} 1 ${NCHANNELS} ${NBATCH}
## convert fp32 network parameter to int16 and dump
./34-prmFP32toI16.py --qbit ${NBIT} ${PROTOTXT_PNA} ${CAFFEMODEL_BNA}

# evaluate 1 image with original/parameter_normalized/activation_quantized network
./30-forward.py ${PROTOTXT_P}   ${CAFFEMODEL_B}   ${IMAGE} ${IMAGEWIDTH} ${IMAGEHEIGHT} ${OUT_LAYER} ${SCALE_DICT}    ${NCHANNELS} ${NBATCH}
mv out.bin        out_org.bin
mv stat_param.txt ${STAT_PARAM}
mv stat_out.txt   ${STAT_OUT}
./30-forward.py ${PROTOTXT_PN}  ${CAFFEMODEL_BN}  ${IMAGE} ${IMAGEWIDTH} ${IMAGEHEIGHT} ${OUT_LAYER} ${SCALE_DICT_N}  ${NCHANNELS} ${NBATCH}
mv out.bin        out_n.bin
mv stat_param.txt ${STAT_PARAM_N}
mv stat_out.txt   ${STAT_OUT_N}
./30-forward.py ${PROTOTXT_PNA} ${CAFFEMODEL_BNA} ${IMAGE} ${IMAGEWIDTH} ${IMAGEHEIGHT} ${OUT_LAYER} ${SCALE_DICT_NA} ${NCHANNELS} ${NBATCH}
mv out.bin        out_na.bin
mv stat_param.txt ${STAT_PARAM_NA}
mv stat_out.txt   ${STAT_OUT_NA}

##
./35-extract_param.py ${PROTOTXT_PNA} ${SCALE_DICT_NA} ${OUT_LAYER} 3 2>&1 | tee scale3.c
