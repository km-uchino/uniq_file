# 準備：in-place処理分離、ネットワークを前半CNN部、RPN部、後半CNN部、最後に分割
faster01.prototxt  : ../KMDNN/model/RpnPrototxt.hと同じ
↓手作業
faster02.prototxt  : in-place処理を分離、層名変更(Conv:_cv, BN:_bn, Scale:_sc, ReLU:_rl)
↓手作業
faster03a.prototxt : faster02.prototxt の 最初〜rpn_cls_score_cv
faster03b.prototxt : faster02.prototxt の rpn_cls_prob ~ roi_pool5
faster03c.prototxt : faster02.prototxt の conv_stage3_block0_proj_shortcut ~ joints_pred。入力層としてroipool5を追加
faster03d.prototxt : faster02.prototxtから cls_prob, rpn_reshape層のみ抽出

# 層名変更
↓01.sh
faster03a2.prototxt : faster03a.prototxt の層名に'/'の入っているものを'_'に変更
faster03a2.caffemodel : faster03a2.prototxt に合わせて caffemodel中の層名を変更

# CBS統合
↓02.sh
faster04a.prototxt, faster04a.caffemodel : CBS統合版(前半)
faster04c.prototxt, faster04c.caffemodel : CBS統合版(後半)

# 後半をFCN化
faster04c.prototxt
↓手作業
faster05c.prototxt  : fc層 -> 1x1 convolutionに変更
↓03.sh
faster05c.caffemodel: faster04c.caffemodel のfc層を 1x1 convolutionに変更

# 全体を結合
↓04.sh
faster05a.prototxt  : faster04a.prototxt, faster03b.prototxt, faster05c.prototxt, faster03d.prototxt を結合
faster05.caffemodel : faster04a.caffemodel, faster05c.caffemodel を結合
↓手作業
faster05.prototxt : faster05a.prototxt より roipool5のdata層を削除


