# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver allcnn_leakyrelu_solver_AdaDelta.prototxt 2>&1 | tee ../log/allcnn_leakyrelu_AdaDelta.log

python ~/bin/caffe-master/tools/extra/parse_log.py ../log/allcnn_leakyrelu_AdaDelta.log .
