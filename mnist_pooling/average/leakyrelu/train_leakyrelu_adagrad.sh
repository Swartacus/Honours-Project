# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver poolcnn_leakyrelu_solver_AdaGrad.prototxt 2>&1 | tee ../log/poolcnn_leakyrelu_AdaGrad.log

  python ~/bin/caffe-master/tools/extra/parse_log.py ../log/poolcnn_leakyrelu_AdaGrad.log .
