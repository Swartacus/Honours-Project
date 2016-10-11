# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver allcnn_relu_solver_AdaGrad.prototxt 2>&1 | tee ../log/allcnn_relu_AdaGrad.log

  python ~/bin/caffe-master/tools/extra/parse_log.py ../log/allcnn_relu_AdaGrad.log .
