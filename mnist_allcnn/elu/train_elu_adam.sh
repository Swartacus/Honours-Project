# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver allcnn_elu_solver_Adam.prototxt 2>&1 | tee ../log/allcnn_elu_Adam.log

python ~/bin/caffe-master/tools/extra/parse_log.py ../log/allcnn_elu_Adam.log .
