# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver poolcnn_elu_solver_AdaDelta.prototxt
  -snapshot cifar-10_elu_AdaDelta_iter_15000.solverstate 2>&1 | tee ../log/poolcnn_elu_AdaDelta.log

python ~/bin/caffe-master/tools/extra/parse_log.py ../log/poolcnn_elu_AdaDelta.log .
