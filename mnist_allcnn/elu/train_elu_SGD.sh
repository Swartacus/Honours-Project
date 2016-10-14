# train ReLU CNN using SGD
set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver allcnn_elu_solver_SGD.prototxt 2>&1 | tee ../log/allcnn_elu_SGD.log

#visualise
python ~/bin/caffe-master/tools/extra/parse_log.py ../log/allcnn_elu_SGD.log .
#gnuplot -persist gnuplot_commands
