set -e

TOOLS=~/bin/caffe-master/build/tools

$TOOLS/caffe train \
  -gpu 0 \
  -solver solver.prototxt 2>&1 | tee log/imagenet_log.log

python ~/bin/caffe-master/tools/extra/parse_log.py log/imagenet_log.log .
