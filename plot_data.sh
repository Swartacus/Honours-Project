set -e

# draw all-cnn cifar-10 relu
python ~/bin/caffe-master/python/draw_net.py cifar-10_allcnn/relu/allcnn_relu_train.prototxt cifar-10_allcnn_relu.png
gnuplot cifar-10_allcnn/relu/gnuplot_commands

# draw all-cnn cifar-10 elu
python ~/bin/caffe-master/python/draw_net.py cifar-10_allcnn/elu/allcnn_elu_train.prototxt cifar-10_allcnn_elu.png
gnuplot cifar-10_allcnn/elu/gnuplot_commands

# draw all-cnn cifar-10 leakyrelu
python ~/bin/caffe-master/python/draw_net.py cifar-10_allcnn/leakyrelu/allcnn_leakyrelu_train.prototxt cifar-10_allcnn_leakyrelu.png
gnuplot cifar-10_allcnn/leakyrelu/gnuplot_commands

# draw all-cnn cifar-100 relu
python ~/bin/caffe-master/python/draw_net.py cifar-100_allcnn/relu/allcnn_relu_train.prototxt cifar-100_allcnn_relu.png
gnuplot cifar-100_allcnn/relu/gnuplot_commands

# draw all-cnn cifar-100 elu
python ~/bin/caffe-master/python/draw_net.py cifar-100_allcnn/elu/allcnn_elu_train.prototxt cifar-100_allcnn_elu.png
gnuplot cifar-100_allcnn/elu/gnuplot_commands

# draw all-cnn cifar-100 leakyrelu
python ~/bin/caffe-master/python/draw_net.py cifar-100_allcnn/leakyrelu/allcnn_leakyrelu_train.prototxt cifar-100_allcnn_leakyrelu.png
gnuplot cifar-100_allcnn/leakyrelu/gnuplot_commands

# draw all-cnn mnist relu
python ~/bin/caffe-master/python/draw_net.py mnist_allcnn/relu/allcnn_relu_train.prototxt mnist_allcnn_relu.png
gnuplot mnist_allcnn/relu/gnuplot_commands

# draw all-cnn mnist elu
python ~/bin/caffe-master/python/draw_net.py mnist_allcnn/elu/allcnn_elu_train.prototxt mnist_allcnn_elu.png
gnuplot mnist_allcnn/elu/gnuplot_commands

# draw all-cnn mnist leakyrelu
python ~/bin/caffe-master/python/draw_net.py mnist_allcnn/leakyrelu/allcnn_leakyrelu_train.prototxt mnist_allcnn_leakyrelu.png
gnuplot mnist_allcnn/leakyrelu/gnuplot_commands

# draw pooling cifar-10 relu
python ~/bin/caffe-master/python/draw_net.py cifar-10_pooling/max/relu/poolcnn_relu_train.prototxt cifar-10_poolcnn_relu.png
gnuplot cifar-10_pooling/max/relu/gnuplot_commands

# draw pooling cifar-10 elu
python ~/bin/caffe-master/python/draw_net.py cifar-10_pooling/max/elu/poolcnn_elu_train.prototxt cifar-10_poolcnn_elu.png
gnuplot cifar-10_pooling/max/elu/gnuplot_commands

# draw pooling cifar-10 leakyrelu
python ~/bin/caffe-master/python/draw_net.py cifar-10_pooling/max/leakyrelu/poolcnn_leakyrelu_train.prototxt cifar-10_poolcnn_leakyrelu.png
gnuplot cifar-10_pooling/max/leakyrelu/gnuplot_commands

# draw all-cnn cifar-100 relu
python ~/bin/caffe-master/python/draw_net.py cifar-100_pooling/max/relu/poolcnn_relu_train.prototxt cifar-100_poolcnn_relu.png
gnuplot cifar-100_pooling/max/relu/gnuplot_commands

# draw all-cnn cifar-100 elu
python ~/bin/caffe-master/python/draw_net.py cifar-100_pooling/max/elu/poolcnn_elu_train.prototxt cifar-100_poolcnn_elu.png
gnuplot cifar-100_pooling/max/elu/gnuplot_commands

# draw all-cnn cifar-100 leakyrelu
python ~/bin/caffe-master/python/draw_net.py cifar-100_pooling/max/leakyrelu/poolcnn_leakyrelu_train.prototxt cifar-100_poolcnn_leakyrelu.png
gnuplot cifar-100_pooling/max/leakyrelu/gnuplot_commands

# draw all-cnn mnist relu
python ~/bin/caffe-master/python/draw_net.py mnist_pooling/max/relu/poolcnn_relu_train.prototxt mnist_poolcnn_relu.png
gnuplot mnist_pooling/max/relu/gnuplot_commands

# draw all-cnn mnist elu
python ~/bin/caffe-master/python/draw_net.py mnist_pooling/max/elu/poolcnn_elu_train.prototxt mnist_poolcnn_elu.png
gnuplot mnist_pooling/max/elu/gnuplot_commands

# draw all-cnn mnist leakyrelu
python ~/bin/caffe-master/python/draw_net.py mnist_pooling/max/leakyrelu/poolcnn_leakyrelu_train.prototxt mnist_poolcnn_leakyrelu.png
gnuplot mnist_pooling/max/leakyrelu/gnuplot_commands
