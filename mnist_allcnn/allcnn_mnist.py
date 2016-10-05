from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def allcnn(lmdb, batch_size):
    '''
    Creates an all-CNN
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.conv3 = L.Convolution(n.relu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.conv6 = L.Convolution(n.conv5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.relu6 = L.ReLU(n.conv6, in_place=True)
    n.conv7 = L.Convolution(n.relu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.relu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.relu8 = L.ReLU(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.relu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.relu9 = L.ReLU(n.conv9, in_place=True)

    n.pool = L.Pooling(n.relu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('all_cnn_train.prototxt','w') as f:
    f.write(str(allcnn('mnist_train_lmdb', 100)))

with open('all_cnn_test.prototxt','w') as f:
    f.write(str(allcnn('mnist_test_lmdb', 100)))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('mnist_allcnn_solver.prototxt')

# solver.net.forward()
# solver.test_nets[0].forward()
# solver.net.backward()

solver.solve()
