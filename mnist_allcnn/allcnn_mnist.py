
from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def allcnn_relu(lmdb, batch_size):
    '''
    Creates an all-CNN that uses ReLU activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)

    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.relu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.relu3 = L.ReLU(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.relu5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.relu6 = L.ReLU(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.relu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.relu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.relu8 = L.ReLU(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.relu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.relu9 = L.ReLU(n.conv9, in_place=True)

    n.pool = L.Pooling(n.relu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


def allcnn_elu(lmdb, batch_size):
    '''
    Creates an all-CNN that uses ELU activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)

    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.elu1 = L.ELU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.elu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.elu2 = L.ELU(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.elu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.elu3 = L.ELU(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.elu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.elu4 = L.ELU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.elu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.elu5 = L.ELU(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.elu5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.elu6 = L.ELU(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.elu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.elu7 = L.ELU(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.elu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.elu8 = L.ELU(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.elu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.elu9 = L.ELU(n.conv9, in_place=True)

    n.pool = L.Pooling(n.elu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_leakyrelu(lmdb, batch_size):
    '''
    Creates an all-CNN that uses LeakyReLU activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, negative_slope = 0.01, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.bn1, negative_slope = 0.01, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.relu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.relu3 = L.ReLU(n.drop1, negative_slope = 0.01, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, negative_slope = 0.01, in_place=True)
    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.relu5 = L.ReLU(n.conv5, negative_slope = 0.01, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.relu5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.relu6 = L.ReLU(n.drop4, negative_slope = 0.01, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.relu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.conv7, negative_slope = 0.01, in_place=True)
    n.conv8 = L.Convolution(n.relu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.relu8 = L.ReLU(n.conv8, negative_slope = 0.01, in_place=True)
    n.conv9 = L.Convolution(n.relu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.relu9 = L.ReLU(n.conv9, negative_slope = 0.01, in_place=True)

    n.pool = L.Pooling(n.relu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_sigmoid(lmdb, batch_size):
    '''
    Creates an all-CNN that uses Sigmoid activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.sig1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.sig2 = L.Sigmoid(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.sig2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.sig3 = L.Sigmoid(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.sig3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.sig4 = L.Sigmoid(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.sig4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.sig5 = L.Sigmoid(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.sig5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.sig6 = L.Sigmoid(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.sig6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.sig7 = L.Sigmoid(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.sig7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.sig8 = L.Sigmoid(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.sig8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.sig9 = L.Sigmoid(n.conv9, in_place=True)

    n.pool = L.Pooling(n.sig9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_tanh(lmdb, batch_size):
    '''
    Creates an all-CNN that uses Hyperbolic Tangent activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.tanh1 = L.TanH(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.tanh1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.tanh2 = L.TanH(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.tanh2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.tanh3 = L.TanH(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.tanh3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.tanh4 = L.TanH(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.tanh4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.tanh5 = L.TanH(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.tanh5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.tanh6 = L.TanH(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.tanh6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.tanh7 = L.TanH(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.tanh7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.tanh8 = L.TanH(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.tanh8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.tanh9 = L.TanH(n.conv9, in_place=True)

    n.pool = L.Pooling(n.tanh9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_absval(lmdb, batch_size):
    '''
    Creates an all-CNN that uses absolute value activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.absval1 = L.AbsVal(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.absval1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.absval2 = L.AbsVal(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.absval2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.absval3 = L.AbsVal(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.absval3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.absval4 = L.AbsVal(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.absval4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.absval5 = L.AbsVal(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.absval5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.absval6 = L.AbsVal(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.absval6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.absval7 = L.AbsVal(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.absval7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.absval8 = L.AbsVal(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.absval8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.absval9 = L.AbsVal(n.conv9, in_place=True)

    n.pool = L.Pooling(n.absval9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_power(lmdb, batch_size):
    '''
    Creates an all-CNN that uses Power activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.power1 = L.Power(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.power1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.power2 = L.Power(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.power2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.power3 = L.Power(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.power3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.power4 = L.Power(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.power4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.power5 = L.Power(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.power5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.power6 = L.Power(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.power6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.power7 = L.Power(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.power7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.power8 = L.Power(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.power8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.power9 = L.Power(n.conv9, in_place=True)

    n.pool = L.Pooling(n.power9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

def allcnn_bnll(lmdb, batch_size):
    '''
    Creates an all-CNN that uses BNLL activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bnll1 = L.BNLL(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.bnll1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv2)
    n.bnll2 = L.BNLL(n.bn1, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv3 = L.Convolution(n.bnll2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.drop1 = L.Dropout(n.conv3, dropout_ratio=0.5)
    n.bnll3 = L.BNLL(n.drop1, in_place=True)

    # second stack
    n.conv4 = L.Convolution(n.bnll3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bnll4 = L.BNLL(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.bnll4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv5)
    n.bnll5 = L.BNLL(n.conv5, in_place=True)
    # Convolution with stride 2 replaces pooling
    n.conv6 = L.Convolution(n.bnll5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.drop4 = L.Dropout(n.conv6, dropout_ratio=0.5)
    n.bnll6 = L.BNLL(n.drop4, in_place=True)

    # final stack and output
    n.conv7 = L.Convolution(n.bnll6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.bnll7 = L.BNLL(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.bnll7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.bnll8 = L.BNLL(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.bnll8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.bnll9 = L.BNLL(n.conv9, in_place=True)

    n.pool = L.Pooling(n.bnll9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.accuracy = L.Accuracy(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()



# with open('all_cnn_train.prototxt','w') as f:
#     f.write(str(allcnn('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
#
# with open('all_cnn_test.prototxt','w') as f:
#     f.write(str(allcnn('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# caffe.set_device(0)
# caffe.set_mode_gpu()

### load the solver and create train and test nets
# solverSGD = None
# solverSGD = caffe.SGDSolver('cifar-10_solver.prototxt')

# solver.net.forward()
# solver.test_nets[0].forward()
# solver.net.backward()

# solverSGD.solve()

# niter = 250
# test_interval = niter / 10
# # losses will also be stored in the log
# train_loss = zeros(niter)
# test_acc = zeros(int(np.ceil(niter / test_interval)))
#
# # the main solver loop
# for it in range(niter):
#     solver.step(1)  # SGD by Caffe
#
#     # store the train loss
#     train_loss[it] = solver.net.blobs['loss'].data
#
#     # run a full test every so often
#     # (Caffe can also do this for us and write to a log, but we show here
#     #  how to do it directly in Python, where more complicated things are easier.)
#     if it % test_interval == 0:
#         print 'Iteration', it, 'testing...'
#         correct = 0
#         for test_it in range(100):
#             solver.test_nets[0].forward()
#             correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
#                            == solver.test_nets[0].blobs['label'].data)
#         test_acc[it // test_interval] = correct / 1e4
#
# _, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(arange(niter), train_loss)
# ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# ax2.set_ylabel('test accuracy')
# ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
# plt.show()
