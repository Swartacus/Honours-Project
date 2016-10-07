from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def cnn_max_elu(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses ELUs
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.elu1 = L.ELU(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.elu1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.elu2 = L.ELU(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.elu2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.elu3 = L.ELU(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.elu3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.elu4 = L.ELU(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.elu4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.elu5 = L.ELU(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.elu5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.elu6 = L.ELU(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.elu6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.elu7 = L.ELU(n.drop7)

    # n.pool = L.Pooling(n.elu7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.elu7, n.label, norm='L2')

    return n.to_proto()

def cnn_max_relu(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses ReLUs
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.relu1 = L.ReLU(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.relu1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.relu2 = L.ReLU(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.relu2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.relu3 = L.ReLU(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.relu3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.relu4 = L.ReLU(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.relu4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.relu5 = L.ReLU(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.relu5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.relu6 = L.ReLU(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.relu6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.relu7 = L.ReLU(n.drop7)

    # n.pool = L.Pooling(n.relu7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.relu7, n.label, norm='L2')

    return n.to_proto()

def cnn_max_leakyrelu(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses LeakyReLUs
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.relu1 = L.ReLU(n.drop1, neagtive_slope=0.01)
    # second stack
    n.conv2 = L.Convolution(n.relu1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.relu2 = L.ReLU(n.drop2, neagtive_slope=0.01)
    # third stack
    n.conv4 = L.Convolution(n.relu2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.relu3 = L.ReLU(n.drop3, neagtive_slope=0.01)
    # fourth stack
    n.conv6 = L.Convolution(n.relu3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.relu4 = L.ReLU(n.drop4, neagtive_slope=0.01)
    # fifth stack
    n.conv8 = L.Convolution(n.relu4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.relu5 = L.ReLU(n.drop5, neagtive_slope=0.01)
    # sixth stack
    n.conv10 = L.Convolution(n.relu5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.relu6 = L.ReLU(n.drop6, neagtive_slope=0.01)
    # seventh stack
    n.conv11 = L.Convolution(n.relu6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.relu7 = L.ReLU(n.drop7, neagtive_slope=0.01)

    # n.pool = L.Pooling(n.relu7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.relu7, n.label, norm='L2')

    return n.to_proto()

def cnn_max_sig(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses Sigmoids
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.sig1 = L.Sigmoid(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.sig1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.sig2 = L.Sigmoid(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.sig2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.sig3 = L.Sigmoid(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.sig3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.sig4 = L.Sigmoid(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.sig4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.sig5 = L.Sigmoid(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.sig5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.sig6 = L.Sigmoid(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.sig6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.sig7 = L.Sigmoid(n.drop7)

    # n.pool = L.Pooling(n.sig7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.sig7, n.label, norm='L2')

    return n.to_proto()


def cnn_max_tanh(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses TanH activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.tanh1 = L.TanH(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.tanh1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.tanh2 = L.TanH(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.tanh2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.tanh3 = L.TanH(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.tanh3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.tanh4 = L.TanH(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.tanh4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.tanh5 = L.TanH(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.tanh5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.tanh6 = L.TanH(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.tanh6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.tanh7 = L.TanH(n.drop7)

    # n.pool = L.Pooling(n.tanh7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.tanh7, n.label, norm='L2')

    return n.to_proto()

def cnn_max_absval(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses Absolute Value activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.absval1 = L.AbsVal(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.absval1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.absval2 = L.AbsVal(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.absval2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.absval3 = L.AbsVal(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.absval3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.absval4 = L.AbsVal(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.absval4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.absval5 = L.AbsVal(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.absval5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.absval6 = L.AbsVal(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.absval6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.absval7 = L.AbsVal(n.drop7)

    # n.pool = L.Pooling(n.absval7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.absval7, n.label, norm='L2')

    return n.to_proto()


def cnn_max_power(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses Powers
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.power1 = L.Power(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.power1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.power2 = L.Power(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.power2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.power3 = L.Power(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.power3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.power4 = L.Power(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.power4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.power5 = L.Power(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.power5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.power6 = L.Power(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.power6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.power7 = L.Power(n.drop7)

    # n.pool = L.Pooling(n.power7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.power7, n.label, norm='L2')

    return n.to_proto()

def cnn_max_bnll(lmdb, batch_size, mean):
    '''
    Creates a CNN that uses BNLL activations
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.bnll1 = L.BNLL(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.bnll1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.bnll2 = L.BNLL(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.bnll2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.bnll3 = L.BNLL(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.bnll3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.bnll4 = L.BNLL(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.bnll4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv8, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.bnll5 = L.BNLL(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.bnll5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.bnll6 = L.BNLL(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.bnll6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.bnll7 = L.BNLL(n.drop7)

    # n.pool = L.Pooling(n.bnll7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.HingeLoss(n.bnll7, n.label, norm='L2')

    return n.to_proto()



with open('cnn_max_relu_train.prototxt','w') as f:
    f.write(str(cnn_max_relu('../../data/cifar-10/cifar10_train_lmdb', 64, 'mean.binaryproto')))

with open('cnn_max_relu_test.prototxt','w') as f:
    f.write(str(cnn_max_relu('../../data/cifar-10/cifar10_test_lmdb', 64, 'mean.binaryproto')))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('cifar-10_solver_SGD.prototxt')

solver.solve()
