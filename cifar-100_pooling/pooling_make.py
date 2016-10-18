from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys, os
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
import avepooling as create_ave
import maxpooling as create_max
import stochpooling as create_stoch
from caffe.proto import caffe_pb2


caffe.set_device(0)
caffe.set_mode_gpu()

'''
*******************************************************************************
Create nets
*******************************************************************************
'''
pools = ['average','max','stochastic']
for p in pools:
    # create relu net:
    with open('{}/relu/poolcnn_relu_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/relu/poolcnn_relu_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_relu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create leakyrelu net:
    with open('{}/leakyrelu/poolcnn_leakyrelu_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/leakyrelu/poolcnn_leakyrelu_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_leakyrelu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create sigmoid net:
    with open('{}/sigmoid/poolcnn_sigmoid_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/sigmoid/poolcnn_sigmoid_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_sig('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create tanh net:
    with open('{}/tanh/poolcnn_tanh_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/tanh/poolcnn_tanh_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_tanh('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create absval net:
    with open('{}/absval/poolcnn_absval_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/absval/poolcnn_absval_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_absval('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create power net:
    with open('{}/power/poolcnn_power_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/power/poolcnn_power_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_power('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create bnll net:
    with open('{}/bnll/poolcnn_bnll_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/bnll/poolcnn_bnll_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_bnll('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

    # create elu net:
    with open('{}/elu/poolcnn_elu_train.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'max':
            f.write(str(create_max.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 64)))
    with open('{}/elu/poolcnn_elu_test.prototxt'.format(p),'w') as f:
        if p == 'average':
            f.write(str(create_ave.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'max':
            f.write(str(create_max.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))
        if p == 'stochastic':
            f.write(str(create_stoch.cnn_elu('../../../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))



'''
*******************************************************************************
Create solvers
*******************************************************************************
'''
# set up solvers:
solverSGD = caffe_pb2.SolverParameter()
solverAdaDelta = caffe_pb2.SolverParameter()
solverAdaGrad = caffe_pb2.SolverParameter()
solverAdam = caffe_pb2.SolverParameter()
solverNAG = caffe_pb2.SolverParameter()
solverRMS = caffe_pb2.SolverParameter()

solverSGD.type = "SGD"
solverAdaDelta.type = "AdaDelta"
solverAdaGrad.type = "AdaGrad"
solverAdam.type = "Adam"
solverNAG.type = "Nesterov"
solverRMS.type = "RMSProp"

solvers = [solverSGD, solverAdaDelta, solverAdaGrad, solverAdam, solverNAG, solverRMS]

path = os.path.dirname(os.path.abspath(__file__))
nets = ['relu','leakyrelu','elu','sigmoid','tanh','absval','power','bnll']
for p in pools:
    for n in nets:
        for s in solvers:
            # s = None
            s.train_net = path + '/{0}/{1}/poolcnn_{1}_train.prototxt'.format(p,n)
            del s.test_net[:]
            s.test_net.append(path + '/{0}/{1}/poolcnn_{1}_test.prototxt'.format(p,n))
            s.test_interval = 150
            del s.test_iter[:]
            s.test_iter.append(100)
            s.max_iter = 15000
            s.base_lr = 0.0001
            if (s.type != 'AdaGrad') and (s.type != 'Adam') and (s.type != 'RMSProp'):
                s.momentum = 0.9
            s.lr_policy = 'step'
            s.gamma = 0.1
            s.stepsize = 5000
            s.weight_decay = 0.0005
            s.display = 50
            s.snapshot = 5000
            s.snapshot_prefix = 'cifar-10_{0}_{1}'.format(n,s.type)
            s.solver_mode = caffe_pb2.SolverParameter.GPU
            stype = s.type
            with open(path + '/{0}/{1}/poolcnn_{1}_solver_{2}.prototxt'.format(p,n,stype),'w') as f:
                f.write(str(s))
            with open(path + '/{0}/log/poolcnn_{1}_{2}.log'.format(p,n,stype),'w') as f:
                f.write('')



    # solver = None
    # solver = caffe.get_solver(path + '/poolcnn_solver_{}.prototxt'.format(stype))
    # niter = 250
    # test_interval = niter/10
    # train_loss = zeros(niter)
    # test_acc = zeros(int(np.ceil(niter / test_interval)))


    # for it in range(niter):
    #     solver.step(1)
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
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(arange(niter), train_loss)
    # ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    # ax1.set_xlabel('iteration')
    # ax1.set_ylabel('train loss')
    # ax2.set_ylabel('test accuracy')
    # ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
    # fig.savefig('{}.png'.format(stype))
    # plt.close(fig)

# solverAdam.train_net = path + '/poolcnn_relu_train.prototxt'
# solverAdam.test_net.append(path + '/poolcnn_relu_test.prototxt')
# solverAdam.test_interval = 500
# solverAdam.test_iter.append(100)
# solverAdam.max_iter = 10000
#
# solverAdam.base_lr = 0.01
# solverAdam.momentum = 0.9
# solverAdam.lr_policy = 'step'
# solverAdam.gamma = 0.1
# solverAdam.stepsize = 1000
# solverAdam.display = 1000
# solverAdam.snapshot = 5000
# solverAdam.snapshot_prefix = 'cifar-10_relu_adam'
# solverAdam.solver_mode = caffe_pb2.SolverParameter.GPU
#
# with open(path + '/poolcnn_solver_adam.prototxt','w') as f:
#     f.write(str(solverAdam))
#
# solver = None
# solver = caffe.get_solver(path + '/poolcnn_solver_adam.prototxt')
# niter = 250
# test_interval = niter/10
# train_loss = zeros(niter)
# test_acc = zeros(int(np.ceil(niter / test_interval)))
#
# for it in range(niter):
#     solver.step(1)
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
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(arange(niter), train_loss)
# ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# ax2.set_ylabel('test accuracy')
# ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
# fig.savefig('adam.png')
# plt.close(fig)
