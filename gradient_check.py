import numpy as np
import rNet as rNet
from random import uniform
import time


def gradient_check(net, input, predictions, cache0, cost, num_of_checks_per_layer=10):
    """
    Compares algorithmic gradients to their numeric counterparts
    :param net: instance of class rNet from module rNet
    :param input: tensor of rank 3 (a,b,c)
    :param predictions: tensor of rank 2 (a,b)
    :param cost: cost function from module rNet
    :param num_of_checks_per_layer: number of derivatives to check per layer
    """
    print('Begining algorithmic gradient calculation')
    t_start = time.time()
    _, dW, _ = net.train_step(input, predictions, cache0=cache0, cost=cost)
    t_end = time.time()
    print('Algorithmic gradient calculation completed in %f' % (t_end - t_start))
    delta = 1e-5
    time_steps, batch, input_size=input.shape
    for l in range(net.num_of_layers()):
        print('\nLayer %d' % (l))
        W = net.layers[l].weights
        for i in range(num_of_checks_per_layer):
            i_ = int(uniform(0, W.size))
            tmp = W.flat[i_]
            W.flat[i_] = tmp + delta
            net.reset()
            for ll in range(net.num_of_layers()-1):
                net.layers[ll].cell=cache0[ll]['c0']
                net.layers[ll].previous = cache0[ll]['h0']
            outputs = net(input)
            C1 = 0
            for t in range(0, time_steps):
                C1 += np.sum(cost(outputs[t], predictions[t]), axis=0)
            W.flat[i_] = tmp - delta
            net.reset()
            for ll in range(net.num_of_layers() - 1):
                net.layers[ll].cell = cache0[ll]['c0']
                net.layers[ll].previous = cache0[ll]['h0']
            outputs = net(input)
            C2 = 0
            for t in range(0, time_steps):
                C2 += np.sum(cost(outputs[t], predictions[t]), axis=0)
            dNum = (C1 - C2) / (2 * delta)
            relative_error = 0 if dNum ==0 and dW[l].flat[i_] == 0 else np.abs(dNum - dW[l].flat[i_]) / np.abs(dNum + dW[l].flat[i_])
            print('Index ', (np.unravel_index(i_, W.shape)))
            print('%f, %f => %e' % (dNum, dW[l].flat[i_], relative_error))
            W.flat[i_] = tmp

if __name__ == '__main__':
    '''
        select the FLAG you want to test
        by default, it will select one randomly
    '''
    FLAGS=('FCr','LSTM')
    FLAG=FLAGS[int(uniform(0, 2))]
    time_steps, batch, input_size, hidden_size, output_size = (100, 10, 84, 512, 84)
    if FLAG=='FCr':
        L_1=rNet.FCr([input_size, hidden_size], activation=rNet.tanh())
        L_2 = rNet.FCr([hidden_size, hidden_size], activation=rNet.tanh())
        cache0 = [{'c0': None, 'h0': np.random.rand(batch, 512)},
              {'c0': None, 'h0': np.random.rand(batch, 512)}, None]
    elif FLAG=='LSTM':
        L_1 = rNet.LSTM([input_size, hidden_size])
        L_2 = rNet.LSTM([hidden_size, hidden_size])
        cache0 = [{'c0': np.random.rand(batch, 512), 'h0': np.random.rand(batch, 512)},
              {'c0': np.random.rand(batch, 512), 'h0': np.random.rand(batch, 512)}, None]
    else:
        assert '%s is not a valid FLAG'%(FLAG)
    L_4 = rNet.FC([hidden_size, output_size], activation=rNet.softmax())
    net = rNet.rNet()
    net.add(L_1)
    net.add(L_2)
    net.add(L_4)
    net.init()
    input_tensor = np.random.randn(time_steps, batch, input_size)
    predictions = np.random.randint(0, output_size, (time_steps, batch))
    cost = rNet.softmax_loss()
    print('SELECTED FLAG = %s' % (FLAG))
    gradient_check(net, input=input_tensor, predictions=predictions, cache0=cache0, cost=cost, num_of_checks_per_layer=10)
