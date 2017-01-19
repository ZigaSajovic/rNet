import numpy as np


class tanh:
    def __call__(self, input):
        return np.tanh(input)

    def d(self, input):
        return 1 - input * input


class sigmoid:
    def __call__(self, input):
        return 1 / (1 + np.exp(-input))

    def d(self, input):
        return input * (1 - input)


class identity:
    def __call__(self, input):
        return input

    def d(self, input):
        return 1


class mean_square:
    def __call__(self, input_1, input_2):
        return np.mean(np.square(input_1 - input_2) / 2, axis=1,keepdims=True)

    def d(self, input_1, input_2):
        return (input_1 - input_2) / np.size(input_1)

class softmax:
    '''
    Only to be used as the last
    non-recursive layer
    '''

    def __call__(self, input):
        shifted_ = input - np.max(input)
        exp_ = np.exp(shifted_)
        return exp_ / np.sum(exp_, axis=1, keepdims=True)

    def d(self, probs, train_on=None):
        '''
        it returns 1,
        as the derivative is covered by a
        previous call to softmax_loss
        function
        '''
        return 1


class softmax_loss:
    '''
    To be used in combination with
    sofmax
    '''

    def __call__(self, input, train_on):
        return -np.log(input[range(train_on.size), train_on])

    def d(self, probs, train_on=None):
        '''
        it computes the softmax loss derivative
        for optimality
        '''
        tmp_ = np.copy(probs)
        tmp_[range(train_on.size), train_on] -= 1
        return tmp_


class rNet:
    layers = []
    num_of_layers = None

    def __init__(self):
        self.num_of_layers = lambda: len(self.layers)

    def __call__(self, input):
        '''
        input is to be a rank 3 tensor
        '''
        out = input
        for l in self.layers:
            out = l(out)
        return out

    def init(self):
        for l in self.layers:
            l.init()

    def reset(self):
        for l in self.layers:
            l.reset()

    def add(self, layer_):
        self.layers.append(layer_)

    def train_step(self, inputs, predictions, cache0=None, cost=softmax_loss()):
        cache = [None] * self.num_of_layers()
        cache_out = [None] * self.num_of_layers()
        out = inputs
        for l in range(0, self.num_of_layers()):
            if cache0 is not None and cache0[l] is not None:
                if cache0[l]['c0'] is not None and cache0[l]['h0'] is not None:
                    out, cache[l] = self.layers[l].eval_for_back_prop(out, cache0[l]['c0'],cache0[l]['h0'])
                elif cache0[l]['h0'] is not None:
                    out, cache[l] = self.layers[l].eval_for_back_prop(out, cache0[l]['h0'])
                else:
                    out, cache[l] = self.layers[l].eval_for_back_prop(out)
            else: out, cache[l] = self.layers[l].eval_for_back_prop(out)
            cache_out[l] = {
                'c0': cache[l]['cn'] if 'cn' in cache[l] else None,
                'h0': cache[l]['hn'] if 'hn' in cache[l] else None}
            time_steps, batch, out_ = out.shape
        dIn = np.zeros([time_steps, batch, out_])
        loss=0
        for t in reversed(range(0, time_steps)):
            dIn[t] = cost.d(out[t], predictions[t])
            loss += np.sum(cost(out[t], predictions[t]), axis=0)
        dW = [None] * self.num_of_layers()
        for l in reversed(range(0, self.num_of_layers())):
            dW[l], dIn, dCache = self.layers[l].time_grad(dIn, cache=cache[l], dCache=None)
        return loss, dW, cache_out

    def save(self, path):
        for i,l in enumerate(self.layers):
            np.save('%s_%d.npy'%(path,i),l.weights)

    def load(self, path):
        for i, l in enumerate(self.layers):
            l.weights=np.load('%s_%d.npy' % (path,i))

class FC:
    '''
        Fully Connected layer
    '''
    shape = None
    weights = None
    activation = None

    def reset(self):
        pass

    def __init__(self, shape, activation=identity()):
        self.shape = shape
        self.activation = activation

    def init(self, scale=1):
        self.weights = np.random.randn(self.shape[0] + 1, self.shape[1]) / np.sqrt(
            self.shape[0] + self.shape[1]) * scale
        self.weights[-1,] = 0

    def __call__(self, input_tensor):
        out, _ = self.eval_for_back_prop(input_tensor)
        return out

    def eval_for_back_prop(self, input_tensor):
        time_steps, batch, in_ = input_tensor.shape
        inputs = np.zeros([time_steps, batch, self.shape[0] + 1])
        outputs = np.zeros([time_steps, batch, self.shape[1]])
        raws = np.zeros([time_steps, batch, self.shape[1]])
        for t in range(0, time_steps):
            inputs[t, :, 0:-1] = input_tensor[t, :, :]
            inputs[t, :, -1] = 1
            raws[t] = inputs[t].dot(self.weights)
            outputs[t] = self.activation(raws[t])
        return outputs, {'inputs': inputs,
                         'raws': raws,
                         'outputs': outputs}

    def time_grad(self, dOut, cache, dCache=None):
        inputs = cache['inputs']
        outputs = cache['outputs']
        time_steps, batch, in_ = inputs.shape
        dIn = np.zeros([time_steps, batch, self.shape[0]])
        dW = np.zeros_like(self.weights)
        for t in reversed(range(0, time_steps)):
            dAct = dOut[t] * self.activation.d(outputs[t])
            '''
                the following line sums the gradients over the entire batch
                proof:
                Let x be a single input and dY a single gradient
                Than
                dW=np.outer(x,dY)=x.T*dY -> * stands for the dot product
                Now, we have (x_i) matrix and (dY_i) matrix
                dW_i=x_i.T*dY_i
                Our desired result is
                dW=dW_1+...+dW_n
                Thus
                dW=x_1.T*dY_1+...+x_n.T*dY_n
                which is precisely the matrix product
                dW=x.T*dY
                where x holds x_i as its rows and dY holds dY_i as its rows
                In other words, a product of two matrices is a sum
                of tensor products of columns and rows of the respected matrices
            '''
            dW += np.dot(inputs[t].T, dAct)
            dIn[t] = dAct.dot(self.weights.T)[:, 0:-1]
        return dW, dIn, None

class FCr:
    '''
        Fully Connected recursive layer
    '''
    shape = None
    weights = None
    previous = None
    activation = None

    def __init__(self, shape, activation=identity()):
        self.shape = shape
        self.activation = activation

    def init(self, scale=1):
        self.weights = np.random.randn(self.shape[0] + self.shape[1] + 1, self.shape[1]) / np.sqrt(
            self.shape[0] + self.shape[1]) * scale
        self.weights[-1, :] = 0

    def reset(self):
        self.previous = None

    def __call__(self, input_tensor):
        out, cache = self.eval_for_back_prop(input_tensor=input_tensor, h0=self.previous)
        self.previous = np.copy(cache['hn'])
        return out

    def eval_for_back_prop(self, input_tensor, h0=None):
        time_steps, batch, in_ = input_tensor.shape
        inputs = np.zeros([time_steps, batch, self.shape[0] + self.shape[1] + 1])
        raws = np.zeros([time_steps, batch, self.shape[1]])
        outputs = np.zeros([time_steps, batch, self.shape[1]])
        if h0 is None: h0 = np.zeros([batch, self.shape[1]])
        for t in range(0, time_steps):
            previous = outputs[t - 1] if t > 0 else h0
            inputs[t, :, -1] = 1
            inputs[t, :, 0:self.shape[0]] = input_tensor[t]
            inputs[t, :, self.shape[0]:-1] = previous
            raws[t] = inputs[t].dot(self.weights)
            outputs[t] = self.activation(raws[t])
        return outputs, {'inputs': inputs,
                         'raws': raws,
                         'outputs': outputs,
                         'h0': h0,
                         'hn': outputs[-1]}

    def time_grad(self, dOut, cache, dCache=None):
        inputs = cache['inputs']
        outputs = cache['outputs']
        time_steps, batch, out_ = outputs.shape
        dW = np.zeros(self.weights.shape)
        dInput = np.zeros(inputs.shape)
        dIn = np.zeros([time_steps, batch, self.shape[0]])
        dh0 = np.zeros([batch, self.shape[1]])
        dH = dOut.copy()
        if dCache is not None:
            dH[-1] += np.copy(dH['dHidden'])
        for t in reversed(range(0, len(dOut))):
            dAct = dH[t] * self.activation.d(outputs[t])
            '''
                the following line sums the gradients over the entire batch
                proof:
                Let x be a single input and dY a single gradient
                Than
                dW=np.outer(x,dY)=x.T*dY -> * stands for the dot product
                Now, we have (x_i) matrix and (dY_i) matrix
                dW_i=x_i.T*dY_i
                Our desired result is
                dW=dW_1+...+dW_n
                Thus
                dW=x_1.T*dY_1+...+x_n.T*dY_n
                which is precisely the matrix product
                dW=x.T*dY
                where x holds x_i as its rows and dY holds dY_i as its rows
                In other words, a product of two matrices is a sum
                of tensor products of columns and rows of the respected matrices
            '''
            dW += np.dot(inputs[t].T, dAct)
            dInput[t] = dAct.dot(self.weights.T)
            dIn[t] = dInput[t, :, 0:self.shape[0]]
            dh = dH[t - 1] if t > 0 else dh0
            dh += dInput[t, :, self.shape[0]:-1]
        return dW, dIn, {'dHidden': dh0}


class LSTM:
    '''
        Long Short Term Memory layer
        Paper can be found at:
        http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    '''
    shape = None
    weights = None
    previous = None
    cell = None

    def __init__(self, shape):
        self.shape = shape

    def init(self, forget_bias_init=3):
        '''
        forget bias initialization as seen in the paper
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
        section 2.2
        '''
        self.weights = np.random.randn(self.shape[0] + self.shape[1] + 1, 4 * self.shape[1]) / np.sqrt(
            self.shape[0] + self.shape[1])
        self.weights[-1, :] = 0
        if forget_bias_init != 0:
            self.weights[- 1, self.shape[1]:2 * self.shape[1]] = forget_bias_init
        self.previous = None
        self.cell = None

    def reset(self):
        self.previous = None
        self.cell = None

    def __call__(self, input_tensor):
        outputs, cache = self.eval_for_back_prop(input=input_tensor, c0=self.cell, h0=self.previous)
        self.cell = np.copy(cache['cn'])
        self.previous = np.copy(cache['hn'])
        return outputs

    def eval_for_back_prop(self, input, c0=None, h0=None, mask_start=0):
        time_steps, batch, in_ = input.shape
        inputs = np.zeros([time_steps, batch, self.shape[0] + self.shape[1] + 1])
        outputs = np.zeros([time_steps, batch, self.shape[1]])
        cells = np.zeros([time_steps, batch, self.shape[1]])
        cells_act = np.zeros([time_steps, batch, self.shape[1]])
        activations = np.zeros([time_steps, batch, 4 * self.shape[1]])
        if c0 is None: c0 = np.zeros([batch, self.shape[1]])
        if h0 is None: h0 = np.zeros([batch, self.shape[1]])
        for t in range(0, time_steps):
            previous = outputs[t - 1] if t > 0 else h0
            previous_cell = cells[t - 1] if t > 0 else c0
            inputs[t, :, -1] = 1
            inputs[t, :, 0:self.shape[0]] = input[t]
            inputs[t, :, self.shape[0]:-1] = previous
            raws_ = inputs[t].dot(self.weights[mask_start:, ])
            activations[t, :, 0:3 * self.shape[1]] = 1. / (1. + np.exp(-raws_[:, 0:3 * self.shape[1]]))
            activations[t, :, 3 * self.shape[1]:] = np.tanh(raws_[:, 3 * self.shape[1]:])
            cells[t] = activations[t, :, self.shape[1]:2 * self.shape[1]] * previous_cell + \
                       activations[t, :, 0:self.shape[1]] * activations[t, :, 3 * self.shape[1]:]
            cells_act[t] = np.tanh(cells[t])
            outputs[t] = activations[t, :, 2 * self.shape[1]:3 * self.shape[1]] * cells_act[t]
        return outputs, {'inputs': inputs,
                         'outputs': outputs,
                         'activations': activations,
                         'cells': cells,
                         'cells_act': cells_act,
                         'h0': h0,
                         'c0': c0,
                         'hn': outputs[-1],
                         'cn': cells[-1]}

    def time_grad(self, next_grad, cache, dCache=None, mask_start=0):
        inputs = cache['inputs']
        outputs = cache['outputs']
        activations = cache['activations']
        cell_act = cache['cells_act']
        cell = cache['cells']
        c0 = cache['c0']
        time_steps, batch, out_ = outputs.shape
        dAct = np.zeros(activations.shape)
        dW = np.zeros(self.weights.shape)
        dInput = np.zeros(inputs.shape)
        dCell = np.zeros(cell.shape)
        dIn = np.zeros([time_steps, batch, self.shape[0]])
        dh0 = np.zeros([batch, self.shape[1]])
        dc0 = np.zeros([batch, self.shape[1]])
        dH = next_grad.copy()
        if dCache is not None:
            dCell[-1] += dCache['dCell']
            dH[-1] += dCache['dHidden']
        for t in reversed(range(0, time_steps)):
            dCell[t] += (1 - cell_act[t] ** 2) * activations[t, :, 2 * self.shape[1]:3 * self.shape[1]] * dH[t]
            # dout
            dAct[t, :, 2 * self.shape[1]:3 * self.shape[1]] = cell_act[t] * dH[t]
            C_previous, dC_previous = (cell[t - 1], dCell[t - 1]) if t > 0 else (c0, dc0)
            # dforget
            dAct[t, :, self.shape[1]:2 * self.shape[1]] = C_previous * dCell[t]
            dC_previous += activations[t, :, self.shape[1]:2 * self.shape[1]] * dCell[t]
            # dwrite_i
            dAct[t, :, 0:self.shape[1]] = activations[t, :, 3 * self.shape[1]:] * dCell[t]
            # dwrite_c
            dAct[t, :, 3 * self.shape[1]:] = activations[t, :, 0:self.shape[1]] * dCell[t]
            # activations
            dAct[t, :, 0:3 * self.shape[1]] *= (1.0 - activations[t, :, 0:3 * self.shape[1]]) * activations[t, :,
                                                                                                0:3 * self.shape[1]]
            dAct[t, :, 3 * self.shape[1]:] *= (1.0 - activations[t, :, 3 * self.shape[1]:] ** 2)
            '''
                the following line sums the gradients over the entire batch
                proof:
                Let x be a single input and dY a single gradient
                Than
                dW=np.outer(x,dY)=x.T*dY -> * stands for the dot product
                Now, we have (x_i) matrix and (dY_i) matrix
                dW_i=x_i.T*dY_i
                Our desired result is
                dW=dW_1+...+dW_n
                Thus
                dW=x_1.T*dY_1+...+x_n.T*dY_n
                which is precisely the matrix product
                dW=x.T*dY
                where x holds x_i as its rows and dY holds dY_i as its rows
                In other words, a product of two matrices is a sum
                of tensor products of columns and rows of the respected matrices
            '''
            dW += np.dot(inputs[t].T, dAct[t])
            dInput[t] = dAct[t].dot(self.weights.T)
            dIn[t] = dInput[t, :, 0:self.shape[0]]
            dh = dH[t - 1] if t > 0 else dh0
            dh += dInput[t, :, self.shape[0]:-1]
        return dW, dIn, {'dHidden': dh0,
                         'dCell': dc0}