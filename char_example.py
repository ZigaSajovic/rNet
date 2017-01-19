import rNet as rNet
import numpy as np
import urllib.request
import os.path

def generate(net,seed_,num_to_gen):
    net.reset()
    x=np.zeros((1,1,vocab_size))
    x[0,0,seed_]=1
    out=index_to_char[seed_]
    for t in range(0,num_to_gen):
        p=net(x)[0,0,:]
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x=np.zeros(x.shape)
        x[0,0,ix]=1
        out+=index_to_char[ix]
    return out


path_='data/tiny_shakespeare.txt'
# path_='data/tiny_nietzsche.txt'

raw=open(path_, 'r').read()

chars = list(set(raw))
chars.sort()
data_size, vocab_size = (len(raw), len(chars))
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

time_steps, batch_size, input_size, hidden_size, output_size = (100, 100, vocab_size, 512, vocab_size)

net = rNet.rNet()
net.add(rNet.LSTM([input_size,hidden_size]))
net.add(rNet.FC([hidden_size,output_size],activation=rNet.softmax()))

net.init()
# where to save the model
model_path='model/L'
cost = rNet.softmax_loss()

# settings for RMSprop + momentum
first_moment=[np.zeros_like(l.weights) for l in net.layers]
second_moment=[np.zeros_like(l.weights) for l in net.layers]
momentum=[np.zeros_like(l.weights) for l in net.layers]
smooth_loss = -np.log(1.0/vocab_size)*time_steps * batch_size # loss at iteration 0
cache0=None

count, count_t=(0,0)
epoch=0
text_pointers = np.random.randint(data_size-time_steps-1, size=batch_size)
learning_rate, nu, mom_decay,=(1e-3, 0.97, 0.9)
clip_range=(-5,5)
print('Learning rate: %f, nu: %f, mom_decay: %f'%(learning_rate,nu,mom_decay))
print('Clip range: ',clip_range)
while True:
    # reset the state every 100 sequences (10000 characters) and save the model
    if count % 100 == 0:
        cache0 = None
        print('Cache cleared')
        net.save(model_path)
        print('Model saved in %s'%(model_path))
    for i in range(text_pointers.size):
        if text_pointers[i] + time_steps + 1 >= data_size:
            text_pointers[i] = 0
    batch_in=np.zeros([time_steps, batch_size, vocab_size])
    batch_out=np.zeros([time_steps, batch_size], dtype=np.uint8)
    for i in range(batch_size):
        b_=[char_to_index[c] for c in raw[text_pointers[i]:text_pointers[i] + time_steps + 1]]
        batch_in[range(time_steps),i,b_[:-1]]=1
        batch_out[:,i]=np.array(b_[1:])
    loss,dW, cache0 = net.train_step(batch_in,batch_out, cache0=cache0, cost=cost)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if count % 10 == 0:
        txt = generate(net, np.random.randint(vocab_size), 200)
        print('----\n %s \n----' % (txt))
        print('epoch: %d, iter %d, smooth loss: %f, loss: %f' % (epoch, count, smooth_loss/(time_steps*batch_size),loss/(time_steps*batch_size)))
    # RMSprop + momentum parameter update
    for param, dparam, mem, mem1, mom in zip(net.layers, dW, second_moment,first_moment,momentum):
        np.clip(dparam,clip_range[0],clip_range[1],dparam)
        mem = nu*mem + (1-nu)*dparam * dparam
        mem1 = nu * mem1 + (1-nu) * dparam
        mom=mom_decay*mom-learning_rate * dparam / np.sqrt(mem - mem1**2 + 1e-8)
        param.weights += mom
    text_pointers += time_steps
    count_t+=time_steps
    count += 1
    if count_t >= data_size:
        epoch += 1
        count_t=0
        text_pointers = np.random.randint(data_size - time_steps - 1, size=batch_size)