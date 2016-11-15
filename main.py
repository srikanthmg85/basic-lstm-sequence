# Single layer LSTM implementation using python numpy
# g = tanh(Wg*cat(h(t-1),x(t) + bg))
# ig = sigmoid(Wi*cat(h(t-1),x(t)) + bi)
# fg = sigmoid(Wf*cat(h(t-1),x(t)) + bf)
# og = sigmoid(Wo*cat(h(t-1),x(t)) + bo)
# Ct = tanh(ig.*g + fg.*Ct-1)
# h(t) = og.*Ct
# Single layer LSTM : y = Wy*h(t) + by
# p = softmax(y)


#              -----
#  h(t-1) ->   |   | -> h(t) -> softmax -> y(t)
#              -----
#                |
#               x(t)


import numpy as np 
import sys 
import os 
from random import uniform
import pdb

def sigmoid(in1):
  out = 1/(1+np.exp(-in1))
  return out


class LSTMUnit:

  def __init__(self,input_txt,num_hidden_units,seq_len,learning_rate,num_epochs):


    f = open(txtFile,'r')
    self.data = f.read()
    chars = list(set(self.data))
    num_data = len(self.data)
    num_chars = len(chars)
    self.char_to_index = {c:i for i,c in enumerate(chars)}
    self.index_to_char = {i:c for i,c in enumerate(chars)}
    self.num_hidden_units = num_hidden_units
    self.vocab_size = num_chars
    self.seq_len = seq_len
    self.num_hidden_units = num_hidden_units
    self.learning_rate = learning_rate     
    self.num_epochs = num_epochs

    cat_len = num_hidden_units + num_chars
    self.Wy = np.random.randn(num_chars,num_hidden_units)*0.01
    self.Wg = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wf = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wo = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wi = np.random.randn(num_hidden_units,cat_len)*0.01
    self.bg = np.zeros((num_hidden_units,1))
    self.bi = np.zeros((num_hidden_units,1))
    self.bf = np.zeros((num_hidden_units,1))
    self.bo = np.zeros((num_hidden_units,1))
    self.by = np.zeros((num_chars,1))

  def forward(self,inputs,outputs,hprev,cprev):

    seq_len = len(inputs)
    
    x= {}
    h = {}
    C = {}
    y= {}
    g = {}
    fg = {}
    og = {}
    ig = {} 
    xc = {}    

    probs = {}
    C[-1] = np.copy(cprev)
    h[-1] = np.copy(hprev)
    loss = 0


    for i in range(seq_len):
      x[i] = np.zeros((self.vocab_size,1))
      x[i][inputs[i]] = 1
      #pdb.set_trace()
      xc[i] = np.reshape(np.hstack((h[i-1].flatten(),x[i].flatten())),(self.num_hidden_units+self.vocab_size,1))
      g[i] = np.tanh(np.dot(self.Wg,xc[i]) + self.bg)
      ig[i] = sigmoid(np.dot(self.Wi,xc[i]) + self.bi) 
      fg[i] = sigmoid(np.dot(self.Wf,xc[i]) + self.bf)
      og[i] = sigmoid(np.dot(self.Wo,xc[i]) + self.bo)
      C[i] = g[i]*ig[i] + C[i-1]*fg[i]
      h[i] = C[i]*og[i]
      y[i] = np.dot(self.Wy,h[i]) + self.by
      probs[i] = np.exp(y[i])/np.sum(np.exp(y[i]))      
      loss += -np.log(probs[i][outputs[i]])

    hprev = h[seq_len-1]
    cprev = C[seq_len-1] 

    return x,h,C,y,g,fg,og,ig,xc,hprev,cprev,probs,loss 


  def backward(self,x,h,C,y,g,fg,og,ig,xc,probs,targets):
    seq_len = len(targets)

    dWg = np.zeros_like(self.Wg)
    dWi = np.zeros_like(self.Wi)
    dWf = np.zeros_like(self.Wf)
    dWo = np.zeros_like(self.Wo)
    dWy = np.zeros_like(self.Wy)
    dbg = np.zeros_like(self.bg)
    dbi = np.zeros_like(self.bi)
    dbf = np.zeros_like(self.bf)
    dbo = np.zeros_like(self.bo)
    dby = np.zeros_like(self.by) 

    dHnext = np.zeros((self.num_hidden_units,1))
    dCnext = np.zeros_like(dHnext)

    for i in reversed(range(seq_len)):
      dy = np.copy(probs[i])
      dy[targets[i]] -= 1
      
      dWy += np.dot(dy,h[i].T)       
      dby += dy

      dh = np.dot(self.Wy.T,dy) + dHnext       
      dog = dh*C[i] 

      dC = dh*og[i] + dCnext 
      dig = dC*g[i] 
      dfg = dC*C[i-1]

      dWi += (ig[i])*(1-ig[i])*np.dot(dig,xc[i].T)
      dWf += (fg[i])*(1-fg[i])*np.dot(dfg,xc[i].T)
      dWo += (og[i])*(1-og[i])*np.dot(dog,xc[i].T)
  
      dbo += dog*(og[i]*(1-og[i])) 
      dbi += dig*(ig[i]*(1-ig[i]))
      dbf += dfg*(fg[i]*(1-fg[i]))

      dg = dC*ig[i]

      dWg += np.dot(dg,xc[i].T)
      dXc = np.dot(self.Wg.T,(dg*(1-g[i]**2)))
      dbg += dg

      dCnext = dC*fg[i]
      dHnext = dXc[:num_hidden_units]

    return dWg,dWi,dWf,dWo,dWy,dbg,dbi,dbf,dbo,dby    

  def sample(self,start,hprev,cprev,num_chars):
    start_idx = start

    #h = np.zeros((self.num_hidden_units,1))
    idx = start_idx
    seq = [self.index_to_char[start]]

    for i in range(num_chars):
      x = np.zeros((self.vocab_size,1))
      x[idx] = 1
      xc = np.resize(np.hstack((hprev.flatten(),x.flatten())),(self.vocab_size+self.num_hidden_units,1))
      g = np.tanh(np.dot(self.Wg,xc))
      ig = sigmoid(np.dot(self.Wi,xc))
      fg = sigmoid(np.dot(self.Wf,xc))
      og = sigmoid(np.dot(self.Wo,xc))
      c1 = g*ig + cprev*fg
      h1 = c1*og
       
      logits = np.dot(self.Wy, h1) + self.by
      probs = np.exp(logits)/sum(np.exp(logits))
      hprev = np.copy(h1)
      cprev = np.copy(c1)
      idx = np.random.choice(range(self.vocab_size),p=probs.ravel())
      seq.append(self.index_to_char[idx])

    return seq

  

  def gradCheck(self,inputs, target, hprev,cprev):

    global Wxh, Whh, Why, bh, by
    num_checks, delta = 10, 1e-5
    x,h,C,y,g,fg,og,ig,xc,_,_,probs,loss = self.forward(inputs,target,hprev,cprev)
    dWg,dWi,dWf,dWo,dWy,dbg,dbi,dbf,dbo,dby = self.backward(x,h,C,y,g,fg,og,ig,xc,probs,target)

    for param,dparam,name in zip([self.Wg,self.Wi, self.Wf, self.Wo, self.Wy, self.bg,self.bi,self.bf,self.bo,self.by], [dWg,dWi, dWf, dWo, dWy, dbg,dbi,dbf,dbo,dby], ['Wg', 'Wi', 'Wf', 'Wo', 'Wy','bg','bi','bf','bo','by']):
      s0 = dparam.shape
      s1 = param.shape
      assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
      print name
      if True:
        for i in xrange(num_checks):
          ri = int(uniform(0,param.size))
          # evaluate cost at [x + delta] and [x - delta]
          old_val = param.flat[ri]
          param.flat[ri] = old_val + delta

          _,_,_,_,_,_,_,_,_,_,_,_,cg0 = self.forward(inputs,target,hprev,cprev)

          param.flat[ri] = old_val - delta

          _,_,_,_,_,_,_,_,_,_,_,_,cg1 = self.forward(inputs,target,hprev,cprev)

          param.flat[ri] = old_val # reset old value for this parameter
          # fetch both numerical and analytic gradient
          grad_analytic = dparam.flat[ri]
          grad_numerical = (cg0 - cg1) / ( 2 * delta )
          rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
          print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
          # rel_error should be on order of 1e-7 or less

 
txtFile = sys.argv[1]
num_hidden_units = 100
seq_len = 25
lstm = LSTMUnit(txtFile,num_hidden_units,seq_len,0.1,100)

hprev = np.zeros((lstm.num_hidden_units,1))
cprev = np.zeros_like(hprev)

inputs = [lstm.char_to_index[ch] for ch in lstm.data[0*lstm.seq_len:(0+1)*lstm.seq_len]]
outputs= [lstm.char_to_index[ch] for ch in lstm.data[0*lstm.seq_len + 1:(0+1)*lstm.seq_len + 1]]
lstm.gradCheck(inputs,outputs,hprev,cprev)
     
 
count = 0

mWg = np.zeros_like(lstm.Wg)
mWi = np.zeros_like(lstm.Wi)
mWf = np.zeros_like(lstm.Wf)
mWo = np.zeros_like(lstm.Wo)
mWy = np.zeros_like(lstm.Wy)
mbg = np.zeros_like(lstm.bg)
mbi = np.zeros_like(lstm.bi)
mbf = np.zeros_like(lstm.bf)
mbo = np.zeros_like(lstm.bo)
mby = np.zeros_like(lstm.by)


for j in range(lstm.num_epochs):

  for i in range(len(lstm.data)/lstm.seq_len):

    if i == 0:
      hprev = np.zeros((lstm.num_hidden_units,1))
      cprev = np.zeros_like(hprev)

    inputs = [lstm.char_to_index[ch] for ch in lstm.data[i*lstm.seq_len:(i+1)*lstm.seq_len]]
    outputs= [lstm.char_to_index[ch] for ch in lstm.data[i*lstm.seq_len + 1:(i+1)*lstm.seq_len + 1]]

    
    x,h,C,y,g,fg,og,ig,xc,hprev,cprev,probs,loss = lstm.forward(inputs,outputs,hprev,cprev)
    dWg,dWi,dWf,dWo,dWy,dbg,dbi,dbf,dbo,dby = lstm.backward(x,h,C,y,g,fg,og,ig,xc,probs,outputs)

    mWg += dWg*dWg
    mWi += dWi*dWi
    mWf += dWf*dWf
    mWo += dWo*dWo
    mWy += dWy*dWy
    mbg += dbg*dbg
    mbi += dbi*dbi
    mbf += dbf*dbf
    mbo += dbo*dbo
    mby += dby*dby


    lstm.Wg -= lstm.learning_rate*dWg/(np.sqrt(mWg + 1e-8))
    lstm.Wi -= lstm.learning_rate*dWi/(np.sqrt(mWi + 1e-8))
    lstm.Wf -= lstm.learning_rate*dWf/(np.sqrt(mWf + 1e-8))
    lstm.Wo -= lstm.learning_rate*dWo/(np.sqrt(mWo + 1e-8))
    lstm.Wy -= lstm.learning_rate*dWy/(np.sqrt(mWy + 1e-8))
    lstm.bg -= lstm.learning_rate*dbg/(np.sqrt(mbg + 1e-8))
    lstm.bi -= lstm.learning_rate*dbi/(np.sqrt(mbi + 1e-8))
    lstm.bf -= lstm.learning_rate*dbf/(np.sqrt(mbf + 1e-8))
    lstm.bo -= lstm.learning_rate*dbo/(np.sqrt(mbo + 1e-8))
    lstm.by -= lstm.learning_rate*dby/(np.sqrt(mby + 1e-8))


    if count%100 == 0:
      seq = lstm.sample(inputs[0],hprev,cprev,50)
      txt = ''.join(ix for ix in seq)
      print txt
    count += 1





