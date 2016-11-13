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
#  h(t-1) ->   |   | -> h(t)
#              -----
#                |
#               x(t)


import numpy as np 
import sys 
import os 


def sigmoid(in):
  out = 1/(1+np.exp(-in))
  return out


class LSTMUnit:

  def __init__(num_hidden_units,input_dim):
    
    cat_len = num_hidden_units + input_dim
    self.Wg = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wf = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wo = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wi = np.random.randn(num_hidden_units,cat_len)*0.01
    self.bg = np.zeros(num_hidden_units,1)
    self.bi = np.zeros(num_hidden_units,1)
    self.bf = np.zeros(num_hidden_units,1)
    self.bo = np.zeros(num_hidden_units,1)

  def forward(inputs,outputs,hprev,cprev)

    seq_len = len(inputs)
    
    x= {}
    h = {}
    C = {}
    y= {}
    probs = {}
    C[-1] = cprev
    h[-1] = hprev
    loss = 0


    for i in range(seq_len):
      x = np.zeros(self.vocab_size,1)
      x[i][inputs[i]] = 1
      xc = np.hstack(h[i-1],x[i])
      g = np.tanh(np.dot(self.Wg,xc) + self.bg)
      ig = sigmoid(np.dot(self.Wi,xc) + self.bi) 
      fg = sigmoid(np.dot(self.Wf,xc) + self.bf)
      og = sigmoid(np.dot(self.Wo,xc) + self.bo)
      C[i] = g*ig + C[i-1]*fg
      h[i] = C[i]*og
      y[i] = np.dot(self.Wy*h[i] + by)
      probs = np.exp(y[i])/np.sum(np.exp(y[i]))      
      loss += -np.log(probs[i][outputs[i])

    hprev = h[seq_len-1]
    cprev = C[seq_len-1] 

    return C,x,sprev,probs,loss 


      
       





