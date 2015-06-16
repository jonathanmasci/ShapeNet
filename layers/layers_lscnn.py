"""
Layers for LSCNN models.
Their interface differs from conventional models as they need as input:
    - the descriptor
    - the LBO eigenfuncitons
    - the area vector
    - the LBO eigenvalues
"""

import sys
import theano.tensor
import theano.tensor as T
import numpy as np
import theano
import theano.sparse as Tsp
from utils import *

import theano.tensor.nnet.conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.basic_ops import gpu_from_host

class ILSCNNLayer:
  """ 
  Minimal interface for a LSCNN layer
  """
  
  def fwd(self, x, V, A, L):
    """
    x: input signal
    V: eigenvectors (Phi)
    A: area
    L: eigenvalues (Lambda)
    """
    raise NotImplementedError(str(type(self)) + " does not implement fwd.")

  def get_params(self):
    raise NotImplementedError(str(type(self)) + " does not implement get_params.")

  def set_params(self, w):
    raise NotImplementedError(str(type(self)) + " does not implement set_params.")   

class LSCNNLayer(ILSCNNLayer):
  def __init__(self, rng, nin, nout, K, taus, layer_name, 
          irange=0.001, activation=relu, init_uniform=True):
    self.rng = rng
    self.nin = nin
    self.nout = nout
    self.K = K
    self.layer_name = layer_name
    self.nfilters = len(taus)
    self.taus = theano.shared(value=np.asarray(taus).astype(np.float32), name=layer_name+'-taus', borrow=True)
    self.activation = activation
    self.init_uniform = init_uniform

    if self.init_uniform:
      W_values = np.asarray(rng.uniform(
              low=-irange,
              high=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
    else:
      W_values = np.asarray(rng.normal(
              loc=0.,
              scale=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((self.nout,), dtype=theano.config.floatX) + 0.5
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b
      
    # parameters of the model
    self.params = [self.W, self.b]

  def sample_ghats(self, taus, L):
    """
    Returns a D x K matrix where K is the number of eigenvalues (L) and D is
    the number of taus; thus each row is a window
    """
    eps = 1e-5
    ghats = T.exp(-(taus.dimshuffle(0,'x') * T.abs_(L[0] - L).dimshuffle('x',0)))
    ghats /= T.sqrt(T.sum(T.sqr(ghats), axis=1) + eps).dimshuffle(0,'x')
    
    return ghats

  def sample_ghat(self, taus, L):
    """
    Samples ghat as linear combination of various ghats parameterized by taus
    """
    return T.sum(self.sample_ghats(taus, L), axis=0)

  def fwd(self, x, V, A, L):
    """
    x : signal
    V : eigenvectors
    A : area 
    L : eigenvalues
    """
    V = V[:,:self.K]
    L = L[:self.K]

    # ghat is already a linear combination. it is faster than doing a
    # traslation and modulation each time of course, everything is linear
    # and thus it can be done
    ghat = self.sample_ghat(self.taus, L)

    rho = T.sqrt(T.sum(A))
    # N x N : large but still doable 880Mb for N=15K
    trasl = rho * T.dot(V, ghat.dimshuffle(0,'x') * V.T)
    trasl = A.dimshuffle(0,'x') * trasl
    
    def step(f,  rho, V, trasl):
      return rho * T.dot((V * f.dimshuffle(0,'x')).T, trasl)    # N x K

    desc, _ = theano.scan(fn=step, non_sequences=[rho,V,trasl], 
        sequences=[x.T])
    desc = desc.dimshuffle(2,0,'x',1) # BC01 format : N x Q x 1 x K
    desc = T.abs_(desc)
    return self.activation(theano.tensor.nnet.conv.conv2d(desc, self.W).flatten(2) + self.b)


  def fwd_old(self, x, V, A, L):
    """
    x : signal
    V : eigenvectors
    A : area 
    L : eigenvalues
    """
    V = V[:,:self.K]
    L = L[:self.K]

    # ghat is already a linear combination. it is faster than doing a
    # traslation and modulation each time of course, everything is linear
    # and thus it can be done
    ghat = self.sample_ghat(self.taus, L)

    rho = T.sqrt(T.sum(A))
    trasl = rho * T.dot(V, ghat.dimshuffle(0,'x') * V.T)
    trasl = A.dimshuffle(0,'x') * trasl
    
    # size Q x K x N, intermediate N x Q x K
    tmp = (V.dimshuffle(0,'x',1) * x.dimshuffle(0,1,'x')).dimshuffle(1,2,0)
    trasl = T.tile(trasl.dimshuffle('x',0,1), [self.nin,1,1])
    # size Q x K x N
    desc = rho * T.batched_dot(tmp, trasl)
    desc = T.abs_(desc)

    desc = desc.dimshuffle(2,0,'x',1) # BC01 format : N x Q x 1 x K
    
    return self.activation(theano.tensor.nnet.conv.conv2d(desc, self.W).flatten(2) + self.b)

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

class LSCNNLayerBSpline(ILSCNNLayer):
  def __init__(self, rng, nin, nout, K, minL, maxL, layer_name, 
          irange=0.001, activation=relu, init_uniform=True):
    self.rng = rng
    self.nin = nin
    self.nout = nout
    self.K = K
    self.minL = minL
    self.maxL = maxL
    self.layer_name = layer_name
    self.activation = activation
    self.init_uniform = init_uniform

    self.evalSamples = np.linspace(self.minL, self.maxL, self.K).astype(np.float32)
    self.dEval = theano.shared(self.evalSamples[1] - self.evalSamples)
    self.evalSamples = theano.shared(self.evalSamples)

    if self.init_uniform:
      W_values = np.asarray(rng.uniform(
              low=-irange,
              high=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
    else:
      W_values = np.asarray(rng.normal(
              loc=0.,
              scale=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((self.nout,), dtype=theano.config.floatX) + 0.5
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    beta_values = np.random.normal(size=(self.nin, self.K, 1),
            loc=0, scale=irange).astype(theano.config.floatX)
    beta = theano.shared(value=beta_values*0+1, name=layer_name+'-beta', borrow=True)

    self.beta = beta
    self.W = W
    self.b = b
      
    # parameters of the model
    self.params = [self.beta, self.W, self.b]
  
  def cubicBSpline(self, L):
    b = T.zeros_like(L)

    idx4 = T.ge(L, 0) * T.lt(L, 1)
    idx3 = T.ge(L, 1) * T.lt(L, 2)
    idx2 = T.ge(L, 2) * T.lt(L, 3)
    idx1 = T.ge(L, 3) * T.le(L, 4)

    b = T.switch(T.eq(idx4, 1), T.pow(L, 3) / 6, b)
    b = T.switch(T.eq(idx3, 1), (-3*T.pow(L-1,3) + 3*T.pow(L-1,2) + 3*(L-1) + 1) / 6, b)
    b = T.switch(T.eq(idx2, 1), ( 3*T.pow(L-2,3) - 6*T.pow(L-2,2)           + 4) / 6, b)
    b = T.switch(T.eq(idx1, 1), (-  T.pow(L-3,3) + 3*T.pow(L-3,2) - 3*(L-3) + 1) / 6, b)
    
    return b.T # b is K x K' and thus, as we multiply from the right with
               # betas, we need to transpose it

  def fwd(self, x, V, A, L):
    """
    x : signal
    V : eigenvectors
    A : area 
    L : eigenvalues
    """
    V = V[:,:self.K]
    L = L[:self.K]
    
    sampleLoc = (L.dimshuffle(0,'x') - self.evalSamples.dimshuffle('x',0)) / self.dEval
    basis = self.cubicBSpline(sampleLoc)
    basis = basis.dimshuffle('x',0,1)
  
    rho = T.sqrt(T.sum(A))
  
    def step(f, beta,   rho, A, V):
      ghat = T.dot(basis, beta.squeeze()).flatten()
      transl = rho * T.dot(V, ghat.dimshuffle(0,'x') * V.T)
      return rho * T.dot((V * f.dimshuffle(0,'x')).T, A.dimshuffle(0,'x') * transl)    # N x K
  
    desc, _ = theano.scan(fn=step, non_sequences=[rho,A,V], 
        sequences=[x.T,self.beta])
    desc = desc.dimshuffle(2,0,'x',1) # BC01 format : N x Q x 1 x K
    desc = T.abs_(desc)
    return self.activation(theano.tensor.nnet.conv.conv2d(desc, self.W).flatten(2) + self.b)

  def fwd_old(self, x, V, A, L):
    """
    x : signal
    V : eigenvectors
    A : area 
    L : eigenvalues
    """
    V = V[:,:self.K]
    L = L[:self.K]

    sampleLoc = (L.dimshuffle(0,'x') - self.evalSamples.dimshuffle('x',0)) / self.dEval
    basis = self.cubicBSpline(sampleLoc)
    basis = basis.dimshuffle('x',0,1)

    rho = T.sqrt(T.sum(A))

    # weight the basis columns for each input function to generate a ghat
    # Q x K, a window for each input function
    ghat = T.squeeze(T.batched_dot(
            T.tile(basis, [self.nin, 1, 1]), 
            self.beta)[:,:,0]) # crazy stuff here, why doesn't squeeze work?
    # Q x K x N
    V_ = T.tile(V.dimshuffle('x',1,0), [self.nin, 1, 1])
    # Q x K x N
    tmp = (ghat.dimshuffle(0,'x',1) * V).dimshuffle(0,2,1)
    # Q x N x N
    transl = rho * T.batched_dot(V_.dimshuffle(0,2,1), tmp)
    transl = A.dimshuffle('x',0,'x') * transl
    # Q x K x N
    tmp = (V.dimshuffle(0,'x',1) * x.dimshuffle(0,1,'x')).dimshuffle(1,2,0)
    # Q x K x N
    desc = rho * T.batched_dot(tmp, transl)
    desc = T.abs_(desc)
    
    desc = desc.dimshuffle(2,0,'x',1) # BC01 format : N x Q x 1 x K
    return self.activation(theano.tensor.nnet.conv.conv2d(desc, self.W).flatten(2) + self.b)

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.beta.set_value(w[0])
    self.W.set_value(w[1])
    self.b.set_value(w[2])

class LSCNNLayerLearnInterp(ILSCNNLayer):
  def __init__(self, rng, nin, nout, K, layer_name, 
          irange=0.001, activation_interp=relu, activation=relu, init_uniform=True):
    self.rng = rng
    self.nin = nin
    self.nout = nout
    self.K = K
    self.layer_name = layer_name
    self.activation = activation
    self.activation_interp = activation_interp
    self.init_uniform = init_uniform

    if self.init_uniform:
      W_values = np.asarray(rng.uniform(
              low=-irange,
              high=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
    else:
      W_values = np.asarray(rng.normal(
              loc=0.,
              scale=irange,
              size=(self.nout, self.nin, 1, self.K)), dtype=theano.config.floatX)
      
    Winterp_values = np.random.normal(loc=0.,scale=irange,
            size=(self.nin, self.K, self.K)).astype(np.float32)

    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)
    Winterp = theano.shared(value=Winterp_values, name=layer_name+'-Winterp', borrow=True)

    b_values = np.zeros((self.nout,), dtype=theano.config.floatX) + 0.5
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)
    
    self.Winterp = Winterp
    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.Winterp, self.W, self.b]

  def fwd(self, x, V, A, L):
    """
    x : signal
    V : eigenvectors
    A : area 
    L : eigenvalues
    """
    V = V[:,:self.K]
    L = L[:self.K]

    L = L.dimshuffle('x','x',0)

    rho = T.sqrt(T.sum(A))
   
    # Q x 1 x K, a window for each input function
    ghat = self.activation_interp(
            T.batched_dot(T.tile(L, [self.nin,1,1]), self.Winterp))
    # Q x K x N
    V_ = T.tile(V.dimshuffle('x',1,0), [self.nin, 1, 1])
    # Q x K x N
    tmp = (ghat * V).dimshuffle(0,2,1)
    
    # Q x N x N
    transl = rho * T.batched_dot(V_.dimshuffle(0,2,1), tmp)
    transl = A.dimshuffle('x',0,'x') * transl
    
    # Q x K x N
    tmp = (V.dimshuffle(0,'x',1) * x.dimshuffle(0,1,'x')).dimshuffle(1,2,0)
    # Q x K x N
    desc = rho * T.batched_dot(tmp, transl)
    desc = T.abs_(desc)
    
    desc = desc.dimshuffle(2,0,'x',1) # BC01 format : N x Q x 1 x K
    return self.activation(theano.tensor.nnet.conv.conv2d(desc, self.W).flatten(2) + self.b)

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.Winterp.set_value(w[0])
    self.W.set_value(w[1])
    self.b.set_value(w[2])

class LSCNNMLPLayer(ILSCNNLayer):
  def __init__(self, rng, n_in, n_out, layer_name, activation=None):
    self.rng = rng
    self.n_in = n_in
    self.n_out = n_out
    self.layer_name = layer_name
    self.activation = activation

    W_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (n_in + n_out)),
               high=np.sqrt(6. / (n_in + n_out)),
               size=(n_in, n_out)), dtype=theano.config.floatX)
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX) + np.float32(0.5)
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.W, self.b]

  def fwd(self, x, V, A, L):
    x = gpu_contiguous(x)
    if x.ndim == 4:
        x = x.flatten(2)
    lin_output = T.dot(x, self.W) + self.b

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

class LSCNNUnitLengthLayer(ILSCNNLayer):
  def __init__(self, layer_name):
    self.layer_name = layer_name

    # parameters of the model
    self.params = []

  def fwd(self, x, V, A, L):
    eps = 1e-5

    x = gpu_contiguous(x)
    if x.ndim == 4:
        x = x.flatten(2)
    return x / (T.sqrt(T.sum(T.sqr(x+eps), axis=1)) + eps)

  def get_params(self):
    return self.params

  def set_params(self, w):
    pass

