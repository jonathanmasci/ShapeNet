import os
import theano.tensor
import theano.tensor as T
import sys
import numpy as np
import theano
import theano.tensor.nnet.conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import cPickle
from theano.tensor.signal.downsample import max_pool_2d
import time
from theano.ifelse import ifelse


def _dropout_from_layer(rng, layer, p):
  """
  p is the probablity of dropping a unit. from https://github.com/mdenil/dropout/blob/master/mlp.py
  """
  srng = theano.tensor.shared_randomstreams.RandomStreams(
      rng.randint(999999))
  # p=1-p because 1's indicate keep and p is prob of dropping
  # The cast is important because
  # int * float32 = float64 which pulls things off the gpu
  mask = theano.tensor.cast(srng.binomial(n=1, p=1.-p, size=layer.shape), theano.config.floatX)

  output = layer * mask
  return output, mask

def zeropad(x, p):
  x_shape_pad = [x.shape[0], x.shape[1], x.shape[2] + p[0], x.shape[3] + p[1]]
  x_ = T.zeros((x_shape_pad))
  x_ = T.set_subtensor(x_[:,:,p[0]//2:p[0]//2+x.shape[2],p[1]//2:p[1]//2+x.shape[3]], x)
  return x_, x_shape_pad

def relu(x):
  return T.maximum(0., x)

def relu_delve(x, alpha=0.01):
  return T.switch(T.lt(x, 0.), alpha * x, x)

def linear_activation(x):
  return x

def enforce_max_col_norm(updated_W, max_col_norm):
  """
  check maxout.py, censor_updates [line 225]
  """
  col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0) + 1e-7)
  desired_norms = T.clip(col_norms, 0, max_col_norm)
  return updated_W * (desired_norms / (1e-7 + col_norms))

def enforce_max_kernel_norm(W, m):
  k_norms = T.sqrt(T.sum(T.sqr(W), axis=(2,3)) + 1e-7).dimshuffle(0,1,'x','x')
  desired_norms = T.clip(k_norms, 0., m)

  return W * (desired_norms / (1e-7 + k_norms))
  
def enforce_sum_to_one_beta(W):
  k_norms = (T.sum(W, axis=1) + 1e-7).dimshuffle(0,'x',1)

  return W / (1e-7 + k_norms)

def filtered_list_dir(path, extensions):
    """
    Returns the list of files in path which end with any of the given extensions
    path: string
    extensions: list of string
    """
    # return os.listdir(path)
    tmp = [fn for fn in os.listdir(path) if not any([fn.endswith(ext) for ext in extensions])]
    return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in extensions])]

