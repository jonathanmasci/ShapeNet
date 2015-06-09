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
from layers.utils import _dropout_from_layer

class LSCNN(object):
  def __init__(self, rng, layers, drop_list):
    self.rng = rng
    self.layers = layers
    self.drop_list = drop_list
    self.w_averages = []
   
    # It can be handy to keep cost and input/target 
    self.cost = None
    self.inputs = []
    self.targets = []

  def fwd(self, x, V, A, L):
    acts = [x]
    for i in xrange(len(self.layers)):
      acts.append(self.layers[i].fwd(
          acts[-1] * (np.float32(1.0) - np.float32(self.drop_list[i])), 
          V, A, L)
          )

    return acts[-1], acts

  def fwd_dropout(self, x, V, A, L, masks=[]):
    """
    mask: if not empty gives the dropout masks. it is used in case of
          siamese nets where the same mask needs to be used.
    """
    if len(masks) == 0:
      compute_masks = True
    else:
      compute_masks = False

    acts = [x]
    for i in xrange(len(self.layers)):
      if compute_masks:
        d_x, m = _dropout_from_layer(self.rng, acts[-1], self.drop_list[i])
        masks.append(m)
      acts.append(self.layers[i].fwd(acts[-1] * masks[i], V, A, L))
   
    return acts[-1], acts, masks

  def get_params(self, to_cpu=False):
    p = []
    for l in self.layers:
      p.extend(l.get_params())
    if to_cpu:
      for i in xrange(len(p)):
        p[i] = p[i].get_value(borrow=False)

    return p

  def print_layers(self):
    for i in self.layers:
      print i.layer_name

  def monitor_params(self, logfun=lambda x : sys.stdout.write(x)):
    logfun('\n===== Monitor =====')
    for i in self.get_params():
      tmp = i.get_value()
      logfun("%s : minv %f maxv %f L2 %f" % (i.name, tmp.min(), tmp.max(), np.linalg.norm(tmp)))
    logfun('===================')

  def set_params(self, w):
    for i in xrange(len(w)):
      if not isinstance(w[i], np.ndarray):
        w[i] = np.asarray(w[i].get_value())

    j = 0
    for i, l in enumerate(self.layers):
      print "Processing layer %i" % i
      if len(l.get_params()) == 0:
        print "...this layer has no parameters"
        continue
      l.set_params(w[j:j+len(l.get_params())])
      j += len(l.get_params())

  def save(self, fname):
    f = file("%s.pkl" % fname, 'wb')
    cPickle.dump(self.get_params(to_cpu=True), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

  def load(self, fname):
    print "Make sure the model has the same structure! Here we load just the parameters!"
    f = file(fname, 'rb')
    tmp = cPickle.load(f)
    f.close()
    self.set_params(tmp)    

