import theano.tensor as T
import numpy as np

def triplets_loss(p0, p1, n0, n1, alpha):
  dp = T.sqrt(T.sum(T.sqr(p0 - p1), axis=1) + 0.0001)
  dn = T.sqrt(T.sum(T.sqr(n0 - n1), axis=1) + 0.0001)

  return T.mean(T.maximum(dp + alpha - dn, 0.))

def siamese_loss(xp0, xp1, xn0, xn1, margin, alpha=0.5):
  assert(alpha > 0 and alpha < 1)
  eps = 1e-5
  lp = 0.5 * T.sum(T.sqr(xp0 - xp1), axis=1) 
  ln = T.sqrt(T.sum(T.sqr(xn0 - xn1), axis=1) + eps)
  ln = 0.5 * T.sqr(T.maximum(0., margin - ln))

  return (1. - alpha) * lp.mean() + alpha * ln.mean()

def shape_net_loss(x, y, srng, margin, alpha=0.5):
  assert(alpha > 0 and alpha < 1)
  eps = 1e-5
  lp = 0.5 * T.sum(T.sqr(x - y), axis=1)
  prm = srng.permutation(n=x.shape[0])
  ln = T.sqrt(T.sum(T.sqr(x - y[prm]), axis=1) + eps)
  ln = 0.5 * T.sqr(T.maximum(0., margin - ln))

  return (1. - alpha) * lp.mean() + alpha * ln.mean()

