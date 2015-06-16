import os
import sys
import time
import zmq
import random
import numpy as np
import scipy.misc
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cPickle
import logging
import yaml
import theano
import theano.tensor as T
from collections import OrderedDict
from layers import *
from itertools import izip

def update_sgd_mom(parameters, gradients, headings, blah, w_constraints, lrate=0.01, mom=0.9,
        fixed_lrate_upds=100, lrate_anneal=0.997, min_lrate=0.00001):
  if headings is None:
    headings = [theano.shared((p * np.float32(0.0)).eval()) for p in parameters]

  updates = OrderedDict()
  for param_i, grad_i, head_i, wcon in zip(parameters, gradients, headings, w_constraints):
    updates[head_i] = mom * head_i - \
            T.maximum(min_lrate, lrate * lrate_anneal**T.maximum(0.0, fixed_lrate_upds)) * grad_i
    updates[param_i] = wcon(param_i + head_i)

  return updates, headings, headings

def adadelta_updates(parameters, gradients, gradients_sq=None, deltas_sq=None, rho=0.95, eps=1e-6):
  # https://blog.wtf.sg/2014/08/28/implementing-adadelta/
  
  if gradients_sq == None:
    # create variables to store intermediate updates
    gradients_sq = [ theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in parameters ]
    deltas_sq = [ theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in parameters ]

  # calculates the new "average" delta for the next iteration
  gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

  # calculates the step in direction. The square
  # root is an approximation to getting the RMS for
  # the average value
  deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

  # calculates the new "average" deltas for the next step.
  deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]
               
  # Prepare it as a list f
  gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
  deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
  parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
                                                          
  return [gradient_sq_updates + deltas_sq_updates + parameters_updates, gradients_sq, deltas_sq]


def train(model, cost, perf_mon, conf, out_path, exp_name, iter_eval, logging,
    queue_size, inport, fetch_data, n_epochs=500, n_batches_valid=30, n_train_to_eval=10):
  logging.info("Getting the gradient...")
  start_time = time.time()
  print "...Model parameters:",
  print model.get_params()
  grads = T.grad(cost, model.get_params()) 
  model.grads = grads
  logging.info("...done %f" % (time.time() - start_time))

  logging.info("Checking if there exist a training checkpoint...")
  gradients_sq, deltas_sq = None, None
  if os.path.isfile(os.path.join(out_path, "%s-best-model.pkl" % exp_name)):
    logging.info("...loading checkpoint")
    f = file(os.path.join(out_path, "%s-train-checkpoint.pkl" % exp_name))
    n_updates, gradients_sq, deltas_sq, rnd_state = cPickle.load(f)
    np.random.set_state(rnd_state)
    logging.info("...starting from n_updates %i", n_updates.get_value())
    f.close()
  else:
    n_updates = theano.shared(np.float32(0.0), name='n_updates')
    logging.info("...starting fresh")
  
  logging.info("Creating update rules...")
  updates, gradients_sq, deltas_sq = adadelta_updates(model.get_params(), grads, gradients_sq, deltas_sq)
  updates.append([n_updates, n_updates + 1.0])
  #updates, gradients_sq, deltas_sq = update_sgd_mom(model.get_params(), grads,
  #        gradients_sq, deltas_sq, model.w_constraints, lrate=0.1, mom=0.9, 
  #        fixed_lrate_upds=100, lrate_anneal=0.997)
  #updates[n_updates] = n_updates + 1.0
  logging.info("...done %f" % (time.time() - start_time))


  # Maximizing a cost function!
  best_perf = -np.inf

  logging.info("Creating train function...")
  start_time = time.time()
  print model.inputs
  train_model = theano.function(inputs=model.inputs, 
          outputs=cost, 
          updates=updates,
          on_unused_input='warn')
    
  model_out = model.fwd(*model.fwd_inputs)[0]
  fwd_model = theano.function(inputs=model.fwd_inputs, outputs=model_out,
          on_unused_input='warn')
  logging.info("...done %f" % (time.time() - start_time))

  logging.info("Getting validation data...")
  start_time = time.time()
  valid_data = []
  iterator = fetch_data(queue_size, inport)
  for i in xrange(n_batches_valid):
    valid_data.append(iterator.next())
  logging.info("...done %f" % (time.time() - start_time))

  print fwd_model(valid_data[0][0], valid_data[0][1], valid_data[0][2],
      valid_data[0][3]).shape
  print valid_data[0][0].shape

  logging.info("Measuring performance on the validation set...")
  start_time = time.time()
  perf_val_batch = perf_mon.eval_measures(valid_data, fwd_model,
          len(model.fwd_inputs), '...[VALID] ', logging)[0]
  logging.info("...done %f" % (time.time() - start_time))
  
  logging.info("Starting training...")
  it_count = 1
  while it_count < n_epochs:
    start_time = time.time()
    bcost = []
    bdata = []
    count = 0
    for i in xrange(conf['iter_per_epoch']):
      data = iterator.next()
      bcost.append(train_model(*data)) 
      #bcost.append(train_model(data[0], data[1], data[2], data[3], data[4], data[5], data[6], 
      #    data[7], data[8], data[9], data[10], data[11], data[12], data[13],
      #    data[14], data[15]))

      if count < n_train_to_eval:
        # to measure performance on the training set
        bdata.append(data)
        count += 1

    logging.info(('...cost %f n_updates %i (%.2fsec)') % (np.mean(bcost), n_updates.get_value(), time.time() - start_time))
    model.save(os.path.join(out_path, "%s-last-model" % exp_name))

    if (it_count % iter_eval) == 0:
      perf_mon.eval_measures(bdata, fwd_model, len(model.fwd_inputs), '...[TRAIN_BATCH] ', logging)[0]
            
      logging.info("Saving training snapshot for current model")
      f = file(os.path.join(out_path, "%s-last-train-checkpoint.pkl" % exp_name), 'wb')
      cPickle.dump([n_updates, gradients_sq, deltas_sq, np.random.get_state()], f, protocol=cPickle.HIGHEST_PROTOCOL)
      f.close()


      logging.info("Measuring performance on the validation set...")
      start_time = time.clock()
      perf_val_batch = perf_mon.eval_measures(valid_data, fwd_model, len(model.fwd_inputs), '...[VALID] ', logging)[0]

      if perf_val_batch > best_perf:
        best_perf = perf_val_batch
        model.save(os.path.join(out_path, "%s-best-model" % exp_name))
              
        logging.info("Saving training snapshot for best validation ")
        f = file(os.path.join(out_path, "%s-train-checkpoint.pkl" % exp_name), 'wb')
        cPickle.dump([n_updates, gradients_sq, deltas_sq, np.random.get_state()], f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

      model.monitor_params(logfun=lambda x : logging.info(x))

    it_count += 1

