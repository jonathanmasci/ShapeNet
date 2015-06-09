# 
# This code is part of DeepShape <...>
#
# Copyright 2015
# Jonathan Masci
# <jonathan.masci@gmail.com>

import os
import sys
import zmq
from proto import training_sample_lscnn_pb2
import numpy as np
import time
import argparse
import scipy.sparse as sp
#from multiprocessing import Pool 
from multiprocessing.dummy import Pool 
from layers.utils import filtered_list_dir
import h5py

def load_mat_field(path, field):
  """
  path : string : absolute path to the mat file to load
  field : string : field to return from the dictionary
  return : np.array (float32) 
  """
  data = h5py.File(path,'r')
  return np.asarray(data[field]).astype(np.float32).T

def load_shape_field(path, field):
  """
  By design shapes are contained in a struct (from Matlab)
  called shape.
  This function loads the shape .mat file and returns the
  requested field.

  path : string : absolute path to the mat file for the shape
  field : string : field in the dict shape to return
  return : np.array (float32)
  """
  data = h5py.File(path, 'r')
  return np.asarray(data['shape'][field]).astype(np.float32).T

def producer(args):
  """
  This function takes care of data loading, sampling of the pairs (positive
  and negatives) and to put data, serialized with protobuf in the zmq queue.

  args.queue_size : int : number of data samples to store in the queue.
      Remember to set this value accordingly in the streamer as well
  args.outport : string : port number where to find the streamer queue
  args.desc_dir : string : absolute path to the input descriptors (e.g.
      GEOVEC, WKS, etc). A matrix of size N x K is needed where N is the
      number of vertices in the shape and K is the number of dimensions of the
      descriptor.
  args.shape_dir : string : absolute path to the input shapes. The structure
      needs to contain the following fields:
          - Phi : matrix of laplace-beltrami eigenfunctions, e.g. matrix of
            size N x D, where D is the number of eigenfunctions to use. 
            100 is what we use in the paper.
          - A : area vector
          - Lambda : eigenvalues
  args.batch_size : int : how many vertices to take. -1 take the entire shape
      and is recommended. LSCNN needs all points to be computed correctly,
      subsampling works with stochastic optimization but beware it is an
      approximation of the original LSCNN net model.
  """
  context = zmq.Context()
  zmq_socket = context.socket(zmq.PUSH)
  zmq_socket.sndhwm = args.queue_size
  zmq_socket.rcvhwm = args.queue_size
  zmq_socket.connect("tcp://127.0.0.1:" + args.outport)

  # List all shapes, filenames must match in each subfolder 
  fnames = filtered_list_dir(args.desc_dir, ['mat'])
  ndata = len(fnames)
  print "Working with %i shapes in %s" % (ndata, args.desc_dir)
  if ndata == 0:
      return

  if args.alltomem == True:
    print "Loading all data into memory, hope it will fit!"
    descs = []
    Vs = []
    As = []
    Lambdas = []
    for f in fnames:
      Vs.append(load_shape_field(os.path.join(args.shape_dir, f), 'Phi'))
      As.append(load_shape_field(os.path.join(args.shape_dir, f), 'A'))
      Lambdas.append(load_shape_field(os.path.join(args.shape_dir, f), 'Lambda'))
      if args.const_fun:
        descs.append(np.ones((As[0].shape[0], 1)).astype(np.float32))
      else:
        descs.append(load_mat_field(os.path.join(args.desc_dir, f), 'desc'))
    get_data = lambda x : [descs[x], Vs[x], As[x], Lambdas[x]]
    print "done"
  else:
    assert(args.const_fun == False) # TODO implement this case
    get_data = lambda x : [
            load_mat_field(os.path.join(args.desc_dir, f), 'desc'),
            load_shape_field(os.path.join(args.shape_dir, f), 'Phi'),
            load_shape_field(os.path.join(args.shape_dir, f), 'A'),
            load_shape_field(os.path.join(args.shape_dir, f), 'Lambda')]

  count = 0
  ndots = 0
  print "Starting sampling"
  while True:
    prm = np.random.permutation(ndata)[:2]

    f_i, V_i, A_i, L_i = get_data(prm[0])
    f_j, V_j, A_j, L_j = get_data(prm[1])
    if A_i.shape[0] != 1 and A_i.shape[1] != 1:
      A_i = np.diag(A_i)
      A_j = np.diag(A_j)

    posprm = np.random.permutation(f_i.shape[0])
    negprm = np.random.permutation(f_i.shape[0])
    
    if args.batch_size != -1:
      posprm = posprm[:args.batch_size]
      negprm = negprm[:args.batch_size]

    f_i = f_i[posprm]
    V_i = V_i[posprm]
    A_i = A_i[posprm]

    f_j_n = f_j[negprm]
    V_j_n = V_j[negprm]
    A_j_n = A_j[negprm]

    f_j = f_j[posprm]
    V_j = V_j[posprm]
    A_j = A_j[posprm]

    tr = training_sample_lscnn_pb2.training_sample_lscnn()

    tr.xp            = f_i.tostring()
    tr.xp_shape.extend(f_i.shape)
    tr.Vxp           = V_i.tostring()
    tr.Vxp_shape.extend(V_i.shape)
    tr.Axp           = A_i.tostring()
    tr.Axp_shape.extend(A_i.shape)
    tr.Lxp           = L_i.tostring()
    tr.Lxp_shape.extend(L_i.shape)

    tr.xn            = f_i.tostring()
    tr.xn_shape.extend(f_i.shape)
    tr.Vxn           = V_i.tostring()
    tr.Vxn_shape.extend(V_i.shape)
    tr.Axn           = A_i.tostring()
    tr.Axn_shape.extend(A_i.shape)
    tr.Lxn           = L_i.tostring()
    tr.Lxn_shape.extend(L_i.shape)
    
    tr.yp            = f_j.tostring()
    tr.yp_shape.extend(f_j.shape)
    tr.Vyp           = V_j.tostring()
    tr.Vyp_shape.extend(V_j.shape)
    tr.Ayp           = A_j.tostring()
    tr.Ayp_shape.extend(A_j.shape)
    tr.Lyp           = L_j.tostring()
    tr.Lyp_shape.extend(L_j.shape)

    tr.yn            = f_j_n.tostring()
    tr.yn_shape.extend(f_j.shape)
    tr.Vyn           = V_j_n.tostring()
    tr.Vyn_shape.extend(V_j.shape)
    tr.Ayn           = A_j_n.tostring()
    tr.Ayn_shape.extend(A_j.shape)
    tr.Lyn           = L_j.tostring()
    tr.Lyn_shape.extend(L_j.shape)
    

    zmq_socket.send(tr.SerializeToString())

    count = count + 1
    if count > 100:
      print ".",
      count = 0
      ndots = ndots + 1
      if ndots > 25:
        print ""
        ndots = 0
 
if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description='Generate data for training \
          LSCNN networks.')
  parser.add_argument('--desc_dir', metavar='desc_dir', type=str,
          dest='desc_dir', required=True,
          help='Where to find the descriptors')
  parser.add_argument('--shape_dir', metavar='shape_dir', type=str,
          dest='shape_dir', required=True,
          help='Where to find the shapes with Phi, A and Lambda fields')
  parser.add_argument('--const_fun', metavar='const_fun', type=int,
          dest='const_fun',
          required=True)
  parser.add_argument('--nthreads', metavar='nthreads', type=int,
          dest='nthreads',
          required=True)
  parser.add_argument('--alltomem', metavar='alltomem', type=int,
          dest='alltomem',
          required=True)
  parser.add_argument('--batch_size', metavar='batch_size', type=int,
          dest='batch_size',
          help='Batch size',
          required=True)
  parser.add_argument('--queue_size', metavar='queue_size', type=int,
          dest='queue_size',
          help='Maximum size of the queue',
          required=True)
  parser.add_argument('--outport', metavar='outport', type=str,
          dest='outport',
          help='Port to send data',
          required=False, default='5579')
  
  args = parser.parse_args()
  print "Generating data with %i threads" % args.nthreads
  pool = Pool(args.nthreads)
  args = [args] * args.nthreads
  pool.map(producer, args)
  pool.join() 
  pool.close() 
 
