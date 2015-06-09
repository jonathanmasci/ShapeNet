import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

class PerformanceMonitor(object):
  def __init__(self):
    pass

  def eval_measures(self, x, fwd, ninputs, prefix, logging, rng=42):
    xp, yp, xn, yn = [], [], [], []
    xp_raw, yp_raw, xn_raw, yn_raw = [], [], [], []
    print "Evaluation on %i mini-batches" % len(x)
    for i in xrange(len(x)):
      xp.append(fwd(*x[i][:ninputs]))
      yp.append(fwd(*x[i][ninputs:2*ninputs]))
      xn.append(fwd(*x[i][2*ninputs:3*ninputs]))
      yn.append(fwd(*x[i][3*ninputs:]))

      xp_raw.append(x[i][0])
      yp_raw.append(x[i][ninputs])
      xn_raw.append(x[i][2*ninputs])
      yn_raw.append(x[i][3*ninputs])

    xp = np.concatenate(xp, axis=0) 
    yp = np.concatenate(yp, axis=0) 
    xn = np.concatenate(xn, axis=0) 
    yn = np.concatenate(yn, axis=0) 
    
    xp_raw = np.concatenate(xp_raw, axis=0) 
    yp_raw = np.concatenate(yp_raw, axis=0) 
    xn_raw = np.concatenate(xn_raw, axis=0) 
    yn_raw = np.concatenate(yn_raw, axis=0) 
    
    np.random.seed(rng)

    dp_raw = np.sqrt(np.sum((xp_raw - yp_raw)**2., axis=1))
    dn_raw = np.sqrt(np.sum((xn_raw - yn_raw)**2., axis=1))
    D_raw = np.concatenate([dp_raw, dn_raw], axis=0)
    
    dp = np.sqrt(np.sum((xp - yp)**2., axis=1))
    dn = np.sqrt(np.sum((xn - yn)**2., axis=1))
    D = np.concatenate([dp, dn], axis=0)
    
    dp_bin = np.sqrt(np.sum((np.float32(xp > 0.) - np.float32(yp > 0.))**2., axis=1))
    dn_bin = np.sqrt(np.sum((np.float32(xn > 0.) - np.float32(yn > 0.))**2., axis=1))
    D_bin = np.concatenate([dp_bin, dn_bin], axis=0)

    y_true = np.zeros((D.shape[0], )) - 1.
    y_true[:dp.shape[0]] = 1.
    
    AUC = roc_auc_score(y_true, -D)
    AUC_raw = roc_auc_score(y_true, -D_raw)
    AUC_bin = roc_auc_score(y_true, -D_bin)
    precision, recall, thresholds = precision_recall_curve(y_true, -D)
    precision_raw, recall_raw, thresholds_raw = precision_recall_curve(y_true, -D_raw)
    precision_bin, recall_bin, thresholds_bin = precision_recall_curve(y_true, -D_bin)
   
    logging.info("...NN AUC (bin) %f (%f), RAW AUC %f", AUC, AUC_bin, AUC_raw)

    return AUC, precision, recall

class PerformanceMonitorShapeNet(object):
  def __init__(self):
    pass

  def eval_measures(self, x, fwd, prefix, logging, rng=42, nmax=1000):
    xp, yp, xn, yn = [], [], [], []
    xp_raw, yp_raw, xn_raw, yn_raw = [], [], [], []
    print "Evaluation on %i mini-batches" % len(x)
    for i in xrange(len(x)):
      prm = np.random.permutation(x[i][0].shape[0])[:nmax]
      xp.append(fwd(x[i][0], x[i][1])[prm])
      yp.append(fwd(x[i][2], x[i][3])[prm])
      xn.append(fwd(x[i][4], x[i][5])[prm])
      yn.append(fwd(x[i][6], x[i][7])[prm])

      xp_raw.append(x[i][0][prm])
      yp_raw.append(x[i][2][prm])
      xn_raw.append(x[i][4][prm])
      yn_raw.append(x[i][6][prm])

    xp = np.concatenate(xp, axis=0) 
    yp = np.concatenate(yp, axis=0) 
    xn = np.concatenate(xn, axis=0) 
    yn = np.concatenate(yn, axis=0) 
    print xp.shape 
    xp_raw = np.concatenate(xp_raw, axis=0) 
    yp_raw = np.concatenate(yp_raw, axis=0) 
    xn_raw = np.concatenate(xn_raw, axis=0) 
    yn_raw = np.concatenate(yn_raw, axis=0) 

    np.random.seed(rng)

    dp_raw = np.sqrt(np.sum((xp_raw - yp_raw)**2., axis=1))
    dn_raw = np.sqrt(np.sum((xn_raw - yn_raw)**2., axis=1))
    D_raw = np.concatenate([dp_raw, dn_raw], axis=0)
    
    dp = np.sqrt(np.sum((xp - yp)**2., axis=1))
    dn = np.sqrt(np.sum((xn - yn)**2., axis=1))
    D = np.concatenate([dp, dn], axis=0)
    
    dp_bin = np.sqrt(np.sum((np.float32(xp > 0.) - np.float32(yp > 0.))**2., axis=1))
    dn_bin = np.sqrt(np.sum((np.float32(xn > 0.) - np.float32(yn > 0.))**2., axis=1))
    D_bin = np.concatenate([dp_bin, dn_bin], axis=0)

    y_true = np.zeros((D.shape[0], )) - 1.
    y_true[:dp.shape[0]] = 1.
    
    AUC = roc_auc_score(y_true, -D)
    AUC_raw = roc_auc_score(y_true, -D_raw)
    AUC_bin = roc_auc_score(y_true, -D_bin)
    precision, recall, thresholds = precision_recall_curve(y_true, -D)
    #precision_raw, recall_raw, thresholds_raw = precision_recall_curve(y_true, -D_raw)
    #precision_bin, recall_bin, thresholds_bin = precision_recall_curve(y_true, -D_bin)
   
    logging.info("...NN AUC (bin) %f (%f), RAW AUC %f", AUC, AUC_bin, AUC_raw)

    return AUC, precision, recall

class PerformanceMonitorLSCNN(object):
  def __init__(self):
    pass

  def eval_measures(self, x, fwd, prefix, logging, rng=42, nmax=1000):
    xp, yp, xn, yn = [], [], [], []
    xp_raw, yp_raw, xn_raw, yn_raw = [], [], [], []
    print "Evaluation on %i mini-batches" % len(x)
    for i in xrange(len(x)):
      prm = np.random.permutation(x[i][0].shape[0])[:nmax]
      xp.append(fwd(x[i][0],  x[i][1],  x[i][2],  x[i][3] )[prm])
      yp.append(fwd(x[i][4],  x[i][5],  x[i][6],  x[i][7] )[prm])
      xn.append(fwd(x[i][8],  x[i][9],  x[i][10], x[i][11])[prm])
      yn.append(fwd(x[i][12], x[i][13], x[i][14], x[i][15])[prm])

      xp_raw.append(x[i][0][prm])
      yp_raw.append(x[i][4][prm])
      xn_raw.append(x[i][8][prm])
      yn_raw.append(x[i][12][prm])

    xp = np.concatenate(xp, axis=0) 
    yp = np.concatenate(yp, axis=0) 
    xn = np.concatenate(xn, axis=0) 
    yn = np.concatenate(yn, axis=0) 
    print xp.shape 
    xp_raw = np.concatenate(xp_raw, axis=0) 
    yp_raw = np.concatenate(yp_raw, axis=0) 
    xn_raw = np.concatenate(xn_raw, axis=0) 
    yn_raw = np.concatenate(yn_raw, axis=0) 

    np.random.seed(rng)

    dp_raw = np.sqrt(np.sum((xp_raw - yp_raw)**2., axis=1))
    dn_raw = np.sqrt(np.sum((xn_raw - yn_raw)**2., axis=1))
    D_raw = np.concatenate([dp_raw, dn_raw], axis=0)
    
    dp = np.sqrt(np.sum((xp - yp)**2., axis=1))
    dn = np.sqrt(np.sum((xn - yn)**2., axis=1))
    D = np.concatenate([dp, dn], axis=0)
    
    dp_bin = np.sqrt(np.sum((np.float32(xp > 0.) - np.float32(yp > 0.))**2., axis=1))
    dn_bin = np.sqrt(np.sum((np.float32(xn > 0.) - np.float32(yn > 0.))**2., axis=1))
    D_bin = np.concatenate([dp_bin, dn_bin], axis=0)

    y_true = np.zeros((D.shape[0], )) - 1.
    y_true[:dp.shape[0]] = 1.
    
    AUC = roc_auc_score(y_true, -D)
    AUC_raw = roc_auc_score(y_true, -D_raw)
    AUC_bin = roc_auc_score(y_true, -D_bin)
    precision, recall, thresholds = precision_recall_curve(y_true, -D)
    #precision_raw, recall_raw, thresholds_raw = precision_recall_curve(y_true, -D_raw)
    #precision_bin, recall_bin, thresholds_bin = precision_recall_curve(y_true, -D_bin)
   
    logging.info("...NN AUC (bin) %f (%f), RAW AUC %f", AUC, AUC_bin, AUC_raw)

    return AUC, precision, recall

