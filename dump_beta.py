import scipy.io
import cPickle
import sys

f = open(sys.argv[1],'r')
data = cPickle.load(f)
f.close()

beta = dict()
beta['beta'] = data[int(sys.argv[2])]
print beta['beta'].shape
scipy.io.savemat(sys.argv[3],beta)
