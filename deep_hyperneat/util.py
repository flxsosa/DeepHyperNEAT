'''
Common functions used throughout DeepHyperNEAT.

Contains iterator tools and basic statistics function. The iterator tools are
copied directly from six (Copyright (c) 2010-2018 Benjamin Peterson).
'''
import sys
import numpy as np

if sys.version_info[0] == 3:
    def iterkeys(d, **kw):
        return iter(d.keys(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))

    def itervalues(d, **kw):
        return iter(d.values(**kw))
else:
    def iterkeys(d, **kw):
        return iter(d.iterkeys(**kw))

    def iteritems(d, **kw):
        return iter(d.iteritems(**kw))

    def itervalues(d, **kw):
        return iter(d.itervalues(**kw))

def mean(x):
    return np.mean(x)

def median(x):
    return np.median(x)

def variance(x):
    return np.var(x)

def stdev(x):
    return np.std(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()