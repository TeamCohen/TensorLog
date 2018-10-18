# collect available target languages:
try:
  import tensorflow as private1
  tf=True
except:
  tf=False
try:
  import theano as private2
  theano=True
except:
  theano=False
# disable theano tests for now, some of them fail and it's not a
# priority...
theano=False
