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