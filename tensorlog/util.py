import os
import inspect

# misc utilities

def memusage():
    """ Memory used by the current process in Gb
    """
    proc_status = '/proc/%d/status' % os.getpid()
    try:
        t = open(proc_status)
        v = t.read()
        t.close()
        i = v.index('VmSize:')
        v = v[i:].split(None,3)
        scale = {'kB': 1024.0, 'mB': 1024.0*1024.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}
        return (float(v[1]) * scale[v[2]]) / (1024.0*1024.0*1024.0)
    except IOError:
        return 0.0

def linesIn(fileLike):
  """ If fileLike is a string, open it as a file and return lines in the file.
  Otherwise, just call fileLike's iterator method and iterate over that.
  Thus, you can use open file handles or strings as arguments to a function f if
  it accesses its arguments thru linesIn:

  def f(fileLikeInput,....):
    ...
    for line in linesIn(fileLikeInput):
      ...

  """
  if isinstance(fileLike,str):
    with open(fileLike) as fp:
      for line in fp:
        yield line
  else:
    d = dict(inspect.getmembers(fileLike))
    assert '__iter__' in d, 'cannot enumerate lines in the object  %r' % fileLike
    for line in fileLike:
      yield line
