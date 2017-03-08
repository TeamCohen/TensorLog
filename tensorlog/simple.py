from tensorlog import comline
from tensorlog import config
from tensorlog import declare
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import program
from tensorlog import tensorflowxcomp
from tensorlog import theanoxcomp

def compiler(target='tensorflow',db=None,prog=None,rule_features=False,summary_file=False):

  if isinstanceof(db,matrixdb.MatrixDB):
    pass
  elif isinstanceof(db,str):
    db = comline.parseDBSpec(db)
  else:
    assert False,'cannot convert %r to a database' % db

  if isinstanceof(prog,program.Program):
    pass
  elif isinstanceof(prog,str):
    prog = comline.parseProgSpec(prog,db,proppr=rule_features)
  else:
    assert False,'cannot convert %r to a program' % prog

  if target=='tensorflow':
    result = tensorflowxcomp.SparseMatDenseMsgCrossCompiler(prog, summaryFile=summary_file)
  elif target=='theano':
    result = theanoxcomp.SparseMatDenseMsgCrossCompiler(prog)
  else:
    assert False,'illegal target %r: valid targets are "tensorflow" and "theano"' % target
