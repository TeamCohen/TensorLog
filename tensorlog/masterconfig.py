import logging

from tensorlog import config

def masterConfig():
  from tensorlog import bpcompiler
  from tensorlog import dataset
  from tensorlog import funs
  from tensorlog import learn
  from tensorlog import matrixdb
  from tensorlog import mutil
  from tensorlog import ops
  from tensorlog import program
  from tensorlog import xcomp

  master =  config.Config()
  master.bpcompiler = bpcompiler.conf
  master.help.bpcompiler = 'config for tensorlog.bpcompiler'
  master.dataset = dataset.conf
  master.help.dataset = 'config for tensorlog.dataset'
  master.funs = funs.conf
  master.help.funs = 'config for tensorlog.funs'
  master.learn = learn.conf
  master.help.learn = 'config for tensorlog.learn'
  master.matrixdb = matrixdb.conf
  master.help.matrixdb = 'config for tensorlog.matrixdb'
  master.mutil = mutil.conf
  master.help.mutil = 'config for tensorlog.mutil'
  master.ops = ops.conf
  master.help.ops = 'config for tensorlog.ops'
  master.program = program.conf
  master.help.program = 'conf for tensorlog.program'
  master.xcomp = xcomp.conf
  master.help.xcomp = 'config for tensorlog.xcomp'
  try:
    from tensorlog import debug
    master.debug = debug.conf
    master.help.debug = 'config for tensorlog.debug'
  except ImportError:
    logging.warn('debug module not imported')
  return master

if __name__ == "__main__":
  masterConfig().pprint()
