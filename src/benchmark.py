# (C) William W. Cohen and Carnegie Mellon University, 2016

import parser
import sys
import time

import declare
import matrixdb
import mutil
import opfunutil
import tensorlog

#
# some timing benchmarks
#


def fbModes():
  modes = []
  for line in open("test/fb15k-valid.preds"):
    pred = line.strip()
    modes.append(declare.ModeDeclaration("%s(i,o)" % pred))
  return modes


def fbProgram(rules, db, modes):
  prog = tensorlog.Program(db=db, rules=rules)
  t = s = 0
  for _k, m in enumerate(modes):
    t += 1
    try:
      prog.compile(m)
      s += 1
    except AssertionError:
      pass
  print "compiled", s, "of", t, "modes successfully"
  return prog


def fbQueries(prog, db, modes):
  modeDict = {}
  for m in modes:
    modeDict[m.functor] = m
  queries = []
  ignored = 0
  for line in open("test/fb15k-valid.examples"):
    k1 = line.find("(")
    k2 = line.find(",")
    pred = line[:k1]
    x = line[k1+1:k2]
    if pred in modeDict and (modeDict[pred], 0) in prog.function:
      m = modeDict[line[:k1]]
      vx = db.onehot(x)
      queries.append((m, vx))
    else:
      ignored += 1
  print len(queries), "queries loaded", "ignored", ignored
  return queries


def runBenchmark(com):
  print "benchmark:",com
  if com=="fb-db-serialize":
    db = matrixdb.MatrixDB.loadFile("test/fb15k-valid.cfacts")
    start = time.time()
    db.serialize("fb15k-valid.db")
  elif com=="fb-db-load":
    start = time.time()
    db = matrixdb.MatrixDB.deserialize("fb15k-valid.db")
  elif com=="fb-rule-parse":
    start = time.time()
    rules = parser.Parser.parseFile("test/fb15k.ppr")
  elif com=="fb-rule-compile":
    db = matrixdb.MatrixDB.deserialize("fb15k-valid.db")
    rules = parser.Parser.parseFile("test/fb15k.ppr")
    print rules.size(),"rules compiled and db loaded"
    modes = fbModes()
    start = time.time()
    prog = fbProgram(rules,db,modes)
  elif com=="fb-rule-answer-native":
    db = matrixdb.MatrixDB.deserialize("fb15k-valid.db")
    rules = parser.Parser.parseFile("test/fb15k.ppr")
    modes = fbModes()
    prog = fbProgram(rules,db,modes)
    modeDict = {}
    print "program loaded and compiled"
    queries = fbQueries(prog,db,modes)
    start = time.time()
    k = 0
    for (m,vx) in queries:
      fun = prog.function[(m, 0)]
      fun.eval(db, [vx], opfunutil.Scratchpad())
      k += 1
      if not k%100: print "answered",k,"queries"
    print "answered",len(queries),"queries at",len(queries)/(time.time() - start),"qps"
  elif com=="fb-rule-answer-matrix":
    db = matrixdb.MatrixDB.deserialize("fb15k-valid.db")
    rules = parser.Parser.parseFile("test/fb15k.ppr")
    modes = fbModes()
    prog = fbProgram(rules,db,modes)
    modeDict = {}
    print "program loaded and compiled"
    queries = fbQueries(prog,db,modes)
    #a list of (m,vx)
    qMode = queries[0][0]
    Xs = [vx for m,vx in queries]
    qX = mutil.stack(Xs)
    start = time.time()
    fun = prog.function[(qMode, 0)]
    fun.eval(db, [qX], opfunutil.Scratchpad())
    print "answered", len(queries), "queries at",
    print len(queries)/(time.time() - start), "qps"
  else:
    assert False, "illegal benchmark task"
  elapsed = time.time() - start
  print "time %.3f" % elapsed
  #raw_input("press enter:")


if __name__ == "__main__":
  if len(sys.argv) > 1:
    for com in sys.argv[1:]:
      runBenchmark(com)
  else:
    for com in ["fb-db-load", "fb-rule-compile", 
                "fb-rule-answer-native", "fb-rule-answer-matrix"]:
      runBenchmark(com)
