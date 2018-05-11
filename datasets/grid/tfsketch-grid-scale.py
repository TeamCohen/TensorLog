import sys
import numpy as NP
import scipy.sparse as ss
import random
import math
import time
import getopt

import tensorflow as tf
from tensorlog import simple,declare
from tensorlog.helper.tfsketch import Sketcher,Sketcher2,SketchSMDMCrossCompiler,SketchData
SKETCHERS={
    "FastSketcher2":Sketcher2,
    "FastSketcher":Sketcher,
    "Sketcher":Sketcher,
    "Sketcher2":Sketcher2
    }

random.seed(31415926)

import grid_scale

#funs.conf.trace=True
#funs.conf.long_trace=True

def retrieveSketcherArgs(argv):
    optlist,args = getopt.getopt(argv,'x',"k= delta= sketcher=".split())
    optdict = dict(optlist)
    #print 'optdict',optdict
    k=optdict.get('--k','160')
    delta = optdict.get('--delta','0.01')
    sketcher_s = optdict.get('--sketcher','FastSketcher2')
    if sketcher_s not in SKETCHERS:
        print "no sketcher named %s" % sketcher_s
        print "available sketchers:"
        print "\n".join(SKETCHERS.keys())
        exit(0)
    sketcher_c = SKETCHERS[sketcher_s]
    return {
        'k':k,
        'delta':delta,
        'sketcher_s':sketcher_s,
        'sketcher_c':sketcher_c
        }
#@profile
def setup_tlog(maxD,factFile,trainFile,testFile):
    tlog = simple.Compiler(target='tensorflow-sketch',db=factFile,prog='grid.ppr')
    tlog.prog.db.markAsParameter('edge',2)
    tlog.prog.maxDepth=maxD

    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--")+1:]
    skopts = retrieveSketcherArgs(argv)
    args = (skopts['k'],skopts['delta'],skopts['sketcher_s'])
    print " ".join(args)


    db = tlog.prog.db
    sketcher = skopts['sketcher_c'](db,int(skopts['k']),float(skopts['delta'])/db.dim(),verbose=False)
    sketcher.describe()
    
    tlog.xc.setSketcher(sketcher)
    
    trainData_native = tlog.load_big_dataset(trainFile)
    testData_native = tlog.load_big_dataset(testFile)
    trainData = SketchData(sketcher,trainData_native)
    testData = SketchData(sketcher,testData_native)
    return (tlog,trainData,testData)

#@profile
def trainAndTest(tlog,trainData,testData,epochs):
    mode = declare.asMode('path/io')
    predicted_y = tlog.xc.sk.unsketch(tlog.inference(mode))
    predicted_y_argmax = tf.argmax(predicted_y,1)
    actual_y = tlog.xc.sk.unsketch(tf.expand_dims(tlog.target_output_placeholder(mode),1))
    correct_predictions = tf.equal(tf.argmax(actual_y,1), predicted_y_argmax)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    unregularized_loss = tlog.loss(mode)
    optimizer = tf.train.AdagradOptimizer(1.0)
    train_step = optimizer.minimize(unregularized_loss)
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    _,(tx,ty) = tlog.minibatches(trainData,trainData.size()).next()#trainData.getX(mode), trainData.getY(mode)
    train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
    _,(ux,uy) = tlog.minibatches(testData,testData.size()).next()#testData.getX(mode),testData.getY(mode)
    test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
    
    acc= session.run([accuracy], feed_dict=test_fd)
    print 'untrained test acc',acc
    
    t0 = time.time()
    for i in range(epochs):
        print 'epoch',i+1
        session.run(train_step, feed_dict=train_fd)
    print 'learning takes',time.time()-t0,'sec'

    acc = session.run([accuracy], feed_dict=test_fd)
    print 'trained test acc',acc
    return acc

def ppSolutions(mat,db):
    d = db.matrixAsSymbolDict(ss.csr_matrix(mat))
    kk = sorted(d.keys())
    for k in kk:
        print k,"=>"
        jj = sorted(d[k].keys())
        for j in jj:
            print ".",j,':',d[k][j]

def runMain():
    if len(sys.argv)<3:
      print "usage:"
      print "  acc [grid-size] [max-depth] [epochs]"
      print "build [grid-size] [max-depth]"
      exit(0)
    (goal,n,maxD,epochs) = grid_scale.getargs()
    print 'grid-acc-expt: %d x %d grid, %d epochs, maxPath %d' % (maxD,n*maxD,epochs,maxD),
    (factFile,_,_) = grid_scale.genInputs(n,maxD,build=False)
    if goal=="acc": 
        (_,trainFile,testFile) = grid_scale.genInputs(1,maxD,build=False)
        (tlog,trainData,testData) = setup_tlog(maxD,factFile,trainFile,testFile)
        trainAndTest(tlog,trainData,testData,epochs)
                                         
if __name__=="__main__":
    runMain()
