import sys
import numpy as NP
import scipy.sparse as ss
import random
import math
import time
import getopt

import tensorflow as tf
from tensorlog import simple,declare

random.seed(31415926)

import grid_scale


def setup_tlog(maxD,factFile,trainFile,testFile):
    tlog = simple.Compiler(target='tensorflow',db=factFile,prog='grid.ppr')
    tlog.prog.db.markAsParameter('edge',2)
    tlog.prog.maxDepth=maxD

    #argv = []
    # if "--" in sys.argv:
    #     argv = sys.argv[sys.argv.index("--")+1:]
    # skopts = retrieveSketcherArgs(argv)

    # db = tlog.prog.db
    # sketcher = skopts['sketcher_c'](db,int(skopts['k']),float(skopts['delta'])/db.dim(),verbose=False)
    # sketcher.describe()
    # args = (skopts['k'],skopts['delta'],skopts['sketcher_s'])

    # tlog.xc.setSketcher(sketcher)
    
    trainData = tlog.load_big_dataset(trainFile)
    testData = tlog.load_big_dataset(testFile)
    #trainData = SketchData(sketcher,trainData_native)
    #testData = SketchData(sketcher,testData_native)
    return (tlog,trainData,testData)

def trainAndTest(tlog,trainData,testData,epochs):
    mode = declare.asMode('path/io')
    predicted_y = tlog.inference(mode)
    predicted_y_argmax = tf.argmax(predicted_y,1)
    actual_y = tlog.target_output_placeholder(mode)
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
    print 'grid-acc-expt: %d x %d grid, %d epochs, maxPath %d -1 -1 native' % (maxD,n*maxD,epochs,maxD)
    (factFile,trainFile,testFile) = grid_scale.genInputs(n,maxD)
    (tlog,trainData,testData) = setup_tlog(maxD,factFile,trainFile,testFile)
    trainAndTest(tlog,trainData,testData,epochs)
                                         
if __name__=="__main__":
    runMain()
