import sys
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
import getopt

# prior probability of an edge
P_EDGE = 0.25

# demo of learning an approximation of the independent-tuples distribution.
# 
# TODO: try different loss functions for distributions, this seems to
# produce models that are sort of similar in ranking but with probabilities
# at a different scale.  euclidean distance? jenson-shannon?
# TODO: output test-set and full-data y's for visualization
# TODO: why do the actual y's include non-zeros for the filler entities?
# TODO: should I normalize this? why?
#
#
# --gendata M - draw M sample interpretations (ie grids, where edges
#   are present with probability P_EDGE), compute all pairs (x,y)
#   where path(x,y) is true, and use this to approximate Pr(path(x,y))
#   in the independent-tuples model.  Store the result on disk.
# 
# --load F - load Pr(path(x,y)) from disk and train against this
# --epochs N: default 1000  -- takes a long time to learn this
# --repeat K: default 1
#
# --n N: grid size is N*N default 10

from tensorlog import simple,program,declare,dbschema,masterconfig,matrixdb
import expt


def setup_tlog(maxD,factFile,trainFile,testFile):
  tlog = simple.Compiler(db=factFile,prog='grid.ppr')
  tlog.prog.db.markAsParameter('edge',2)
  tlog.prog.maxDepth = maxD
  print 'loading trainData,testData from',trainFile,testFile
  masterconfig.masterConfig().dataset.normalize_outputs = False
  trainData = tlog.load_small_dataset(trainFile)
  testData = tlog.load_small_dataset(testFile)
  return (tlog,trainData,testData)

def trainAndTest(tlog,trainData,testData,epochs,n):
  mode = 'path/io'
  tlog.A = tf.Variable(1.0, "A")
  tlog.B = tf.Variable(0.0, "B")
  logits = tlog.A*tlog.proof_count(mode) + tlog.B
  predicted_y = tf.sigmoid(logits)
  actual_y = tlog.target_output_placeholder(mode)

  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=actual_y,logits=logits))
  optimizer = tf.train.AdamOptimizer(0.1)
  train_step = optimizer.minimize(loss)

  session = tf.Session()
  session.run(tf.global_variables_initializer())

  (ux,uy) = testData[mode]
  test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
  (tx,ty) = trainData[mode]
  train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
  init_haty = session.run(predicted_y,feed_dict=test_fd)
  train_loss = session.run(loss, feed_dict=train_fd)
  test_loss = session.run(loss, feed_dict=test_fd)
  print 'init train loss',train_loss,'test loss',test_loss

  t0 = time.time()
  print 'epoch',
  for i in range(epochs):
    print i+1,
    session.run(train_step, feed_dict=train_fd)
    if (i+1)%3==0:
      test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
      train_loss = session.run(loss, feed_dict=train_fd)
      test_loss = session.run(loss, feed_dict=test_fd)
      print (i+1),'train loss',train_loss,'test loss',test_loss
      print 'epoch',
  print 'done'
  print 'learning takes',time.time()-t0,'sec'

  test_loss = session.run(loss, feed_dict=test_fd)
  print 'test loss',test_loss

  print 'A,B',session.run([tlog.A,tlog.B],feed_dict=test_fd)
  y,haty = session.run([predicted_y,actual_y],feed_dict=test_fd)
  sio.savemat('testpreds.mat',{'y':y,'haty':haty,'init_haty':init_haty})

  return test_loss

def nodeName(i,j):
    return '%d,%d' % (i,j)

def generateGrid(n,outf):
    fp = open(outf,'w')
    #cell types
    for i in range(1,n+1):
      for j in range(1,n+1):
        fp.write('cell\t%s\n' % nodeName(i,j))
    for i in range(1,n+1):
        for j in range(1,n+1):
            for di in [-1,0,+1]:
                for dj in [-1,0,+1]:
                    if (1 <= i+di <= n) and (1 <= j+dj <= n):
                        fp.write('edge\t%s\t%s\t%f\n' % (nodeName(i,j),nodeName(i+di,j+dj),EDGE_WEIGHT))

def genInputs(n,sampleSize):
    #generate grid
    stem = 'inputs/g%d' % n
    baseFactFile = stem+'.cfacts'
    generateGrid(n,baseFactFile)
    db = matrixdb.MatrixDB.loadFile(baseFactFile)
    e = db.matEncoding[('edge',2)].todense()
    d = db.dim()
    e[e>0] = 1.0
    s = np.zeros_like(e)
    t0 = time.time()
    for t in range(sampleSize):
        # pick a random set of edges
        r = np.multiply(np.random.rand(d,d), e)
        r[r >  1-P_EDGE] = 1.0
        r[r <= 1-P_EDGE] = 0.0
        #print '*** e has',e.sum(),'edges and r has',r.sum()
        # construct tc = r + r^2 + ... + r^n
        tc = r
        r_pow_i = np.eye(d,d)
        for i in range(1,n+1):
            r_pow_i = np.dot(r_pow_i,r)
            r_pow_i[r_pow_i>0] = 1.0
            tc += r_pow_i
        tc[tc>0] = 1.0
        #print 'tc has',tc.sum(),'edges'
        s += tc
        if (t) % 100 == 0:
            elapsed = time.time()-t0
            print 'finished',t,'samples in',elapsed,'sec at',(t+1)/elapsed,'samples per sec'
    s *= 1.0/sampleSize
    print 'max edges',(d-3)*(d-3),'edges',s.sum(),'frac',s.sum()/((d-3)*(d-3))
    outfile = stem+'_y.txt'
    with open(outfile,'w') as fp:
        for r in range(3,d):
            fp.write('%d' % r)
            for c in range(3,d):
              fp.write(' %g' % (s[r,c]))
            fp.write('\n')
    return outfile

def loadPrecomputed(db,loadFile):
    d = db.dim()
    target_y = np.zeros((d,d))
    print 'target_y',target_y.shape
    for line in open(loadFile):
        parts = line.strip().split(" ")
        r = int(parts[0])
        for j,p in enumerate(parts[1:]):
            target_y[r,j+3] = float(p)
    return target_y

def runMain(n='10',epochs='1000',repeat='1',gendata='0',load='precomputed-distributions/g10_y-sample-1M.txt'):
  n = int(n)
  epochs = int(epochs)
  repeat = int(repeat)
  gendata = int(gendata)
  if gendata>0:
    print 'generating data with',gendata,'trials'
    outfile = genInputs(n,gendata)
    print 'stored in',outfile
  else:
    losses = []
    for r in range(repeat):
        print 'trial',r+1
        if len(losses)>0: print 'running avg',sum(losses)/len(losses)
        (factFile,trainFile,testFile) = expt.genInputs(n)
        (tlog,trainData,testData) = setup_tlog(n,factFile,trainFile,testFile)
        target_y = loadPrecomputed(tlog.db,load)
        mode = 'path/io'
        (tx,_) = trainData[mode]
        trainData[mode] = (tx, np.dot(tx,target_y))
        (ux,_) = testData[mode]
        testData[mode] = (ux,np.dot(ux,target_y))
        loss = trainAndTest(tlog,trainData,testData,epochs,n)
        losses.append(loss)
    print 'losses',losses,'average',sum(losses)/len(losses)

if __name__=="__main__":
  optlist,args = getopt.getopt(sys.argv[1:],"x:",['n=','epochs=','repeat=','gendata=', 'load='])
  optdict = dict(map(lambda(op,val):(op[2:],val),optlist))
  print 'optdict',optdict
  runMain(**optdict)
