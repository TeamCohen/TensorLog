import logging
import time

import tensorflow as tf

from tensorlog import dataset
from tensorlog import declare
from tensorlog import expt
from tensorlog import learn
from tensorlog import plearn
from tensorlog import comline
from tensorlog import simple

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info('level is info')

    db = comline.parseDBSpec('inputs/cora.cfacts')
    prog = comline.parseProgSpec("cora.ppr",db,proppr=True)
    prog.setRuleWeights(ruleIdPred='rule')
    for p in ['kaw','ktw','kvw']:
      prog.db.markAsParam(p,1)
      prog.db.setParameter(p,1,db.ones())
    prog.maxDepth = 1

    tlog = simple.Compiler(db=db, prog=prog, autoset_db_params=False)
    train_data = tlog.load_small_dataset('inputs/train.examples')
    test_data = tlog.load_small_dataset('inputs/test.examples')

    print 'loaded db and train/test datasets:',type(train_data),'keys',train_data.keys()
    mode = 'samebib/io'
    TX,TY = train_data[mode]
    UX,UY = test_data[mode]
    inference = tlog.inference(mode)
    loss = tlog.loss(mode)
    optimizer = tf.train.AdagradOptimizer(0.1)
    train_step = optimizer.minimize(loss)
    train_batch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, tlog.target_output_placeholder_name(mode):UY}

    t0 = time.time()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    epochs = 30
    for i in range(epochs):
      # progress
      print 'epoch',i+1,'of',epochs
      session.run(train_step, feed_dict=train_batch_fd)
    print 'learning time',time.time()-t0,'sec'

    predicted = session.run(inference, feed_dict=test_batch_fd)
    m = declare.asMode('samebib/io')
    native_test_data = dataset.Dataset({m:tlog.xc._unwrapOutput(UX)},{m:tlog.xc._unwrapOutput(UY)})
    savedTestExamples = 'tmp-cache/cora-test.examples'
    savedTestPredictions = 'tmp-cache/cora-test.solutions.txt'
    native_test_data.saveProPPRExamples(savedTestExamples,tlog.db)
    expt.Expt.predictionAsProPPRSolutions(savedTestPredictions,'samebib',tlog.db,tlog.xc._unwrapOutput(UX),tlog.xc._unwrapOutput(predicted))
    print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' % (savedTestExamples,savedTestPredictions)
