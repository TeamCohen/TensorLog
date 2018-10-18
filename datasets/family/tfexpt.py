import logging
import time

import tensorflow as tf
from tensorlog import simple
import expt

def setup_tlog():
    (prog, native_trainData, native_testData) = expt.setExptParams()
    tlog = simple.Compiler(db=prog.db, prog=prog, autoset_db_params=False)
    train_data = tlog.annotate_big_dataset(native_trainData)
    test_data = tlog.annotate_small_dataset(native_testData)
    return (tlog,train_data,test_data,native_trainData.modesToLearn())

def trainAndTest(tlog,train_data,test_data,modes):
    result={}
    for mode in modes:
        print(mode)
        loss = tlog.loss(mode)
        optimizer = tf.train.AdagradOptimizer(0.1)
        train_step = optimizer.minimize(loss)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        t0 = time.time()
        epochs = 10
        for i in range(epochs):
            b = 0
            print('epoch',i+1,'of',epochs)
            for (_,(TX,TY)) in tlog.minibatches(train_data,batch_size=125):
                train_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
                session.run(train_step, feed_dict=train_fd)
                b += 1
        print('learning time',time.time()-t0,'sec')

        predicted_y = tlog.inference(mode)
        actual_y = tlog.target_output_placeholder(mode)
        correct_predictions = tf.equal(tf.argmax(actual_y,1),tf.argmax(predicted_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        mode_str = str(mode)
        if mode_str in test_data:
            UX,UY = test_data[mode_str]
            test_fd = {tlog.input_placeholder_name(mode):UX, tlog.target_output_placeholder_name(mode):UY}
            acc = session.run(accuracy, feed_dict=test_fd)
            print(mode_str, 'test acc',acc)
            result[mode_str] = acc
    return result

def runMain():
    params = setup_tlog()
    return trainAndTest(*params)

if __name__=='__main__':
    #params = setup_tlog()
    accs = runMain()
    for mode,acc in list(accs.items()):
        print(mode,'acc',acc)
