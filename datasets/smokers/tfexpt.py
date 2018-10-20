import sys
import time

from tensorlog import simple
import tensorflow as tf

import expt

def runMain():
    (ti,sparseX) = expt.setExptParams()
    X = sparseX.todense()

    # compile all the functions we'll need before I set up the session
    tlog = simple.Compiler(db=ti.db, prog=ti.prog, autoset_db_params=False)
    for modeString in ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]:
        _ = tlog.inference(modeString)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    start0 = time.time()
    for modeString in ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]:
        session.run(tf.global_variables_initializer())
        print('eval',modeString, end=' ')
        fd = {tlog.input_placeholder_name(modeString):X}
        session.run(tlog.inference(modeString), feed_dict=fd)
        print('time',time.time() - start0,'sec')
    tot = time.time() - start0
    print('total time',tot,'sec')
    return tot

if __name__=="__main__":
    t = runMain()
    print('time',t)
