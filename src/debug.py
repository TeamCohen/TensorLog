# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for debugging/visualization
#

import Tkinter as TK
import ttk
import time

import dataset
import matrixdb
import tensorlog
import declare
import learn
import mutil
import config

class Debugger(object):

    @staticmethod
    def evaluatedFunction(initProgram=None,trainData=None,targetPred=None):

        if targetPred: targetPred = declare.asMode(targetPred)
        ti = tensorlog.Interp(initProgram=initProgram)

        tmodes = trainData.modesToLearn()
        if targetPred!=None: 
            assert targetPred in tmodes, 'target predicate %r not in training data' % targetPred

        singlePred = targetPred!=None or (targetPred==None and len(tmodes)==1)
        assert singlePred,'multiple preds not implemented'

        mode = targetPred or tmodes[0]
        assert trainData.hasMode(mode)
        X,T = trainData.getX(mode),trainData.getY(mode)
        
        fun = initProgram.getPredictFunction(mode)
        P = fun.eval(initProgram.db, [X])
        return fun,P
    
    @staticmethod
    def render(fun):
        root = TK.Tk()
        tree = ttk.Treeview(root)
        tree["columns"]=("fun","input","output")
        tree.column("input")
        tree.column("output")
        tree.heading("input", text="input")
        tree.heading("output", text="output")
        def populate(tree,funs,parent):
            for offset,fun in enumerate(funs):
                child = tree.insert(parent,offset,text=repr(type(fun)),values=("in","out"))
                populate(tree, fun.children(), child)
        populate(tree,[fun],"")
        tree.pack()
        root.mainloop()

if __name__ == "__main__":
    
    db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
    trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
    prog.setWeights(db.ones())
    fun,P = Debugger.evaluatedFunction(initProgram=prog,trainData=trainData,targetPred="predict/io")
    print "\n".join(fun.pprint())
    Debugger.render(fun)
        
