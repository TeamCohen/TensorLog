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
    
    def render(self,prog,fun):
        self.root = TK.Tk()
        self.scrollbar = ttk.Scrollbar(self.root)
        self.scrollbar.grid(row=0,column=0)
        self.tree = ttk.Treeview(self.root,height=30)
        self.tree["columns"]=("comment","output")
        self.tree.column("#0", width=300 )
        self.tree.column("comment", width=300 )
        self.tree.column("output", width=200)
        self.tree.heading("comment", text="comment")
        self.tree.heading("output", text="output")
        self.treeOuputs = {}
        self.populate([fun],"")

        self.msg = ttk.Treeview(self.root,height=30)        
        self.msg["columns"] = ("weight")
        self.msg.heading("weight", text="weight")
        self.msg.grid(row=0,column=2)
        self.msgItems = set()
        self.tree.bind("<Double-1>", self.OnDoubleClick)

        self.tree.grid(row=0,column=1)
        self.tree.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.tree.yview)
        
    def OnDoubleClick(self,event):
        key = self.tree.selection()[0]
        m = self.treeOuputs[key]
        print "you clicked on", key,"key",type(m),"shape",m.shape
        for it in self.msgItems:
            self.msg.delete(it)
        self.msgItems = set()
        dOfD = prog.db.matrixAsSymbolDict(self.treeOuputs[key])
        for r in sorted(dOfD.keys()):
            rowChild = self.msg.insert("",r,text="row %d" % r,open=True)
            self.msgItems.add(rowChild)
            for offset,sym in enumerate(sorted(dOfD[r].keys())):
                #TODO why are None keys in these?
                if sym!=None:
                    w = dOfD[r][sym]
                    child = self.msg.insert(rowChild,offset,text=sym,values=("%.5f" % w),open=True)


    def populate(self,funs,parent):
        for offset,fun in enumerate(funs):
            description = fun.pprintSummary()
            comment = fun.pprintComment()
            output = fun.output if fun.output!=None else '???'
            key = "iid%d" % len(self.treeOuputs.keys())
            child = self.tree.insert(
                parent,offset,iid=key,text=description,values=(comment,mutil.summary(output)),open=True)
            self.treeOuputs[key] = output
            self.populate(fun.children(), child)



if __name__ == "__main__":
    
    UNTRAINED_MODEL = False

    if UNTRAINED_MODEL:
        db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
    else:
        db = matrixdb.MatrixDB.deserialize('toy-trained.db')
    trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
    if UNTRAINED_MODEL:
        prog.setWeights(db.ones())
    fun,P = Debugger.evaluatedFunction(initProgram=prog,trainData=trainData,targetPred="predict/io")
    print "\n".join(fun.pprint())
    d = Debugger()
    d.render(prog,fun)
    d.root.mainloop()
