# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for debugging/visualization
#

#TODO next: try double-clicking to see the deltas by swapping from treeOutputs to treeDeltas

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

    def __init__(self):
        #TODO switch this somehow more reasonably, maybe from where
        #the double-click happens?
        self.viewDelta = False

    #TODO make this non-static, maybe part of the init?
    @staticmethod
    def evaluatedFunction(initProgram=None,trainData=None,targetPred=None):
        """Evaluate a function, so the outputs are cached, compute the
        gradient so the deltas are cached, and return it."""

        assert targetPred,'most specify targetPred'
        mode = declare.asMode(targetPred)
        assert trainData.hasMode(mode)
        X = trainData.getX(mode)
        Y = trainData.getY(mode)
        fun = initProgram.getPredictFunction(mode)
        P = fun.eval(initProgram.db, [X])
        learner = learn.Learner(prog)
        grad = learner.crossEntropyGrad(mode,X,Y)
        return fun
    
    def render(self,prog,fun):
        #set up a window
        self.root = TK.Tk()
        #put a scrollbar on the left
        self.scrollbar = ttk.Scrollbar(self.root)
        self.scrollbar.grid(row=0,column=0)
        #set up a treeview widget and tie it to the scrollbar
        self.tree = ttk.Treeview(self.root,height=30)
        self.tree.grid(row=0,column=1)
        self.tree.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.tree.yview)
        #adjust the columns
        self.tree["columns"]=("comment","output","delta")
        self.tree.column("#0", width=300 )
        self.tree.column("comment", width=300 )
        self.tree.column("output", width=150)
        self.tree.column("delta", width=150)
        self.tree.heading("comment", text="comment")
        self.tree.heading("output", text="output")
        self.tree.heading("delta", text="delta")
        # save the function/op deltas and outputs for each tree node,
        # indexed by the tree id
        self.treeOutputs = {}
        self.treeDeltas = {}
        #fill the tree with the function and its children
        self.populateTree([fun],"")

        # set up another treeview to display the function output/deltas,
        # which will be triggered when you doubleclick
        self.msg = ttk.Treeview(self.root,height=30)        
        self.msg["columns"] = ("weight")
        self.msg.heading("weight", text="weight")
        self.msg.grid(row=0,column=2)
        self.msgItems = set()
        self.tree.bind("<Double-1>", self.DisplayMsg)
        
    def DisplayMsg(self,event):
        key = self.tree.selection()[0]
        if self.viewDelta:
            print 'viewing deltas'
            m = self.treeDeltas[key]
        else:
            print 'viewing outputs'
            m = self.treeOutputs[key]
        for it in self.msgItems:
            self.msg.delete(it)
        self.msgItems = set()
        dOfD = prog.db.matrixAsSymbolDict(m)
        for r in sorted(dOfD.keys()):
            rowChild = self.msg.insert("",r,text="row %d" % r,open=True)
            self.msgItems.add(rowChild)
            for offset,sym in enumerate(sorted(dOfD[r].keys())):
                #TODO why are None keys in these?
                if sym!=None:
                    w = dOfD[r][sym]
                    child = self.msg.insert(rowChild,offset,text=sym,values=("%.5f" % w),open=True)


    def populateTree(self,funs,parent):
        for offset,fun in enumerate(funs):
            description = fun.pprintSummary()
            comment = fun.pprintComment()
            output = fun.output if fun.output!=None else '???'
            if fun.delta!=None:
                deltaSummary = mutil.summary(fun.delta)
            else:
                deltaSummary = "___"
            key = "iid%d" % len(self.treeOutputs.keys())
            child = self.tree.insert(
                parent,offset,iid=key,text=description,
                values=(comment,mutil.summary(output),deltaSummary),open=True)
            self.treeOutputs[key] = output
            self.treeDeltas[key] = fun.delta
            self.populateTree(fun.children(), child)

if __name__ == "__main__":
    
    TRAINED = True

    if not TRAINED:
        db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
    else:
        db = matrixdb.MatrixDB.deserialize('toy-trained.db')
    trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
    if not TRAINED:
        prog.setWeights(db.ones())
    fun = Debugger.evaluatedFunction(initProgram=prog,trainData=trainData,targetPred="predict/io")

    print "\n".join(fun.pprint())
    d = Debugger()
    d.viewDelta = True
    d.render(prog,fun)
    d.root.mainloop()
