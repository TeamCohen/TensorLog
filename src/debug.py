# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for debugging/visualization
#

#TODO next: try double-clicking to see the deltas by swapping from treeOutputs to treeDeltas

import Tkinter as TK
import ttk
import tkFont
import time

import dataset
import matrixdb
import tensorlog
import declare
import learn
import mutil
import config

class Debugger(object):

    def __init__(self,initProgram,targetPred,trainData,gradient=False):
        self.rendered = False
        self.prog = initProgram
        self.trainData = trainData
        self.targetPred = targetPred

        #evaluate the function so the outputs are cached
        assert self.targetPred,'most specify targetPred'
        self.mode = declare.asMode(self.targetPred)
        assert self.trainData.hasMode(self.mode)
        self.X = self.trainData.getX(self.mode)
        self.Y = self.trainData.getY(self.mode)
        self.fun = self.prog.getPredictFunction(self.mode)
        self.P = self.fun.eval(self.prog.db, [self.X])

        # find the symbols that correspond to the inputs
        dd = self.prog.db.matrixAsSymbolDict(self.X)
        self.xSymbols = [d.keys()[0] for d in dd.values()]

        # evaluate the gradient so that's cached
        if gradient:
            self.learner = learn.Learner(prog)
            self.grad = self.learner.crossEntropyGrad(self.mode,self.X,self.Y)
    
    def render(self):
        #set up a window
        self.root = TK.Tk()
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=16,weight='bold')
        self.root.option_add("*Font", default_font)
        #labels on the top
        self.treeLabel = ttk.Label(self.root,text="Listing of %s" % str(self.mode))
        self.treeLabel.grid(row=0,column=1,sticky=TK.EW)
        self.msgLabel = ttk.Label(self.root,text="Details")
        self.msgLabel.grid(row=0,column=2,sticky=TK.EW)
        #put a scrollbars on the left and right
        #these don't work now?
#        self.scrollbarL = ttk.Scrollbar(self.root)
#        self.scrollbarL.grid(row=1,column=0)
#        self.scrollbarR = ttk.Scrollbar(self.root)
#        self.scrollbarR.grid(row=1,column=4)
        #set up a treeview widget and tie it to the left scrollbar
        self.tree = ttk.Treeview(self.root)
        self.tree.grid(row=1,column=1,sticky=TK.NSEW)
#        self.tree.config(yscrollcommand=self.scrollbarL.set)
#        self.scrollbarL.config(command=self.tree.yview)
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
        self.populateTree([self.fun],"")

        # set up another treeview to display the function output/deltas,
        # which will be triggered when you doubleclick
        self.msg = ttk.Treeview(self.root,height=30)        
        self.msg["columns"] = ("weight")
        self.msg.heading("weight", text="weight")
        self.msg.grid(row=1,column=2)
        self.msgItems = set()
        #tree will fill the msg window on doubleclick
        self.tree.bind("<Double-1>", self.DisplayMsg)
        # tie it to the right scrollbar
#        self.tree.config(yscrollcommand=self.scrollbarR.set)
#        self.scrollbarR.config(command=self.msg.yview)
        
    def DisplayMsg(self,event):
        """display the message sent by with an op
        or the output for a function."""
        key = self.tree.selection()[0]
        # figure out where we clicked - returns #0, #1, ... 
        colStr = self.tree.identify_column(event.x)
        colNum = int(colStr[1:])
        tag = self.tree.item(key,option='text')
        if colNum>=3:
            m = self.treeDeltas[key]
            if m==None:
                self.msgLabel.config(text='Delta for %s unavailable' % tag)
            else:
                self.msgLabel.config(text='Delta for %s' % tag)
        else:
            self.msgLabel.config(text='Output for %s' % tag)
            m = self.treeOutputs[key]
        for it in self.msgItems:
            self.msg.delete(it)
        self.msgItems = set()
        if m!=None:
            dOfD = self.prog.db.matrixAsSymbolDict(m)
            rowVector = len(dOfD.keys())==1
            for r in sorted(dOfD.keys()):
                rowName = "Row Vector:" if rowVector else self.xSymbols[r]
                rowChild = self.msg.insert("",r,text=rowName,open=True)
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
            key = "iid%d" % len(self.treeOutputs.keys())
            child = self.tree.insert(
                parent,offset,iid=key,text=description,
                values=(comment,mutil.pprintSummary(fun.output),mutil.pprintSummary(fun.delta)),open=True)
            self.treeOutputs[key] = fun.output
            self.treeDeltas[key] = fun.delta
            self.populateTree(fun.children(), child)

    def mainloop(self):
        if not self.rendered:
            self.render()
        self.root.mainloop()

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

    db = Debugger(prog,"predict/io",trainData,gradient=False)
#    default_font = tkFont.nametofont("TkDefaultFont")
#    default_font.configure(size=14)
    #ttk.Style().theme_use('clam')
    db.root.mainloop()

