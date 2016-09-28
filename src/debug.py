# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for debugging/visualization
#

import sys
import Tkinter as TK
import ttk
import tkFont
import time

import tensorlog
import dataset
import matrixdb
import tensorlog
import declare
import learn
import mutil
import config
import opfunutil

conf = config.Config()
conf.sortByValue = True;   conf.help.sortByValue = "In displaying message values, sort entries by weight if true, by name if false."
conf.fontsize = None;      conf.help.fontsize = "Size of font, eg 14"
conf.fontweight = None;    conf.help.fontsize = "Weight of font, eg 'bold'"

class Debugger(object):

    def __init__(self,initProgram,targetPred,trainData,gradient=False):
        self.rendered = False
        self.sortByValue = conf.sortByValue
        self.prog = initProgram
        self.trainData = trainData
        self.targetPred = targetPred

        #evaluate the function so the outputs are cached
        assert self.targetPred,'most specify targetPred'
        self.mode = declare.asMode(self.targetPred)
        assert self.trainData.hasMode(self.mode),"No mode '%s' in trainData" % self.mode
        self.X = self.trainData.getX(self.mode)
        self.Y = self.trainData.getY(self.mode)
        self.fun = self.prog.getPredictFunction(self.mode)
        self.pad = opfunutil.Scratchpad()
        self.P = self.fun.eval(self.prog.db, [self.X], self.pad)
        # find the symbols that correspond to the inputs
        dd = self.prog.db.matrixAsSymbolDict(self.X)
        self.xSymbols = [d.keys()[0] for d in dd.values()]

        # evaluate the gradient so that's cached
        if gradient:
            learner = learn.OnePredFixedRateGDLearner(self.prog, tracer=learn.Tracer.silent)
            self.grad = learner.crossEntropyGrad(self.mode, self.X, self.Y, pad=self.pad)
        else:
            self.grad = None
    
    def render(self):
        #set up a window
        self.root = TK.Tk()
        default_font = tkFont.nametofont("TkDefaultFont")
        if conf.fontsize:
            default_font.configure(size=conf.fontsize)
        if conf.fontweight:
            default_font.configure(weight=conf.fontweight)
        self.root.option_add("*Font", default_font)
        #labels on the top
        self.treeLabel = ttk.Label(self.root,text="Listing of %s" % str(self.mode))
        self.treeLabel.grid(row=0,column=1,sticky=TK.EW)
        self.msgLabel = ttk.Label(self.root,text="Details")
        self.msgLabel.grid(row=0,column=2,sticky=TK.EW)
        #put a scrollbars on the left and right
        #these don't work now? maybe they worked with pack?
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
        self.tree.bind("<Button-1>", self.DisplayMsg)
        # tie it to the right scrollbar
#        self.tree.config(yscrollcommand=self.scrollbarR.set)
#        self.scrollbarR.config(command=self.msg.yview)
        
    def DisplayMsg(self,event):
        """display the message sent by with an op
        or the output for a function."""
        key = self.tree.identify_row(event.y)
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
                def sortKey(k): 
                    if self.sortByValue==True:
                        return -dOfD[r][k]
                    else:
                        return k
                for offset,sym in enumerate(sorted(dOfD[r].keys(), key=sortKey)):
                    #why are some of these None?
                    if sym!=None:
                        w = dOfD[r][sym]
                    child = self.msg.insert(rowChild,offset,text=sym,values=("%.5f" % w),open=True)


    def populateTree(self,funs,parent):
        for offset,fun in enumerate(funs):
            description = fun.pprintSummary()
            comment = fun.pprintComment()
            key = "iid%d" % len(self.treeOutputs.keys())
            funOutput = self.pad[fun.id].output
            if self.grad:
                #todo: clean up
                if 'delta' in self.pad[fun.id].__dict__:
                    funDelta = self.pad[fun.id].delta
                else:
                    funDelta = None
            else:
                funDelta = None
            child = self.tree.insert(
                parent,offset,iid=key,text=description,
                values=(comment,mutil.pprintSummary(funOutput),mutil.pprintSummary(funDelta)),open=True)
            self.treeOutputs[key] = funOutput
            self.treeDeltas[key] = funDelta
            self.populateTree(fun.children(), child)

    def mainloop(self):
        if not self.rendered:
            self.render()
        self.root.mainloop()

if __name__ == "__main__":
    
    def usage():
        print 'debug.py [usual tensorlog options] mode [inputs]'

    optdict,args = tensorlog.parseCommandLine(sys.argv[1:])
    dset = optdict.get('trainData') or optdict.get('testData')
    if dset==None and len(args)<2:
        usage()
        print 'debug on what input? specify --trainData or give a function input'
    elif len(args)<1:
        usage()
    elif dset and len(args)>2:
        print 'using --trainData not the function input given'
    elif dset:
        mode = declare.asMode(args[0])
        Debugger(optdict['prog'],mode,dset,gradient=True).mainloop()
    else:
        mode = declare.asMode(args[0])
        X = optdict['prog'].db.onehot(args[1])
        dset = dataset.Dataset({mode:X},{mode:optdict['prog'].db.zeros()})
        Debugger(optdict['prog'],mode,dset,gradient=False).mainloop()

