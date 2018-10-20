import unittest
import tfexpt
import expt

from tensorlog import matrixdb
from tensorlog import program
from tensorlog import dataset

class TestNative(unittest.TestCase):

  def setUp(self):
    (self.n,self.maxD,self.epochs) = (16,8,20)
    (self.factFile,trainFile,testFile) = expt.genInputs(self.n)

#    (self.factFile,self.trainFile,self.testFile) = ('inputs/g16.cfacts','inputs/g16-train.exam','inputs/g16-test.exam')
    self.db = matrixdb.MatrixDB.loadFile(self.factFile)
    self.prog = program.Program.loadRules("grid.ppr",self.db)
    self.trainData = dataset.Dataset.loadExamples(self.prog.db,trainFile)
    self.testData = dataset.Dataset.loadExamples(self.prog.db,testFile)

  def testIt(self):
    acc,loss = expt.accExpt(self.prog,self.trainData,self.testData,self.n,self.maxD,self.epochs)
    print('acc',acc)
    self.assertTrue(acc >= 0.85)
    times = expt.timingExpt(self.prog)
    for t in times:
      print('time',t)
      self.assertTrue(t < 0.05)

class TestAccTF(unittest.TestCase):

  def setUp(self):
    (self.n,self.maxD,self.epochs) = (16,8,20)
    (self.factFile,self.trainFile,self.testFile) = expt.genInputs(self.n)
    (self.tlog,self.trainData,self.testData) = tfexpt.setup_tlog(self.maxD,self.factFile,self.trainFile,self.testFile)

  def testIt(self):
    acc = tfexpt.trainAndTest(self.tlog,self.trainData,self.testData,self.epochs)
    print('acc',acc)
    self.assertTrue(acc >= 0.85)

if __name__ == "__main__":
  unittest.main()
