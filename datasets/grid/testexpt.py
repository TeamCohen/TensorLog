import unittest
import expt

from tensorlog import matrixdb
from tensorlog import program

class TestAcc(unittest.TestCase):

  def setUp(self):
    (self.n,self.maxD,self.epochs) = (16,8,20)
    (self.factFile,self.trainFile,self.testFile) = expt.genInputs(self.n)
    self.db = matrixdb.MatrixDB.loadFile(self.factFile)
    self.prog = program.Program.loadRules("grid.ppr",self.db)

  def testIt(self):
    acc,loss = expt.accExpt(self.prog,self.trainFile,self.testFile,self.n,self.maxD,self.epochs)
    self.assertTrue(acc >= 0.95)
    times = expt.timingExpt(self.prog)
    for t in times:
      self.assertTrue(t < 0.005)

if __name__ == "__main__":
  unittest.main()
