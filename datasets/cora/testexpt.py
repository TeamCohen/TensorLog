import unittest
import tfexpt
import expt

class TestAccNative(unittest.TestCase):

  def testIt(self):
    acc,loss = expt.runMain()
    self.assertTrue(acc >= 0.1065)

class TestAccTF(unittest.TestCase):

  def testIt(self):
    acc = tfexpt.runMain(saveInPropprFormat=False)
    self.assertTrue(acc >= 0.29)

if __name__ == "__main__":
  unittest.main()
