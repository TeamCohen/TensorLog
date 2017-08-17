import unittest
import tfexpt
import expt

class TestAccNative(unittest.TestCase):

  def testIt(self):
    acc,loss = expt.runMain(250)
    self.assertTrue(acc >= 0.205)

class TestAccTF(unittest.TestCase):

  def testIt(self):
    acc = tfexpt.runMain(250)
    self.assertTrue(acc >= 0.27)

if __name__ == "__main__":
  unittest.main()
