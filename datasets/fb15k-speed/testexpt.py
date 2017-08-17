import unittest
import tfexpt
import expt

class TestTimeNative(unittest.TestCase):

  def testIt(self):
    fps,qps1,qps2 = expt.runMain()
    self.assertTrue(fps >= 1500.0)  # compilation
    self.assertTrue(qps1 >= 300.0)  # minibatches size = 1
    self.assertTrue(qps2 >= 2000.0)  # minibatches size = as large as possible

class TestTimeTF(unittest.TestCase):

  def testIt(self):
    fps,qps = tfexpt.runMain()
    self.assertTrue(fps >= 2.0)
    self.assertTrue(qps >= 20.0)

if __name__ == "__main__":
  unittest.main()
