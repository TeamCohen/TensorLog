import unittest
import tfexpt
import expt

class TestTimeNative(unittest.TestCase):

  # these tests are GPU-dependent
  def testIt(self):
    fps,qps1,qps2 = expt.runMain()
    print('fps,qps1,qps2 are',fps,qps1,qps2)
    self.assertTrue(fps >= 650.0)  # compilation
    self.assertTrue(qps1 >= 100.0)  # minibatches size = 1
    self.assertTrue(qps2 >= 750.0)  # minibatches size = as large as possible

class TestTimeTF(unittest.TestCase):

  def testIt(self):
    fps,qps = tfexpt.runMain()
    print('fps,qps are',fps,qps)
    self.assertTrue(fps >= 1.5)
    self.assertTrue(qps >= 8.0)

if __name__ == "__main__":
  unittest.main()
