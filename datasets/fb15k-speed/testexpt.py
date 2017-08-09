import unittest
import tfexpt
import expt

class TestTimeNative(unittest.TestCase):

  # these tests are GPU-dependent
  def testIt(self):
    fps,qps1,qps2 = expt.runMain()
    print 'fps,qps1,qps2 are',fps,qps1,qps2
    self.assertTrue(fps >= 700.0)  # compilation
    self.assertTrue(qps1 >= 250.0)  # minibatches size = 1
    self.assertTrue(qps2 >= 1500.0)  # minibatches size = as large as possible

class TestTimeTF(unittest.TestCase):

  def testIt(self):
    fps,qps = tfexpt.runMain()
    print 'fps,qps are',fps,qps
    self.assertTrue(fps >= 2.0)
    self.assertTrue(qps >= 20.0)

if __name__ == "__main__":
  unittest.main()
