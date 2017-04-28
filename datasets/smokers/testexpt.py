import unittest
import tfexpt
import expt

class TestTimeNative(unittest.TestCase):

  def testIt(self):
    time = expt.runMain()
    self.assertTrue(time <= 0.1) 

class TestTimeTF(unittest.TestCase):

  def testIt(self):
      time = tfexpt.runMain()
      self.assertTrue(time < 0.5)

if __name__ == "__main__":
  unittest.main()
