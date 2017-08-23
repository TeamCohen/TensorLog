import unittest
import tfexpt
import expt

class TestNative(unittest.TestCase):
    def setUp(self):
        (self.prog, self.trainData, self.testData) = expt.setExptParams()
    def testIt(self):
        acc,loss = expt.accExpt(self.prog, self.trainData, self.testData)
        print "acc",acc
        self.assertTrue(acc >= 0.71)

class TestAccTF(unittest.TestCase):
    def setUp(self):
        self.params = tfexpt.setup_tlog()
    def testIt(self):
        accs = tfexpt.trainAndTest(*self.params)
        self.assertTrue(False) # TODO

if __name__=='__main__':
    unittest.main()
    
