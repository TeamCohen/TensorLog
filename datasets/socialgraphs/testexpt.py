import unittest
import demo

class TestAccTF(unittest.TestCase):

  def testCiteseer(self):
    init_acc,final_acc = demo.runMain("--stem citeseer".split())
    self.assertTrue( 0.62 <= init_acc < final_acc < 0.65 )

  def testCora(self):
    init_acc,final_acc = demo.runMain("--stem cora".split())
    self.assertTrue( 0.75 <= init_acc < final_acc < 0.80 )

  def testDolphins(self):
    init_acc,final_acc = demo.runMain("--stem dolphins".split())
    self.assertTrue( init_acc == final_acc == 1.0 )

  def testFootball(self):
    init_acc,final_acc = demo.runMain("--stem football --regularizer_scale 1.0".split())
    self.assertTrue( 0.43 < init_acc < 0.45 )
    self.assertTrue( 0.70 < final_acc < 0.75 )

  def testKarate(self):
    init_acc,final_acc = demo.runMain("--stem karate".split())
    self.assertTrue( 0.90 < init_acc < 1.0 )
    self.assertTrue( 0.90 < final_acc < 1.0 )

  def testUMBC(self):
    init_acc,final_acc = demo.runMain("--stem umbc --link_scale 0.1".split())
    self.assertTrue( 0.94 < init_acc < final_acc < 0.95)

if __name__ == "__main__":
  unittest.main()
