import unittest
import tfexpt
import expt

class TestNative(unittest.TestCase):
    def setUp(self):
        (self.prog, self.trainData, self.testData) = expt.setExptParams()
    def testIt(self):
        acc,loss = expt.accExpt(self.prog, self.trainData, self.testData)
        print("acc",acc)
        self.assertTrue(acc >= 0.71)

TF_EXPECTED = {'i_husband/io':1.0,
               'i_brother/io':1.0,
               'i_uncle/io':0.5,
               'i_daughter/io':0.0, # not sure about this
               'i_wife/io':1.0,
               'i_aunt/io':0.5,
               'i_mother/io':1.0,
               'i_sister/io':1.0,
               'i_son/io':0.666666,
               'i_niece/io':0.0, # not sure about this
               'i_father/io':0.666666,
               'i_nephew/io':0.0 # not sure about this
           }
class TestAccTF(unittest.TestCase):
    def setUp(self):
        self.params = tfexpt.setup_tlog()
    def testIt(self):
        accs = tfexpt.trainAndTest(*self.params)
        for mode,acc in list(TF_EXPECTED.items()):
            self.assertTrue(accs[mode] >= acc,msg="%s:%g<%g" % (mode,accs[mode],acc))

        

if __name__=='__main__':
    unittest.main()
    
