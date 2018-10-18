import unittest
import tfexpt
import expt

TF_REFERENCE = {
    'concept_worksfor/io':1.0,
    'concept_atdate/io':0.733333,
    'concept_languageofcountry/io':1.0,
    'concept_personleadsorganization/io':1.0,
    'concept_agriculturalproductcamefromcountry/io':1.0,
    'concept_mutualproxyfor/io':1.0,
    'concept_subpartof/io':1.0,
    'concept_politicianholdsoffice/io':1.0,
    'concept_agentcontrols/io':1.0,
    'concept_agentcollaborateswithagent/io':1.0,
    'concept_teamalsoknownas/io':1.0,
    'concept_citylocatedingeopoliticallocation/io':1.0,
    'concept_automobilemakerdealersincountry/io':1.0,
    'concept_agentactsinlocation/io':1.0,
    'concept_istallerthan/io':1.0,
    'concept_personbelongstoorganization/io':1.0,
    'concept_competeswith/io':1.0,
    'concept_personhasresidenceingeopoliticallocation/io':1.0,
    'concept_atlocation/io':1.0,
    'concept_weaponmadeincountry/io':0.0,
    'concept_automobilemakercardealersinstateorprovince/io':1.0,
    'concept_countrylocatedingeopoliticallocation/io':1.0,
    'concept_productproducedincountry/io':0.0,
    'concept_agentinvolvedwithitem/io':1.0,
    'concept_agentcreated/io':1.0,
    'concept_agentparticipatedinevent/io':1.0,
    'concept_journalistwritesforpublication/io':1.0,
    'concept_hasofficeincity/io':1.0,
    'concept_locationlocatedwithinlocation/io':1.0,
    'concept_subpartoforganization/io':1.0,
    'concept_proxyfor/io':1.0,
    'concept_acquired/io':1.0
    }
        
class TestNative(unittest.TestCase):
    def testIt(self):
        acc,loss = expt.runMain()
        print("acc",acc)
        self.assertTrue(acc >= 0.69)

class TestAccTF(unittest.TestCase):
    def setUp(self):
        self.params = tfexpt.setup_tlog()
    def testIt(self):
        accs = tfexpt.trainAndTest(*self.params)
        for mode,acc in list(accs.items()):
            self.assertTrue(acc >= TF_REFERENCE[mode],"%s:%g<%g"%(mode,acc,TF_REFERENCE[mode]))

if __name__=='__main__':
    unittest.main()
    
