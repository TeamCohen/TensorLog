# (C) William W. Cohen and Carnegie Mellon University, 2016

class Config(object):

    def __init__(self):
        self.help = ConfigHelp()

    """A container for configuration options"""
    def pprint(self,depth=0):
        if not depth: print '===='
        for key,val in self.__dict__.items():
            if key!='help':
                self._explain(depth,key,val)
                if type(val)==type(Config()):
                    val.pprint(depth+1)

    def _explain(self,depth,key,val):
        tmp = '| '*depth + key + ':'
        if type(val)!=type(Config()):
            tmp += ' '+repr(val)
        if key in self.help.__dict__:
            print '%-40s %s' % (tmp,self.help.__dict__[key])
            

class ConfigHelp(object):
    """A parallel object that stores help about configurations."""
    pass

