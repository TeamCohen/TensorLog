import tensorlog
import declare
import learn
import time

if __name__=="__main__":
    mode = declare.ModeDeclaration('samebib(i,o)')
    ti = tensorlog.Interp(initFiles=["cora.cfacts","cora.ppr"])
