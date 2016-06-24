TLOG=../../src

default:
	echo make what?

setup: wnet.db wnet-learned.ppr

wnet.db: wnet.cfacts
	time python $(TLOG)/matrixdb.py --serialize wnet.cfacts wnet.db

wnet.cfacts: wnet-ruleids.cfacts
	cut -f2- raw/train.cfacts raw/valid.cfacts | perl -nae 'print join("\t",$$F[0],"s".$$F[1],"s".$$F[2]),"\n"' > wnet-core.cfacts
	python bin/convert-examples.py
	cat wnet-*.cfacts > wnet.cfacts

wnet-learned.ppr wnet-ruleids.cfacts:
	python bin/convert-rules.py

clean:
	rm -f *.pyc *.prof *~
	rm -f wnet-*-train.examples wnet-*-test.examples wnet-*-test.solutions.txt wnet-*-train.cfacts wnet-*-valid.cfacts
	rm -rf *-trained.db	

