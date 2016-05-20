default:
	echo make what?

wnet.db: wnet.cfacts
	time python $(TLOG)/matrixdb.py --serialize wnet.cfacts wnet.db

wnet.cfacts: wnet-ruleids.cfacts
	cut -f2- raw/train.cfacts raw/valid.cfacts > wnet-core.cfacts
	python bin/convert-examples.py
	cat wnet-*.cfacts > wnet.cfacts

wnet-learned.ppr wnet-ruleids.cfacts:
	python bin/convert-rules.py

clean:
	rm -f *.pyc *.prof *~
	rm hypernym-train.examples hypernym-test.examples hypernym-test.solutions.txt
	rm -rf hypernym-trained.db	
