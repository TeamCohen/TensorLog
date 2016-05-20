default:
	echo make what?

fb.db: fb.cfacts
	time python $(TLOG)/matrixdb.py --serialize fb.cfacts fb.db

fb.cfacts: fb-ruleids.cfacts
	cut -f2- raw/train.cfacts raw/valid.cfacts > fb-core.cfacts
	python bin/convert-examples.py
	cat fb-*.cfacts > fb.cfacts

fb-learned.ppr fb-ruleids.cfacts:
	python bin/convert-rules.py

clean:
	rm -f *.pyc *.prof *~
