short-tests:
	python testtensorlog.py

long-tests: short-tests benchmark-test wnet-test cora-test-v2

benchmark-test:
	python benchmark.py

#fb15k-valid.db: matrixdb.py
#	python $< --serialize test/fb15k-valid.cfacts:test/fb15k.tensorLog.cfacts $@

fb15k-valid.db:
	python matrixdb.py --serialize test/fb15k-valid.cfacts  fb15k-valid.db

wnet-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet/; python wnet-expt.py; \
	 proppr eval hypernym-test.examples hypernym-test.solutions.txt --metric auc --defaultNeg)
	echo expected . micro: 0.93437229711

cora-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/cora/; python cora-expt.py; \
	 proppr eval cora-test.examples cora-test.solutions.txt --metric auc --defaultNeg)
	echo expected . micro: 0.975902362875

cora-test-v2:
	(export PYTHONPATH=`pwd`; cd ../datasets/cora/; python cora-exptv2.py; \
	 proppr eval tmp-cache/cora-test.examples tmp-cache/cora-test.solutions.txt --metric auc --defaultNeg)
	echo expected . micro: 0.975902362875

prepare-for-long-tests:
	make fb15k-valid.db 
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet; make setup)
	(export PYTHONPATH=`pwd`; cd ../datasets/cora; make setup)
snap:
	tar -czf backup/snap`date +%m.%d.%H.%M.%S`.tgz *.py Makefile
	ls -alh backup

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`

wc-all:
	wc `ls *.py | grep -v pyparsing`

clean:
	rm -f *.pyc *.py~ *.prof
	rm -f toy-test.examples toy-test.solutions.txt toy-train.examples 
	rm -rf toy-trained.db
