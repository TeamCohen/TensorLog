include Makefile.config
NOTIFY:="DATASETS" variable not set. For long tests, clone a copy of tensorlog-datasets.git and put the path to it in your Makefile.config. For more details, see Makefile.config.template

Makefile.config: Makefile.config.template
	cp $< $@

# unit tests that take a few secs to run
short-tests:
	nosetests testtensorlog.py

xctest:
	nosetests testxcomp.py

# tests that take a few minutes to run
# run 'make prepare-for-long-tests' in a fresh install before you run these
long-tests: short-tests benchmark-test wnet-test cora-test grid-test

prepare-for-long-tests:
	make fb15k-valid.db 
ifneq ($(DATASETS),)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/wordnet; make setup)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/cora; make setup)
else
	$(error $(NOTIFY))
endif

fb15k-valid.db:
	PYTHONPATH=$(PYTHONPATH):../; python matrixdb.py --serialize tensorlog/test-data/fb15k-valid.cfacts tensorlog/fb15k-valid.db

#
# individual long tests
#

benchmark-test:
	PYTHONPATH=../; python benchmark.py

wnet-test:
ifneq ($(DATASETS),)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/wordnet/; make clean; make; make check)
else
	$(error $(NOTIFY))
endif

cora-test:
ifneq ($(DATASETS),)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/cora/; make clean; make; make check)
else
	$(error $(NOTIFY))
endif

grid-test:
	(cd ../; PYTHONPATH=`pwd`; cd datasets/grid/; make clean; make; make check)

textcattoy-expt:
	PYTHONPATH=../; python -m expt --db test-data/textcattoy.cfacts --prog test-data/textcat.ppr --trainData test-data/toytrain.exam --testData test-data/toytrain.exam --proppr

# debug-test needs to be interactive
debug-test:
	PYTHONPATH=../; python expt.py --prog test-data/textcat.ppr --db test-data/textcattoy.cfacts \
		--trainData test-data/toytrain.exam --testData test-data/toytest.exam --proppr +++ --savedModel expt-model.db
	PYTHONPATH=../; python debug.py --prog test-data/textcat.ppr --db expt-model.db --trainData test-data/toytrain.exam --testData test-data/toytest.exam --proppr predict/io

# word count

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`

wc-all:
	wc `ls *.py | grep -v pyparsing`

# cleanup

clean:
	rm -f *.pyc *.py~ *.prof 
	rm -f toy-test.examples toy-test.solutions.txt toy-train.examples 
	rm -rf toy-trained.db

dist:
	tar -cvf ../tensorlog-dist.tar ../LICENSE Makefile Makefile.config.template *.py test-data/*.*
