include Makefile.config
NOTIFY:="DATASETS" variable not set. For long tests, clone a copy of tensorlog-datasets.git and put the path to it in your Makefile.config. For more details, see Makefile.config.template

Makefile.config: Makefile.config.template
	cp $< $@

# unit tests that take a few secs to run  - all of these are about a minute all told
short-tests:
	python -m unittest testtensorlog
	python -m unittest testxcomp $(XCOMP_FILTER)
	python -m unittest testtf

all-automatic-tests: short-tests grid-test cora-test wikimovies-test simple-comline-test smokers-test

# tests that take a few minutes to run
# run 'make prepare-for-long-tests' in a fresh install before you run these
long-tests: benchmark-test wnet-test cora-test grid-test

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
# individual longer tests, which should take about 1min each
#

cora-test:
ifneq ($(DATASETS),)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/cora/; make clean; make; make check; make clean; make unittest)
else
	$(error $(NOTIFY))
endif

grid-test:
	(cd ../; PYTHONPATH=`pwd`; cd datasets/grid/; make clean; make; make check; make clean; make unittest)

wikimovies-test:
	(cd ../datasets/wikimovies/; make clean; make unittest)

smokers-test:
	(cd ../datasets/smokers/; make clean; make unittest)

simple-comline-test:
	python simple.py experiment --db test-data/textcattoy.cfacts --prog test-data/textcat.ppr --train test-data/toytrain.exam  --test test-data/toytest.exam


# not converted yet....
wnet-test:
ifneq ($(DATASETS),)
	(cd ../; PYTHONPATH=`pwd`; cd $(DATASETS)/wordnet/; make clean; make; make check)
else
	$(error $(NOTIFY))
endif

benchmark-test:
	PYTHONPATH=../; python benchmark.py

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
