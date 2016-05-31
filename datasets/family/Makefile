
VPATH= raw

all: kinship-rule.cfacts
	python family.py

kinship-rule.cfacts: kinship-train-isg.ppr
	grep -v ^# $< | \
	sed 's/.*{//;s/[)]*}.*//;s/[(,]/\t/g' | \
	sort | uniq > $@