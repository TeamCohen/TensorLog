
RULESET:=google-nonrecursive

PROGRAM:=google-fact.${RULESET}-sample.cfacts:${RULESET}.sample.ppr

VPATH=../../../src

all: tensorlog.py
	python -i $< --programFiles ${PROGRAM} --proppr --debug
