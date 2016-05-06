#!/bin/bash -x

QUERY=concept_professionistypeofprofession
RULESET=google-recursive

echo ^${QUERY} > queries.new.txt


for t in 1 2 3 4 5 6 7 8 9 10; do
    echo "TRIAL $t"
    mv queries.new.txt queries.txt
    grep -f queries.txt ${RULESET}.ppr | \
	sed 's/^.*:-//;s/{.*//;s/[)],/)\n/g;s/ //g;' | \
	sed 's/[(].*//' | \
	grep ^concept | sed 's/^/^/' | sort >/tmp/foo
    sort -m <queries.txt </tmp/foo | uniq > queries.new.txt
    FOO=`diff -q queries.txt queries.new.txt`; 
    if [ -z "$FOO" ]; then break; fi
    break;
done

grep -f queries.txt ${RULESET}.ppr > ${RULESET}.sample.ppr
sed 's/^.*:-//;s/{.*//;s/[)],/)\n/g;s/ //g;' ${RULESET}.sample.ppr | \
    sed 's/[(].*//' | \
    grep ^fact | grep -f - google-fact.cfacts > google-fact.${RULESET}-sample.cfacts
