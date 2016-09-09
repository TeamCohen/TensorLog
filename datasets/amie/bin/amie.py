import re

STOP=re.compile("['(),]")

def sanitize(s):
    return 's_%s' % STOP.sub("",s[1:-1])
