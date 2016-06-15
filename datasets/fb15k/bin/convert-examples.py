import sys
import tensorlog
import re

def cvtExamples(fIn,fOut,prefix,targetPred):
    fp = open(fOut,'w')
    regex = re.compile('interp\((i_\w+),(\w+),(\w+)')
    for line in open(fIn):
        parts = line.strip().split("\t")
        m = regex.search(parts[0])
        pred = m.group(1)
        queryX = m.group(2)
        pos = []
        if pred==targetPred:
            for ans in parts[1:]:
                #print pred,queryX,line.strip()
                if ans[0]=='+':
                    m = regex.search(ans[1:])
                    pos.append(m.group(3))
                #print pred,queryX,pos,line.strip()
            if pos:
                for p in pos: fp.write('%s_%s\t%s\t%s\n' % (prefix,pred,queryX,p))
    print 'produced',fOut

if __name__ == "__main__":
    #for rel in ['people_x_person_x_profession','film_x_actor_x_film_x_film_x_performance_x_film','people_x_person_x_nationality','music_x_genre_x_artists']:
    for rel in ['location_x_location_x_containedby','common_x_topic_x_webpage_x_common_x_webpage_x_category',
                'location_x_administrative_division_x_country','location_x_administrative_division_x_second_level_division_of']:
        for pref in ['train','valid']:
            cvtExamples('inputs/%s.examples' % pref, 'fb-%s-%s.cfacts' % (rel,pref), pref, 'i_%s' % rel)
