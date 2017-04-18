import sys
import collections
import logging
import random

# This code converts data from the WikiMovies dataset, distributed by
# Facebook https://research.fb.com/downloads/babi/, into a
# Tensorlog-readable format.  It expects to be run in a directory that
# contains the untarred WikiMovies dataset as a subdirectory named
# 'movieqa' (which is what their tarball unpacks to by default),
# expects a subdirectory named tlog-format to hold the generated
# files.
#
# The questions are stored in the files
# ./movieqa/questions/wiki_entities/wiki_entities_qa_(dev|train|test).txt:
# The format is "1 question? TAB ans1, ..."  For example,
# +------------------------------------------------------------
# |1 what language is Turistas in?	English, Portuguese
# |1 what is the genre for the film He Who Gets Slapped?	Drama, Thriller, Romance
# |1 what is the primary language in the film Hamlet?	English
# |...
#
# The knowledge base is stored in
# ./movieqa/knowledge_source/wiki_entities and format is like this:
# +------------------------------------------------------------
# |1 Kismet directed_by William Dieterle
# |2 Kismet written_by Edward Knoblock
# |3 Kismet starred_actors Marlene Dietrich, Edward Arnold, Ronald Colman, James Craig
# |4 Kismet release_year 1944
# |5 Kismet in_language English
# |6 Kismet has_tags bd-r
# |7 Kismet has_plot Hafiz, a rascally beggar on the periphery of the court of Baghdad, schemes [...]
# |
# |1 Flags of Our Fathers directed_by Clint Eastwood
# |2 Flags of Our Fathers written_by Paul Haggis, Ron Powers, James Bradley
#
# All lines are space separated as "movie title" "relation_type"
# "object" except plots seem to have commas in them. (But plots don't
# seem to be used anyway.)
#
# Extraction and converstion is messy because entities and commas in
# them (eg "East Side, West Side") and multiple slot fillers in the KB
# and multiple answers in the question files are only
# comma-separated.
#
#
# There are also many systematic errors in the answers, and some of
# them this code will fix: ERRORS[wrong_name] is the correct KB name
# for an entity.

ERRORS = {'Dance , Girl':'Dance , Girl , Dance',
          'Reuben':'Reuben , Reuben',
          'Rachel':'Rachel , Rachel',
          'Walking':'Walking, Walking',
          'It\'s a Mad , Mad , Mad World':'It\'s a Mad , Mad , Mad , Mad , World',
          '99 and 44/100% Dead':'99 and 44/100 Dead',
          'King of the Monsters!':'Godzilla, King of the Monsters!',
          'Good Morning , Miss Dove':'Good Morning, Miss Dove',
          'Fight , Zatoichi': 'Fight , Zatoichi , Fight',
          'Corrina': 'Corrina , Corrina',
          'Olivier , Olivier': 'Olivier',
          'Puff , Puff , Pass': 'Puff , Pass',
          'Run , Man , Run': 'Run , Man',
          'Oh , God! , You Devil': 'God! , You Devil',
          'My Love , Arise': 'Arise , My Love',
          'You\'re a Good Man , Charlie Brown': 'You\'re a Good Man'
          }

# RELS is all the relations used in the KB
RELS = "directed_by has_genre has_imdb_rating has_imdb_votes has_plot has_tags release_year starred_actors written_by in_language".split(" ")
DIGITS = set("0123456789")
STOP_WORDS = set("a the , to of and in his is with ... an on for he who by her when from their that as but has at are him be it this where have".split(" "))

def simple_tokenize(sent):
  """ Tokenize but add spaces around the commmas."""
  comma_sep_parts = map(lambda p:p.strip(), sent.split(","))
  return " , ".join(comma_sep_parts).split(" ")

class EntityMatcher(object):
  """ Used to find entities from a KB in questions.
  """

  def __init__(self,kb):
    self.index = collections.defaultdict(set)
    self.word_set = set()
    self.name_set = set()
    self.absorb_kb(kb)

  def absorb_kb(self,kb):
    """Add all the entities in the KB.
    """
    for rel in kb:
      for sub in kb[rel]:
        self.add(sub)
        if rel!="has_plot":
          for obj in kb[rel][sub]:
            self.add(obj)
        else:
          self.word_set.update(kb[rel][sub])

  def add(self,name):
    """ Add a single entity name.
    """
    self.name_set.add(name)
    tokens = simple_tokenize(name)
    # self.index is sort of a brain-damaged trie that makes matching
    # token sequences a bit faster.  for an entity w0....wk,
    # index[word0] = (w1,....,wk)
    suffix = tuple(tokens[1:])
    self.index[tokens[0]].add(suffix)

  def longest_match_at(self,i,words):
    """If words[i:i+k] matches an entity name (w0...wk)
    then return the 'suffix' of the match, ie (w1...wk).
    If there is no match return None.
    """
    longest_match = None
    if words[i] in self.index:
      for suffix in self.index[words[i]]:
        matches = True
        for j in range(len(suffix)):
          if len(words) <= i+1+j:
            matches = False
          elif suffix[j]!=words[i+1+j]:
            matches = False
        if matches and ((not longest_match) or len(longest_match)<len(suffix)):
          longest_match = suffix
    return longest_match

  def as_entity_list(self,sent,keep_nonentities=False):
    """When keep_nonentities==False, then given a tokenized
    comma-separated list of entity names, return the entity names from
    the list (discarding anything that's not an entity).  When
    keep_nonentities==True, then tokens that are not entities are
    retained, so what is returned is a version of the sentence where
    entity names have been converted to chunks.
    """
    words = simple_tokenize(sent)
    result = []
    i = 0
    while i<len(words):
      m = self.longest_match_at(i,words)
      if m is None and words[i]==',':
        i += 1
      elif m is None:
        if keep_nonentities:
          result.append(words[i])
        else:
          pass
          #logging.warn('unmatched token %r in %r discarded' % (words[i],sent))
        i += 1
      else:
        result.append(" ".join([words[i]] + list(m)))
        i += len(m)+1
    return result

def trim_digits(line):
  """Remove leading digit-space combinations from a line of text.
  """
  assert line[0] in DIGITS
  assert line[1]==' '
  return line[2:]

def load_kb():
  """Return a KB, which is a nested dictionary so that kb[rel][entity1]
  is mapped to the set of entities entity2 such that
  rel(entity1,entity2) is true.
  """
  kb = collections.defaultdict(lambda:collections.defaultdict(set))
  for line in open('movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt'):
    line = line.strip()
    if len(line)>0:
      line = trim_digits(line)
      for rel in RELS:
        k = line.find(rel)
        if k>=0:
          sub = line[:k-1]
          sub = " ".join(simple_tokenize(sub))
          assert len(sub)>0, "empty movie in line %r" % line
          obj = line[k+len(rel)+1:]
          assert len(obj)>0, "empty object part in line %r" % line
          if rel=='has_plot':
            objs = set(simple_tokenize(obj.lower())) - STOP_WORDS
          else:
            objs = obj.split(", ")
          for o in objs:
            assert len(obj)>0, "empty object in object part %r in line %r" % (objs,line)
            kb[rel][sub].add(o)
  # Duplicate some things that are named wrong in the answers files,
  # i.e. give them aliases that match the incorrect versions of these
  # entities.
  for rel in RELS:
    for sub in ERRORS:
      kb[rel][ERRORS[sub]] = kb[rel][sub]
  print 'loaded kb'
  return kb

def write_kb(kb,filename,subject_type,extra_declares=[]):
  """Write a KB out in .cfacts format.  Assumes the KB is as constructed
  by load_kb, and the type of the first argument of every relation is
  of the type defined by subject_type.  This also outputs type
  declarations but the only types are entities and words.
  """
  with open(filename,'w') as fp:
    for rel in kb:
      if rel=='has_plot' or rel=="has_feature":
        obj_type = 'word'
      else:
        obj_type = 'entity'
      fp.write('# :- %s(%s,%s_t)\n' % (rel,subject_type,obj_type))
    for decl in extra_declares:
      fp.write('# :- %s\n' % decl)
    for rel in kb:
      for sub in kb[rel]:
        for obj in kb[rel][sub]:
          fp.write('\t'.join([rel,sub,obj]) + '\n')

def write_exams(em,qas,stem,filename,n=None):
  """Write examples, and also return a second KB-like structure ex_kb,
  which contains the facts that describe the examples, namely the
  'mentions_entity' and 'has_feature' relations.  Em is an
  EntityMatcher, qas is a list of (question,answer_list) pairs,
  filename is the output file, and n is an upper bound on how many
  things to write out.
  """
  if (n is not None) and (n<len(qas)): qas = random.sample(qas,n)
  ex_kb = collections.defaultdict(lambda:collections.defaultdict(set))
  with open(filename,'w') as fp:
    for i,(question,answers) in enumerate(qas):
      if len(answers)==0:
        logging.warn('no answers for question %r' % question)
      else:
        qid = 'q%s%04d' % (stem,i)
        toks = simple_tokenize(question)
        entity_tokens = em.as_entity_list(question,keep_nonentities=False)
        for e in entity_tokens:
          ex_kb['mentions_entity'][qid].add(e)
        for w in toks:
          ex_kb['has_feature'][qid].add(w)
        fp.write('# ' + question + ' ' + repr(answers) + '\n')
        fp.write('\t'.join(['answer',qid]+answers) + '\n')
  return ex_kb

def load_questions(stem,em):
  """Load questions into a list of question,answer_list pairs.  The
  trailing ? from each question is discarded to simplify tokenization.
  """
  result = []
  for line in open('movieqa/questions/wiki_entities/wiki-entities_qa_%s.txt' % stem):
    line = trim_digits(line.strip())
    question,answer_part = line.split("\t")
    if question[-1]=="?": question = question[:-1]
    else: logging.warn('weird question %r' % question)
    answers = em.as_entity_list(answer_part, keep_nonentities=False)
    result.append((question,answers))
  print 'loaded',len(result),'qas'
  return result

def check_answers(qas,em):
  """Code I used to find erroneous answers (currently unused).
  """
  tot = missing = 0.0
  missing_set = set()
  for (question,answers) in qas:
    for a in answers:
      tot += 1
      if a not in em.name_set:
        missing += 1
        missing_set.add(a)
        print 'missing %r' % a,'in',question,answers
  print 'tot',tot,'missing',missing
  return missing_set

if __name__ == "__main__":
  # Decide what size dataset to create
  n=1000 if len(sys.argv)<2 else int(sys.argv[1])
  # Load the KB
  kb = load_kb()
  # Write the KB
  write_kb(kb,'tlog-format/kb-core.cfacts','entity_t')
  # Create the entity matcher, which we need for generating the
  # mentions_entity data and parsing the comma-separated answers
  em = EntityMatcher(kb)
  # Echo some status info
  print len(em.word_set),'plot words'
  print len(em.name_set),'entity names'
  # Write out the examples
  for s in "dev", "train", "test":
    print 'writing',s,'examples....'
    qas = load_questions(s,em)
    ex_kb = write_exams(em,qas,s,'tlog-format/%s-%d.exam' % (s,n), n=n)
    extra_declares = ['answer(question_t,entity_t)']
    write_kb(ex_kb,'tlog-format/kb-%s-%d.cfacts' % (s,n), 'question_t',extra_declares=extra_declares)
