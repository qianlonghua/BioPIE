"""
task-related class definitions as follows:
EntityMention, RelationMention, EventMention etc.
"""

from ie_utils import *

def relation_type(type, stype=None, rvsid=False):
    rtype = type if type is not None else 'None'
    if stype is not None: rtype += '.' + stype
    if rvsid: rtype += '.R'
    return rtype

#
def get_single_ner_label(label_schema='IO', idx=0, etype=None, sno=0, eno=0):
    label = 'O'
    if label_schema == 'IO':
        label = 'I-{}'.format(etype)
    elif label_schema == 'BIO':
        label = '{}-{}'.format('B' if idx == sno else 'I', etype)
    elif label_schema == 'BIEO':
        label = 'I'
        if idx == sno:
            label = 'B'
        elif idx == eno - 1:
            label = 'E'
        label = '{}-{}'.format(label, etype)
    elif label_schema == 'SBIEO':
        label = 'I'
        if idx == sno == eno - 1:
            label = 'S'
        elif idx == sno:
            label = 'B'
        elif idx == eno - 1:
            label = 'E'
        label = '{}-{}'.format(label, etype)
    return label

# get entity mentions [spos, epos, type] from a sequence of labels
def get_entity_mentions_from_labels(plabels):
    ltype, lpos, eno = ['O', -1, 1]  # the last type, position
    ems = []
    for j, tag in enumerate(plabels):
        type = 'O'
        if tag != 'O':  tag, type = tag.split('-')
        if (tag in 'SBO' or type != ltype):  # possible mention end
            if ltype != 'O' and lpos >= 0:   # new entity mention
                ems.append([eno, lpos, j, ltype])
                eno += 1
            lpos = j
        ltype = type
    # for the last entity in the sentence
    if ltype != 'O' and lpos >= 0:
        ems.append([eno, lpos, len(plabels), ltype])
    return ems

# between positions between gold and recognized, can be optimized
def match_gold_pred_entity_mentions(inst):
    # clear old gpno (gold/predicted index no)
    for em in inst.emlist:  em.gpno = -1
    for rem in inst.remlist:  rem.gpno = -1
    #
    for i, em in enumerate(inst.emlist):
        for j, rem in enumerate(inst.remlist):
            # positions exactly match
            if em.hsno == rem.hsno and em.heno == rem.heno and rem.gpno < 0:
                em.gpno, rem.gpno = j, i
                break
    return


def set_ner_labeled_words(words, em):
    if em.hsno < 0 or em.heno < 0:
        return
    words[em.hsno] = '{}{}'.format('{', words[em.hsno])
    words[em.heno - 1] = '{}{}{}'.format(words[em.heno - 1], '}-', em.type[:3])


def get_ner_labeled_sentences(inst, rstlines, verbose=0):
    gwords, rwords = inst.words[:], inst.words[:]
    for em in inst.emlist:
        set_ner_labeled_words(gwords, em)
    for rem in inst.remlist:
        set_ner_labeled_words(rwords, rem)
    if inst.emlist or inst.remlist:
        if verbose < 4 and gwords == rwords:     # output True Positive for verbose >= 4
            return
        rstlines.extend(['{}-{}'.format(inst.id, inst.no), ' '.join(gwords), ' '.join(rwords)])
    return


def collect_entity_statistics(inst, counts, etypedict):
    # collect true positives
    for em in inst.emlist:
        gno = etypedict[em.type]
        counts[gno] += 1
    return


class EntityMention(object):
    # entity mention class
    def __init__(self,
                 id = None,
                 type = None,
                 stype = None,
                 eclass = None, # mention class
                 level = None,  # mention level
                 name = None,
                 linkdb = None,
                 linkid = None,
                 visible = True,   # True for placeholder in the sentence, False to nested/duplicated entities
                 lineno = -1,
                 hsno = -1,     # head start
                 heno = -1,     # head end
                 esno = -1,     # extension start
                 eeno = -1,     # extension end
                 gpno = -1      # gold-pred reciprocal
                 ):
        self.id = id
        self.type = type
        self.stype = stype
        self.eclass = eclass
        self.level = level
        self.name = name
        self.linkdb = linkdb
        self.linkid = linkid
        self.visible = visible
        #
        self.lineno = lineno
        self.hsno = hsno
        self.heno = heno
        self.esno = esno
        self.eeno = eeno
        # gold-recognized no
        self.gpno = gpno

    def __copy__(self, lineno=None, hsno=None, heno=None):
        em = EntityMention()
        for att in self.__dict__.keys(): em.__dict__[att] = self.__dict__[att]
        if lineno is not None:  em.lineno = lineno
        if hsno is not None:  em.hsno = hsno
        if heno is not None:  em.heno = heno
        return em

    def __str__(self):
        #attrs = ['{}:{}'.format(key, self.__dict__[key]) for key in self.__dict__.keys() if self.__dict__[key] is not None]
        #return ' '.join(attrs)
        sname = self.name.replace(' ', '_')
        spos = '{} {}|{}|{}|{}'.format(self.lineno, self.hsno, self.heno, self.esno, self.eeno)
        return '{} {}|{}|{}|{}|{}|{}|{} {}'.format(self.id, self.type, self.stype, self.linkdb, self.linkid, self.visible, sname, self.gpno, spos)

    def __lt__(self, other):
        return self.lineno < other.lineno or self.hsno < other.hsno or (self.hsno == other.hsno and self.heno <= other.heno)

    def set_around_delimiters(self, words, sep='#', recover=False):
        if self.heno - self.hsno == 1:
            ename = self.name.replace('_', ' ') if recover else words[self.hsno]
            words[self.hsno] = '{} {} {}'.format(sep, ename, sep)
        else:
            words[self.hsno] = '{} {}'.format(sep, words[self.hsno])
            words[self.heno-1] = '{} {}'.format(words[self.heno-1], sep)


# class for sequence labeling
class SequenceLabel(object):
    def __init__(self,
                 id = None,
                 no = None,
                 text = '',
                 emlist=None,
                 offsets = None
                 ):
        self.id = id
        self.no = no
        self.text = text
        self.emlist = emlist
        self.remlist = []
        #
        self.words = []
        self.offsets = offsets
        self.labels = []
        self.bwords = []    # for BERT
        self.boffsets = []  # for BERT

    def __str__(self):
        sidtext = '\nID: {}-{}\nTEXT: {}'.format(self.id, self.no, self.text)
        swords = '' if not self.words else '\nWORDS: {}'.format(' '.join(self.words))
        sbwords = '' if not self.bwords else '\nbWORDS: {}'.format(' '.join(self.bwords))
        slabels = '' if not self.labels else '\nLABELS: {}'.format(' '.join(self.labels))
        elist = '\n'.join(['{}'.format(em) for em in self.emlist])
        selist = '\nGOLD ENTITIES:\n{}'.format(elist) if len(elist) > 0 else ''
        relist = '\n'.join(['{}'.format(em) for em in self.remlist])
        srelist = '\nRECOGNIZED ENTITIES:\n{}'.format(relist) if len(relist) > 0 else ''
        return '{}{}{}{}{}{}'.format(sidtext, swords, sbwords, slabels, selist, srelist)

    # generate bert sequence of word pieces like [CLS] ... [SEP]
    # align words to word pieces via boffsets
    def generate_wordpiece_bert_sequence(self, tokenizer):
        words = self.text.split()  # case-insensitive ?
        bwords = tokenizer.tokenize(self.text)  # automatically to lowercase
        j, li, plen = 0, 1, 0  # first position is 1 [CLS]
        boffsets = []
        for i in range(1, len(bwords) - 1):  # exclude [CLS] and [SEP]
            pieceID = bwords[i].startswith('##')  # word piece id
            blen = len(bwords[i][2:] if pieceID else bwords[i])
            if len(words[j]) != blen:
                plen += blen
                if len(words[j]) == plen:  # a word is complete
                    boffsets.append([li, i + 1])
                    j, li, plen = j + 1, i + 1, 0
            else:  # a single word matches
                boffsets.append([i, i + 1])
                j, li = j + 1, i + 1
        self.words, self.bwords, self.boffsets = words, bwords, boffsets
        return

    # modify esno, eeno in terms of bert sequence
    def locate_entity_mention_in_bert_sequence(self, emlist):
        for em in emlist:
            if em.hsno >= len(self.boffsets):
                print(self)
                print(em)
                for i, pair in enumerate(self.boffsets):
                    print(i, self.words[i], pair, self.bwords[pair[0]:pair[1]])
            em.esno = self.boffsets[em.hsno][0]    # starting location
            em.eeno = self.boffsets[em.heno-1][1]  # ending location
        return

    # prepare instance features and labels
    def generate_instance_features(self, bertID, tokenizer, schema='IO'):
        if bertID:
            # set words, bwords and boffsets
            self.generate_wordpiece_bert_sequence(tokenizer)
            # set esno, eeno as the start and end positions in the bert sequence
            self.locate_entity_mention_in_bert_sequence(self.emlist)
        else:
            # make word sequence
            self.generate_word_sequence()
        # set labels according to entity positions in the word/bert sequence
        self.labels = self.generate_sequence_labels(self.emlist,bertID, schema)
        return

    def generate_word_sequence(self, case_sensitive=False):
        text = self.text.lower() if not case_sensitive else self.text
        self.words = text.split(' ')
        for i, word in enumerate(self.words):
            if is_float(word):  self.words[i] = '[NUM]'
        return

    def generate_sequence_labels(self, emlist, bertID=False, schema='IO'):
        words = self.bwords if bertID else self.words
        labels = ['O'] * len(words)
        for em in emlist:
            sno = em.esno if bertID else em.hsno
            eno = em.eeno if bertID else em.heno
            for i in range(sno, eno):
                labels[i] = get_single_ner_label(schema, idx=i, etype=em.type, sno=sno, eno=eno)
        return labels

    def output_instance_candidate(self, bertID=False):
        return [(self.bwords if bertID else self.words), self.labels]

    # between positions between gold and recognized, can be optimized
    def match_gold_pred_entity_mentions(self):
        # clear old gpno
        for em in self.emlist:  em.gpno = -1
        for rem in self.remlist:  rem.gpno = -1
        #
        for i, em in enumerate(self.emlist):
            for j, rem in enumerate(self.remlist):
                # positions exactly match
                if em.hsno == rem.hsno and em.heno == rem.heno and rem.gpno < 0:
                    em.gpno, rem.gpno = j, i
                    # print(em)
                    # print(rem)
                    break
        # print(self)
        return

    def recognize_entity_mentions_from_labels(self, plabels, lineno=0, bertID=False):
        # clear old entity mentions
        for em in self.emlist:   em.gpno = -1  # gold-predicted entry no
        self.remlist = []
        #
        words = self.bwords if bertID else self.words
        ems = get_entity_mentions_from_labels(plabels)
        for eno, spos, epos, type in ems:
            id = 'T{}'.format(eno)
            name = '_'.join(words[spos:epos])  # should map to self.words
            if bertID:
                em = EntityMention(id=id, type=type, name=name, lineno=lineno, esno=spos, eeno=epos)
            else:
                em = EntityMention(id=id, type=type, name=name, lineno=lineno, hsno=spos, heno=epos)
            self.remlist.append(em)
        return

    # esno, eeno --> hsno, heno, exactly match the starting and ending position
    def convert_bert_entity_mention_positions(self):
        errID = False
        for rem in self.remlist:
            for i, boffset in enumerate(self.boffsets):
                if rem.hsno < 0 and boffset[0] == rem.esno:
                    rem.hsno = i
                if rem.heno < 0 and boffset[1] == rem.eeno:
                    rem.heno = i+1
                if rem.hsno >= 0 and rem.heno >= 0:  break
            # there are cases when an entity begins in the middle of a word, or ends in the middle of a word.
            if not (rem.hsno >= 0 and rem.heno >= 0):
                rem.hsno = rem.heno = -1
                errID = True
                # print(self)
                # for i, word in enumerate(self.words):  print(self.boffsets[i], word)
                # print(rem)
        if not errID:  return
        # print(self)
        # for rem in self.remlist:
        #     if rem.hsno < 0 or rem.heno < 0:
        #         print(rem)
        return


# class for relation extraction
class RelationMention(object):
    def __init__(self,
                 id = None,
                 type = None,
                 stype = None,
                 rvsid = False, # reverse or not
                 name = None,   # type name
                 sname = None,  # subtype name
                 emid1 = None,
                 emid2 = None,
                 text = '',
             ):
        self.id = id
        self.type = type
        self.stype = stype
        self.rvsid = rvsid
        self.name = name
        self.sname = sname

        self.emid1 = emid1
        self.emid2 = emid2        # None if unary relations
        self.text = text          # in line
        self.words = []
        self.bwords = []

        self.ptype = None         # predicted type
        self.pstype = None        # predicted stype
        self.prvsid = False       # predicted reverse id

        self.visited = False      # visited mark

    def set_predicted_type(self, ptype=None, pstype=None, prvsid=None):
        self.ptype = ptype
        self.pstype = pstype
        self.prvsid = prvsid

    def __copy__(self, text=None):
        rm = RelationMention()
        for att in self.__dict__.keys(): rm.__dict__[att] = self.__dict__[att]
        if text is not None: rm.text = text
        return rm

    def __str__(self):
        srel = '{}|{}|{}|{}|{}|{}|{}'.format(self.type, self.stype, self.rvsid, self.name, self.sname, self.ptype, self.prvsid)
        return '{} {}|{} {} "{}"'.format(self.id, self.emid1, self.emid2, srel, self.text)

    # for relation extraction
    def generate_instance_features(self, bertID=False, tokenizer=None, schema=None):
        self.words = self.text.split()
        if bertID and tokenizer: self.bwords = tokenizer.tokenize(self.text)

    # return word list and type name
    def output_instance_candidate(self, bertID=False):
        rtype = relation_type(type=self.type, rvsid=self.rvsid)
        words = self.bwords if bertID else self.words
        return [words, rtype]

    # return a sentence with two entitiy placeholders replaced by types
    def blind_relation_entity_mentions(self, em1, em2=None, bertID=False):
        words = self.text.split(' ')
        if not bertID:
            words[em1.hsno] = '# {} #'.format(em1.type[:4])
            if em2:  words[em2.hsno] = '@ {} @'.format(em2.type[:4])
        else:
            words[em1.hsno] = '[{}]'.format(em1.type[:4])
            if em2:  words[em2.hsno] = '[{}]'.format(em2.type[:4])
        self.text = ' '.join(words)
        return

    def get_re_result(self):
        gtype = relation_type(type=self.type, rvsid=self.rvsid)  # gold type
        ptype = relation_type(type=self.ptype, rvsid=self.prvsid)  # predicted type
        if gtype != 'None' or ptype != 'None':
            return [gtype, ptype, self.id, self.text]
        return None

    # insert #...# and @...@ around entity 1 and entity 2
    def insert_entity_mentions_delimiters(self, em1, em2=None, recover=False):
        words = self.text.split(' ')
        em1.set_around_delimiters(words, sep='#', recover=recover)
        if em2: # binary relations
            em2.set_around_delimiters(words, sep='@', recover=recover)
        self.text = ' '.join(words)
        return


class QuestionAnswer(object):
    def __init__(self,
                 id=None,
                 q=None,
                 ):
        self.id = id
        self.q = q
        self.ans = []
        self.rans = []

    def __str__(self):
        sa = '\t'.join(['{}'.format(a) for a in self.ans])
        sra = '\t'.join(['{}'.format(a) for a in self.rans])
        return 'Q: {}\nA: {}\nRA: {}'.format(self.q, sa, sra)

    def add_answer(self, a=None, gold=True):
        if gold:  self.ans.append(a)
        else: self.rans.append(a)

class Answer(object):
    def __init__(self, text=None, spos=-1, epos=-1):
        self.text = text
        self.spos = spos
        self.epos = epos

    def __str__(self):
        return '{}[{}:{}]'.format(self.text, self.spos, self.epos)

# usd for segmented NER
class segSequenceLabel(SequenceLabel):
    def __init__(self,
                 id = None,
                 no = None,
                 text = '',
                 emlist=None,
                 offsets = None,
                 smlist=None
                 ):
        super(segSequenceLabel, self).__init__(id, no, text, emlist, offsets)
        self.smlist = smlist
        self.slabels = []  # for segment entity mention

    def __str__(self):
        tsl = super(segSequenceLabel,self).__str__()
        slist = '\n'.join(['{}'.format(em) for em in self.smlist])
        sslist = '\nSEGMENTS:\n{}'.format(slist) if len(slist) > 0 else ''
        sslabels = '' if not self.slabels else '\nsLABELS: {}'.format(' '.join(self.slabels))
        return '{}{}{}'.format(tsl, sslist, sslabels)

    # prepare instance features and labels
    def generate_instance_features(self, bertID, tokenizer, eschema='SBIEO', sschema='IO'):
        # if bertID:
        #     # set words, bwords and boffsets
        #     self.generate_wordpiece_bert_sequence(tokenizer)
        #     # set esno, eeno as the start and end positions in the bert sequence
        #     self.locate_entity_mention_in_bert_sequence(self.emlist)
        #     self.locate_entity_mention_in_bert_sequence(self.smlist)
        # else:
        #     # make word sequence
        #     self.generate_word_sequence()
        # # set labels according to entity positions in the word/bert sequence
        # self.labels = self.generate_sequence_labels(self.emlist, bertID, eschema)
        # prepare sequence labels for emlist
        super(segSequenceLabel,self).generate_instance_features(bertID, tokenizer, eschema)
        if bertID:
            self.locate_entity_mention_in_bert_sequence(self.smlist)
        self.slabels = self.generate_sequence_labels(self.smlist, bertID, sschema)
        return

    def output_instance_candidate(self, bertID=False):
        return [(self.bwords if bertID else self.words), self.labels, self.slabels]
