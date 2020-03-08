"""
task-related class definitions as follows:
EntityMention, RelationMention, EventMention etc.
"""

from ie_utils import *


def get_single_ner_label(label_schema='IO', idx=0, etype=None, sno=0, eno=0):
    """
    get an individual sequence label for an entity mention
    :param label_schema: four labeling schema are available, 'IO', 'BIO', 'BIEO', 'SBIEO'
    :param idx: the curremt token position
    :param etype: entity type
    :param sno: starting position
    :param eno: ending position
    :return: the target label
    """
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

def get_entity_mentions_from_labels(plabels):
    """
    # get entity mentions from labels
    :param plabels: list of labels
    :return: list of entity mentions [eno, spos, len, type]
    """
    ltype, lpos, eno = ['O', -1, 1]  # the last type, position
    ems = []    # list of entity mentions
    for i, tag in enumerate(plabels):
        type = 'O'
        if '-' in tag:  tag, type = tag.split('-')  # sometimes PAD will appear
        # possible mention end, the current tag is SBO or the previous tag is SE or types are different
        if (tag in 'SBO' or type != ltype or (i > 0 and plabels[i-1][0] in 'SE')):
            if ltype != 'O' and lpos >= 0:   # new entity mention with valid type and position
                ems.append([eno, lpos, i, ltype])
                eno += 1
            lpos = i  # for SB or I with new type, a new entity mention begins
        ltype = type  # type is continuous for entity mention
    # for the last entity in the sentence
    if ltype != 'O' and lpos >= 0:
        ems.append([eno, lpos, len(plabels), ltype])
    return ems


def match_gold_pred_entity_mentions(inst):
    """
    match positions between gold and recognized entity mentions, can be optimized,
    if matched, set gpno for each other, and emid for remlist
    :param inst:
    :return:
    """
    # clear old gpno (gold/predicted index no)
    for em in inst.emlist:  em.gpno = -1
    for rem in inst.remlist:  rem.gpno = -1
    #
    emno = 0
    for i, em in enumerate(inst.emlist):
        if int(em.id[1:]) > emno:  emno = int(em.id[1:])
        for j, rem in enumerate(inst.remlist):
            # positions exactly match
            if em.hsno == rem.hsno and em.heno == rem.heno and rem.gpno < 0:
                em.gpno, rem.gpno, rem.id = j, i, em.id
                break
    # refresh emids in the remlist
    for rem in inst.remlist:
        if rem.gpno < 0:
            emno += 1
            rem.id = 'T{}'.format(emno)
    return

# collect confusion matrix for instance or document
def collect_ner_confusion_matrix(inst, confusions, etypedict):
    # collect true positives and false negatives
    for em in inst.emlist:
        gno = etypedict[em.type]
        pno = -1    # Entity --> O
        if em.gpno >= 0:    # matched
            pno = etypedict[inst.remlist[em.gpno].type]
        confusions[gno][pno] += 1
    # collect false positives
    for rem in inst.remlist:
        if rem.gpno < 0 and rem.hsno >= 0 and rem.heno >= 0:  # O --> valid BERT entity
            pno = etypedict[rem.type]
            confusions[-1][pno] += 1
            if rem.hsno < 0 or rem.heno < 0:    # for Bert fragments
                print(rem)
    return


class Entity(object):
    def __init__(self,
                 id=None,
                 type=None,
                 stype=None,
                 # linkdb=None,
                 linkids=None,
                 linkname=None
                 ):
        self.id = id
        self.type = type
        self.stype = stype
        #self.linkdb = linkdb
        self.linkids = linkids
        self.linkname = linkname

    def __str__(self):
        return '{} {} {} {}'.format(self.id, self.type, '|'.join(self.linkids), self.linkname)

class EntityMention(object):
    # entity mention class
    def __init__(self,
                 did = None,    # doc id
                 id = None,
                 type = None,
                 stype = None,
                 eclass = None, # mention class
                 level = None,  # mention level
                 name = None,
                 linkids = None,   # list of linkids
                 visible = True,   # True for placeholder in the sentence, False to nested/duplicated entities
                 lineno = -1,
                 hsno = -1,     # head start
                 heno = -1,     # head end
                 esno = -1,     # extension start
                 eeno = -1,     # extension end
                 gpno = -1      # gold-pred reciprocal
                 ):
        self.did = did
        self.id = id
        self.type = type
        self.stype = stype
        self.eclass = eclass
        self.level = level
        self.name = name
        # self.linkdb = linkdb
        self.linkids = ['None'] if not linkids else linkids  # list of linkids
        self.plinkids = []      # list of predicted linkids
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
        return '{}|{} {}|{}|{}|{}|{}|{} {}'.format(self.did, self.id, self.type, self.stype, '|'.join(self.linkids), self.visible, sname, self.gpno, spos)

    def __lt__(self, other):
        return self.lineno < other.lineno or self.hsno < other.hsno or (self.hsno == other.hsno and self.heno <= other.heno)

    def set_around_delimiters(self, words, sep='#', recover=False):
        if self.heno - self.hsno == 1:
            ename = self.name.replace('_', ' ') if recover else words[self.hsno]
            words[self.hsno] = '{} {} {}'.format(sep, ename, sep)
        else:
            words[self.hsno] = '{} {}'.format(sep, words[self.hsno])
            words[self.heno-1] = '{} {}'.format(words[self.heno-1], sep)

    # mark entity mentions in sequence
    def mark_sequence_entity_mentions(self, words):
        if self.hsno < 0 or self.heno < 0:  return
        words[self.hsno] = '{}{}'.format('{', words[self.hsno])
        words[self.heno - 1] = '{}{}{}'.format(words[self.heno - 1], '}-', self.type[:3])

class Token(object):
    # class for a token in a sentence
    def __init__(self,
                 no = -1,
                 text = '',
                 offsets = None,
                 POS = '',
                 chunk = '',
                 dep = '',
                 label = 'O'    # default label
                 ):
        self.no = no
        self.text = text
        self.POS = POS
        self.chunk = chunk
        self.dep = dep
        # other features
        self.label = label
        self.plabel = 'O'           # predicted label
        self.offsets = offsets if offsets else [-1, -1]      # token --> char positions in the doc text
        self.boffsets = [-1, -1]    # token --> word-piece positions in the bert sequence

    def __str__(self):
        return '{}|{}|{}-{}|{}-{}'.format(self.no, self.text, self.offsets[0], self.offsets[1],
                                          self.boffsets[0], self.boffsets[1])
    def __copy__(self):
        token = Token()
        for att in self.__dict__.keys(): token.__dict__[att] = self.__dict__[att]
        return token

class Tokens(object):
    def __init__(self, words):
        self.tokens = [Token(no=i, text=word) for i, word in enumerate(words)]

    def __len__(self):  return len(self.tokens)

    def __getitem__(self, idx):  return self.tokens[idx]

    def __str__(self):  return '\n'.join([t.__str__() for t in self.tokens])

    def __copy__(self):
        tokens = Tokens([])
        tokens.tokens = [token.__copy__() for token in self.tokens]
        return tokens

    @property
    def words(self):  return [t.text for t in self.tokens]

    @words.setter
    def words(self, words):
        for i, word in enumerate(words):
            self.tokens[i].text = word

    @property
    def POSes(self):  return [t.POS for t in self.tokens]

    @POSes.setter
    def POSes(self, POSes):
        for i, POS in enumerate(POSes):
            self.tokens[i].POS = POS

    @property
    def chunks(self):  return [t.chunk for t in self.tokens]

    @chunks.setter
    def chunks(self, chunks):
        for i, chunk in enumerate(chunks):
            self.tokens[i].chunk = chunk

    @property
    def offsets(self):  return [t.offsets for t in self.tokens]

    @property
    def labels(self):  return [t.label for t in self.tokens]

    @property
    def plabels(self):  return [t.plabel for t in self.tokens]

    @plabels.setter
    def plabels(self, plabels):  # update predicted labels for Tokens
        for i in range(min(len(plabels), len(self.plabels))):
            self.tokens[i].plabel = plabels[i]

    @property
    def boffsets(self):  return [t.boffsets for t in self.tokens]

    def insert(self, index, word):  # disregard other fields
        self.tokens.insert(index, Token(text=word))

    def convert_case_num(self, tcfg):    #
        for i, token in enumerate(self.tokens):
            text = None
            if is_float(token.text):  text = '[NUM]'
            elif not (tcfg.case_sensitive or token.text[:4] in tcfg.bld_ent_types):
                text = token.text.lower()
            if text:  self.tokens[i].text = text
        return

    def blind_entity_mention_placeholders(self, bld_ent_mode=0):
        """
        blind the entity token text in the tokens
        :param bld_ent_mode:
        :return:
        """
        edict = {}
        for i, word in enumerate(self.words):
            etype, emid = is_bio_entity(word)
            if not emid: continue  # not an entity mention
            # increase entity sequence no like GENE_1, CHEM_1, DISE_1
            if bld_ent_mode == 0:  # 0-entity type, 1-type_seq, 2-entity id
                self.tokens[i].text = etype
            elif bld_ent_mode == 1:
                if etype not in edict:  edict[etype] = 1
                eno = edict[etype]
                edict[etype] += 1
                self.tokens[i].text = '{}_{}'.format(etype, eno)
            #if ENT_BRACKET_ID: words[i] = '[' + words[i] + ']'
        return

# class for sequence labeling
class SequenceLabel(object):
    def __init__(self,
                 id = None,
                 no = None,
                 emlist=None,
                 tokens = None      # text and offsets are ready
                 ):
        self.id = id
        self.no = no
        self.emlist = emlist
        self.remlist = []
        #
        self.tokens = tokens
        self.btokens = Tokens([])

    def __str__(self):
        sidtext = '\nID: {}-{}\nTEXT: {}'.format(self.id, self.no, self.text)
        swords = '' if not self.words else '\nWORDS: {}'.format(' '.join(self.words))
        sbwords = '' if not self.btokens.words else '\nbWORDS: {}'.format(' '.join(self.btokens.words))
        slabels = '' if not self.labels else '\nLABELS: {}'.format(' '.join(self.labels))
        sblabels = '' if not self.blabels else '\nbLABELS: {}'.format(' '.join(self.blabels))
        elist = '\n'.join(['{}'.format(em) for em in self.emlist])
        selist = '\nGOLD ENTITIES:\n{}'.format(elist) if len(elist) > 0 else ''
        relist = '\n'.join(['{}'.format(em) for em in self.remlist])
        srelist = '\nRECOGNIZED ENTITIES:\n{}'.format(relist) if len(relist) > 0 else ''
        return '{}{}{}{}{}{}{}'.format(sidtext, swords, sbwords, slabels, sblabels, selist, srelist)

    @property
    def text(self):  return ' '.join(self.words)

    @property
    def words(self): return self.tokens.words

    @property
    def offsets(self): return self.tokens.offsets

    @property
    def boffsets(self): return self.tokens.boffsets

    @property
    def labels(self): return self.tokens.labels

    @property
    def blabels(self): return self.btokens.labels

    def get_tokens(self, bertID=False):  return (self.btokens if bertID else self.tokens)

    def generate_wordpiece_bert_sequence(self, tokenizer):
        """
        generate bert sequence of word pieces like [CLS] ... [SEP] in btokens
        align words to word pieces via boffsets
        :param tokenizer:
        :return:
        """
        words = self.tokens.words  # case-insensitive ?
        bwords = tokenizer.tokenize(' '.join(words))  # word pieces
        self.btokens = Tokens(bwords)
        #
        j, li, plen = 0, 1, 0  # first position is 1 [CLS]
        for i in range(1, len(bwords) - 1):  # exclude [CLS] and [SEP]
            pieceID = bwords[i].startswith('##')  # word piece id
            blen = len(bwords[i][2:] if pieceID else bwords[i])
            if len(words[j]) != blen:   # multiple word pieces
                plen += blen
                if len(words[j]) == plen:  # a word is complete
                    self.tokens[j].boffsets = [li, i + 1]
                    j, li, plen = j + 1, i + 1, 0
            else:  # a single word matches
                self.tokens[j].boffsets = [i, i + 1]
                j, li = j + 1, i + 1
        return

    # modify esno, eeno in terms of bert sequence
    def locate_entity_mention_in_bert_sequence(self, emlist):
        for em in emlist:
            if em.hsno >= len(self.boffsets):
                print(self)
                print(em)
                for i, pair in enumerate(self.boffsets):
                    print(i, self.words[i], pair, self.words[pair[0]:pair[1]])
            em.esno = self.boffsets[em.hsno][0]    # starting location
            em.eeno = self.boffsets[em.heno-1][1]  # ending location
        return

    # prepare instance features and labels
    def generate_instance_feature_label(self, tcfg, Doc):
        if tcfg.bertID:
            # set tokens.boffsets and btokens
            self.generate_wordpiece_bert_sequence(tcfg.tokenizer)
            # set esno, eeno as the start and end positions in the bert sequence
            self.locate_entity_mention_in_bert_sequence(self.emlist)
        else:
            # make word sequence for non-BERT model
            self.generate_word_sequence(tcfg)
        # set labels according to entity positions in the word/bert sequence
        self.generate_sequence_labels(self.emlist,tcfg.bertID, Doc.elabel_schema)
        return

    def generate_word_sequence(self, tcfg):
        self.tokens.convert_case_num(tcfg)

    def generate_sequence_labels(self, emlist, bertID=False, schema='IO'):
        for em in emlist:
            sno = em.esno if bertID else em.hsno
            eno = em.eeno if bertID else em.heno
            for i in range(sno, eno):
                label = get_single_ner_label(schema, idx=i, etype=em.type, sno=sno, eno=eno)
                self.get_tokens(bertID)[i].label = label
        return

    def get_instance_feature_label(self, tcfg):
        bertID = tcfg.bertID
        return [self.get_tokens(bertID).words, self.get_tokens(bertID).labels]

    def recognize_entity_mentions_from_labels(self, lineno=0, bertID=False):
        # clear old entity mentions
        for em in self.emlist:   em.gpno = -1  # gold-predicted entry no
        self.remlist = []
        #
        words = self.get_tokens(bertID).words
        plabels = self.get_tokens(bertID).plabels
        ems = get_entity_mentions_from_labels(plabels)
        for eno, spos, epos, type in ems:
            emid = 'T{}'.format(eno)
            name = '_'.join(words[spos:epos])  # should map to self.words
            if bertID:  # the sequencelabel's id is the doc's id
                em = EntityMention(did=self.id, id=emid, type=type, name=name, lineno=lineno, esno=spos, eeno=epos)
            else:
                em = EntityMention(did=self.id, id=emid, type=type, name=name, lineno=lineno, hsno=spos, heno=epos)
            self.remlist.append(em)
        return

    def convert_bert_entity_mention_positions(self):
        """
        exactly match the starting and ending positions
        :return: esno, eeno --> hsno, heno
        """
        errID = False
        for rem in self.remlist:
            rem.visible = True
            for i, boffset in enumerate(self.boffsets):
                if rem.hsno < 0 and boffset[0] == rem.esno:
                    rem.hsno = i
                if rem.heno < 0 and boffset[1] == rem.eeno:
                    rem.heno = i+1
                if rem.hsno >= 0 and rem.heno >= 0:  break
            # there are cases when an entity begins in the middle of a word, or ends in the middle of a word.
            if not (rem.hsno >= 0 and rem.heno >= 0):
                rem.hsno = rem.heno = -1
                rem.visible = False
                errID = True
        self.remlist = [rem for rem in self.remlist if rem.visible]
        return

    # match positions between gold and recognized, can be optimized
    def match_gold_pred_instances(self):
        match_gold_pred_entity_mentions(self)

    def collect_ner_confusion_matrix(self, confusions, etypedict):
        collect_ner_confusion_matrix(self, confusions, etypedict)

    def get_ner_labeled_sentences(self, rstlines, verbose=0):
        if verbose < 2:  return  # valid for verbose>=2
        #gwords, rwords = self.words[:], self.words[:]
        gwords, rwords = self.words, self.words
        for em in self.emlist:
            em.mark_sequence_entity_mentions(gwords)
        for rem in self.remlist:
            rem.mark_sequence_entity_mentions(rwords)
        if self.emlist or self.remlist:
            if verbose >= 3 or gwords != rwords:  # 2-erroneous instances, 3-all instances
                rstlines.extend(['{}-{}'.format(self.id, self.no), ' '.join(gwords), ' '.join(rwords)])
        return

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
    def generate_instance_feature_label(self, tcfg, Doc):
        super(segSequenceLabel,self).generate_instance_feature_label(tcfg, Doc)
        if tcfg.bertID:
            self.locate_entity_mention_in_bert_sequence(self.smlist)
        self.slabels = self.generate_sequence_labels(self.smlist, tcfg.bertID, Doc.slabel_schema)
        return

    def output_instance_candidate(self, bertID=False):
        return [(self.bwords if bertID else self.words), self.labels, self.slabels]


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

