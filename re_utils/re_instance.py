from ner_utils.ner_instance import *

def get_relation_label(type, stype=None, rvsid=False):
    """
    :param type:  major type
    :param stype: subtype
    :param rvsid: reverse if
    :return: relation label like 3.R
    """
    rlabel = type if type is not None else 'None'
    if stype is not None: rlabel += '.' + stype
    if rvsid: rlabel += '.R'
    return rlabel


# relational facts
class Relation(object):
    def __init__(self,
                 id=None,
                 type=None,
                 stype=None,
                 name=None,
                 sname=None,
                 eid1=None,
                 eid2=None
                 ):
        self.id = id
        self.type = type
        self.stype = stype
        self.name = name
        self.sname = sname
        self.eid1 = eid1
        self.eid2 = eid2

    def __str__(self):
        return '{} {}|{}|{}|{}'.format(self.id, self.type, self.stype, self.eid1, self.eid2)


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
                 tokens = None,
                 hsno1 = -1,
                 heno1 = -1,
                 hsno2 = -1,
                 heno2 = -1
                ):

        self.id = id
        self.type = type
        self.stype = stype
        self.rvsid = rvsid
        self.name = name
        self.sname = sname

        self.emid1 = emid1
        self.emid2 = emid2        # None if unary relations
        self.hsno1 = hsno1
        self.heno1 = heno1
        self.hsno2 = hsno2
        self.heno2 = heno2

        self.tokens = tokens.__copy__() if tokens else Tokens([])
        self.btokens = Tokens([])   # for bert sequence

        self.ptype = None         # predicted type
        self.pstype = None        # predicted stype
        self.prvsid = False       # predicted reverse id

        self.visited = False      # visited mark

    @property
    def words(self):  return self.tokens.words

    @property
    def text(self):  return ' '.join(self.words)

    def get_tokens(self, bertID=False):  return (self.btokens if bertID else self.tokens)

    def set_predicted_type(self, ptype=None, pstype=None, prvsid=None):
        self.ptype = ptype
        self.pstype = pstype
        self.prvsid = prvsid

    def __copy__(self, tokens=None, hsno1=None, heno1=None, hsno2=None, heno2=None):
        rm = RelationMention()
        for att in self.__dict__.keys(): rm.__dict__[att] = self.__dict__[att]
        # copy entity positions
        if hsno1 is not None:  rm.hsno1 = hsno1
        if heno1 is not None:  rm.heno1 = heno1
        if hsno2 is not None:  rm.hsno2 = hsno2
        if heno2 is not None:  rm.heno2 = heno2
        #
        if tokens:  rm.tokens = tokens.__copy__()
        return rm

    def __str__(self):
        srel = '{}|{}|{}|{}|{}|{}|{}'.format(self.type, self.stype, self.rvsid, self.name, self.sname, self.ptype, self.prvsid)
        return '{} {}|{} {} "{}"'.format(self.id, self.emid1, self.emid2, srel, self.text)

    # for relation extraction
    def generate_instance_feature_label(self, tcfg, Doc):
        if tcfg.bertID:
            if tcfg.tokenizer:
                words = tcfg.tokenizer.tokenize(self.text)
                self.btokens = Tokens(words)
        else:   # non-BERT models
            self.tokens.convert_case_num(tcfg)
        return

    # return word list and type name
    def get_instance_feature_label(self, tcfg):
        words = self.get_tokens(tcfg.bertID).words
        label = get_relation_label(type=self.type, rvsid=self.rvsid)
        #
        hsno1, heno1, hsno2, heno2 = self.hsno1, self.heno1, self.hsno2, self.heno2
        if hsno1 > tcfg.max_seq_len-4:  hsno1 = tcfg.max_seq_len - 4
        if hsno2 > tcfg.max_seq_len-3:  hsno2 = tcfg.max_seq_len - 3
        if heno1 > tcfg.max_seq_len-3:  heno1 = tcfg.max_seq_len - 3
        if heno2 > tcfg.max_seq_len-2:  heno2 = tcfg.max_seq_len - 2

        empos = [hsno1, heno1, hsno2, heno2, len(words)]
        return [words, label, empos]

    def get_re_labeled_instance(self, rstlines, verbose=0):
        gtype = get_relation_label(type=self.type, rvsid=self.rvsid)  # gold type
        ptype = get_relation_label(type=self.ptype, rvsid=self.prvsid)  # predicted type
        if gtype != 'None' or ptype != 'None':
            if verbose >=3 or gtype != ptype:
                rstlines.append('{}-->{}\t{}\t{}'.format(gtype, ptype, self.text, self.id))
        return

    def mark_entity_mention(self, sno, eno, etype, sep='#', mark_mode=0):
        """
        mark an entity mention in the tokens of relation mention
        :param em:
        :param sep:
        :param mark_mode:
        :return:
        """
        # if eno - sno == 1:
        #     if mark_mode == 1:  # #@ mode
        #         self.tokens[sno].text = '{} {} {}'.format(sep, self.tokens[sno].text, sep)
        #     else:   # entity type mode
        #         self.tokens[sno].text = etype[:4]   #em.type[:4]
        # elif mark_mode == 1:    # no action for multiple-word entity mention in entity type mode
        #     self.tokens[sno].text = '{} {}'.format(sep, self.tokens[sno].text)
        #     self.tokens[eno-1].text = '{} {}'.format(self.tokens[eno-1].text, sep)
        if mark_mode == 1:  # #@ mode
            self.tokens.insert(sno, word=sep)
            self.tokens.insert(eno+1, word=sep)
        elif eno - sno == 1:    # entity type mode for single-token entity mention
            self.tokens[sno].text = etype[:4]
        return

    def mark_entity_mention_pair(self, em1, em2, mark_mode=0):
        if mark_mode == 0:  return
        self.mark_entity_mention(self.hsno1, self.heno1, em1.type, sep='#', mark_mode=mark_mode)
        # modifying entity mention positions
        if mark_mode == 1:
            self.hsno1 += 1
            self.heno1 += 2
            #
            self.hsno2 += 2
            self.heno2 += 2
        if em2:
            self.mark_entity_mention(self.hsno2, self.heno2, em2.type, sep='@', mark_mode=mark_mode)
            if mark_mode == 1:
                self.hsno2 += 1
                self.heno2 += 2
        return

    def collect_re_confusion_matrices(self, docset, rconfusions, confusions, gold=False):
        """
        collect confusion matrices from a relation mention
        :param docset: DocSet with rtypedict and rlabeldict
        :param rconfusions: confusion matrix with reverse relationships
        :param confusions: confusion matrix w/o reverse relationships
        :param gold: gold relation mention or not
        :return:
        """
        # gold & predicted type
        gtype = get_relation_label(type=self.type, rvsid=self.rvsid)    # gold type
        ptype = get_relation_label(type=self.ptype, rvsid=self.prvsid)  # predicted type
        # gold relation mention or false positive
        if gold or gtype == 'None':
            gno = docset.rlabeldict[gtype]
            pno = docset.rlabeldict[ptype]
            rconfusions[gno][pno] += 1
        # calculate confusion matrix without reverse relations
        if docset.rvsID:
            gtype = get_relation_label(type=self.type, rvsid=False)
            ptype = get_relation_label(type=self.ptype, rvsid=False)
            if gold or gtype == 'None':
                if gtype == 'None':  gtype = 'Avg.'  # None in labels <--> Avg. in types
                if ptype == 'None':  ptype = 'Avg.'
                gno = docset.rtypedict[gtype]
                pno = docset.rtypedict[ptype]
                confusions[gno][pno] += 1
        return

