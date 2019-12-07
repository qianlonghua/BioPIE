from ie_doc import *

# document class for RE
class reDoc(ieDoc):
    def __init__(self,
                 id=None,
                 no=None,
                 title='',
                 text=''
                 ):

        super(reDoc, self).__init__(id, no, title, text)
        self.rmdict = {}  # relation mention dictionary from emid1-emid2 --> RelationMention
        self.rrmdict = {}  # recognized relation mentions

    def __str__(self):
        sdoc = super(reDoc, self).__str__()
        rlist = '\n'.join(['{}'.format(rm) for _, rm in self.rmdict.items()])
        srlist = '\nRELATIONS:\n{}'.format(rlist) if len(rlist) > 0 else ''
        rlist = '\n'.join(['{}'.format(rm) for _, rm in self.rrmdict.items()])
        srrlist = '\nRECOGNIZED RELATIONS:\n{}'.format(rlist) if len(rlist) > 0 else ''
        return '{}{}{}'.format(sdoc, srlist, srrlist)

    def append_relation_mention(self, rm, gold=True):
        if gold:  self.rmdict[rm.id] = rm
        else:  self.rrmdict[rm.id] = rm

    # generate sentences for a document
    # fmt: 's'-sentence, 'a'-abstract, 'f'-full-text
    def generate_document_sentences(self, tcfg, Doc, task='re', fmt='a', verbose=0):
        self.replace_bio_special_chars(BIO_SPECIAL_CHARS)
        # for BEL, do not split the sentence
        lines = split_bio_sentences(fmt, self.text)
        # convert to class instances
        for i, line in enumerate(lines):
            dline = tokenize_bio_sentence(line)
            if tcfg.sent_simplify:
                dline = simplify_bio_sentence(dline, 'PAREN')
            dline = tokenize_bio_eos(dline)
            # make a sentence
            snt = Doc(id=self.id, no=i, title=self.title, text=dline)
            self.recover_entity_mentions(tcfg, snt)
            self.transfer_relation_mentions(snt)
            self.sntlist.append(snt)
        return

    def transfer_relation_mentions(self, snt):
        # build relation mention list for the sentence
        for rid, rm in self.rmdict.items():
            if rm.visited:  continue
            if rm.emid1 in snt.emdict and rm.emid2 in snt.emdict:
                em1 = snt.emlist[snt.emdict[rm.emid1]]
                em2 = snt.emlist[snt.emdict[rm.emid2]]
                # whether em1 and em2 are overlapped
                if em1.lineno == em2.lineno and (em1.hsno == em2.hsno or em1.heno == em2.heno):
                    print('Overlapped entities:\n{}\n{}'.format(em1, em2))
                    rm.text = snt.text
                    print(rm)
                else:
                    snt.append_relation_mention(rm.__copy__())  # text=snt.text
                    rm.visited = True   # set the visited mark
        return

    # generate relation candidates for a sentence
    # differTypeID: relation candidates are between entities with different types
    def generate_sentence_instances(self, tcfg):
        candidates = []
        if len(self.emlist) < 2: return candidates
        # GET1-->GENE_1, CHT2-->CHEM_1 for bert, is it helpful???
        sent = blind_text_entity_placeholders(self.text)
        for i, em1 in enumerate(self.emlist):
            for j in range(i+1, len(self.emlist)):
                em2 = self.emlist[j]
                if em1.hsno == em2.hsno or em1.heno == em2.heno:    # overlapped entity mentions
                    continue
                if tcfg.diff_ent_type and em1.type == em2.type:             # different types of entities are needed
                    continue
                rm = self.generate_relation_mention_candidate(sent, em1, em2)
                candidates.append(rm)
        return candidates

    #
    def generate_relation_mention_candidate(self, sent, em1, em2=None):
        rid = '{}{}'.format(em1.id, '-{}'.format(em2.id) if em2 else '')
        if rid in self.rmdict:  # positive
            rm = self.rmdict[rid].__copy__(text=sent)
        else:  # negative or predicted instances, default type is None
            emid2 = em2.id if em2 else None
            rm = RelationMention(id=rid, type=None, emid1=em1.id, emid2=emid2, text=sent)
        # pmid-lineno-eid1-eid2
        rm.id = '{}-{}-{}'.format(self.id, self.no, rid)
        # GENE_n-->GENE, CHEM_n-->CHEM
        # rm.blind_relation_entity_mentions(em1, em2, bertID=bertID)
        # GENE_n--># GENE #, CHEM_n-->@ CHEM @
        rm.insert_entity_mentions_delimiters(em1, em2, recover=False)
        return rm


# Doc class for unary relation extraction
class ureDoc(reDoc):
    # build unary relation mention list for the sentence
    def transfer_relation_mentions(self, snt):
        for rid, rm in self.rmdict.items():
            if rm.visited:  continue
            if rm.emid1 in snt.emdict:
                snt.append_relation_mention(rm.__copy__())  # text=snt.text
                rm.visited = True   # set the visited mark
        return

    # generate urelation candidates for a sentence
    def generate_sentence_instances(self, tcfg):
        if len(self.emlist) < 1: return []  # one entity is OK
        sent = blind_text_entity_placeholders(self.text)
        candidates = [self.generate_relation_mention_candidate(sent, em1) for em1 in self.emlist]
        return candidates

