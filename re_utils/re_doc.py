from ner_utils.ner_doc import *
from re_utils.re_instance import *

# document class for RE
class reDoc(nerDoc):
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

    def transfer_document_annotations(self, tcfg):
        """
        postprocess document sentences
        :param tcfg:
        :return:
        """
        for snt in self.sntlist:
            self.recover_entity_mentions(tcfg, snt)
            self.transfer_relation_mentions(snt)
            # build the tokens for the sentence
            snt.build_tokens()
            # todo, load various linguistic features here, like POS, dep, etc.
            # ...
        return

    def transfer_relation_mentions(self, snt):
        """
        transfer relation mentions from a document to its sentence
        :param snt: the target sentence
        :return: set visited if a relation mention is transferred to the sentence
        """
        # build relation mention list for the sentence
        for rid, rm in self.rmdict.items():
            if rm.visited:  continue
            em1 = snt.get_entity_mention(rm.emid1)
            em2 = snt.get_entity_mention(rm.emid2)
            # if two entity mentions occur in the sentence
            if em1 and em2:
                # whether em1 and em2 are overlapped
                if em1.lineno == em2.lineno and (em1.hsno == em2.hsno or em1.heno == em2.heno):
                    print('Overlapped entities:\n{}\n{}'.format(em1, em2))
                    #rm.text = snt.text
                    print(rm)
                else:  # transfer the relation mention
                    nrm = rm.__copy__(hsno1=em1.hsno, heno1=em1.heno, hsno2=em2.hsno, heno2=em2.heno)
                    snt.append_relation_mention(nrm)
                    rm.visited = True   # set the visited mark
        return

    def generate_sentence_instances(self, tcfg):
        """
        generate relation instances for a sentence
        :param tcfg:
        :return: list of instances
        """
        instances = []
        if len(self.emlist) < 2: return instances
        # GENET1-->GENE_1, CHEMT2-->CHEM_1 for bert, is it helpful???
        tokens = self.tokens.__copy__()
        tokens.blind_entity_mention_placeholders(tcfg.bld_ent_mode)
        # generate relation instances in a pair-wise way
        for i, em1 in enumerate(self.emlist):
            for j in range(i+1, len(self.emlist)):
                em2 = self.emlist[j]
                if em1.hsno == em2.hsno or em1.heno == em2.heno:    # overlapped entity mentions
                    continue
                if tcfg.diff_ent_type and em1.type == em2.type:     # different types of entities are needed
                    continue
                rm = self.generate_relation_mention_candidate(tokens, em1, em2, tcfg.mark_ent_pair)
                instances.append(rm)
        return instances

    def generate_relation_mention_candidate(self, tokens, em1, em2=None, mark_mode=0):
        """
        generate a relation candidate
        :param sent: sentence
        :param em1: 1st entity mention
        :param em2: 2nd entity mention
        :return: relation mention
        """
        rid = '{}{}'.format(em1.id, '-{}'.format(em2.id) if em2 else '')
        if rid in self.rmdict:  # positive instance
            rm = self.rmdict[rid].__copy__(tokens=tokens)   # ???
        else:  # negative or predicted instances, default type is None
            emid2 = em2.id if em2 else None
            hsno2, heno2 = -1, -1
            if em2:  hsno2, heno2 = em2.hsno, em2.heno  # for unary relation mentions
            rm = RelationMention(id=rid, type=None, emid1=em1.id, emid2=emid2, tokens=tokens,
                                 hsno1=em1.hsno, heno1=em1.heno, hsno2=hsno2, heno2=heno2)
        # pmid-lineno-eid1-eid2
        rm.id = '{}-{}-{}'.format(self.id, self.no, rid)
        # mark two entity mentions, like GENE_n-->GENE | # GENE #, CHEM_n-->CHEM | @ GENE @
        rm.mark_entity_mention_pair(em1, em2, mark_mode=mark_mode)
        return rm

    def match_gold_pred_instances(self):
        """
        match relation mentions between gold and recognized
        :return: set rm.ptype, rm.prvsid and rrm.type, rrm.rvsid accordingly
        """
        # rmdict: rid --> relation mention
        for rid, rm in self.rmdict.items():
            rm.set_predicted_type(ptype=None, prvsid=False)
            # recognized rm, no matter the relation type
            if rid in self.rrmdict:
                rrm = self.rrmdict[rid]
                if rrm.ptype == 'None': # duplicated rid, using the last one
                    print('ErrDupRelMen: {} {}'.format(self.id, rid))
                rm.set_predicted_type(ptype=rrm.ptype, prvsid=rrm.prvsid)
                rrm.type, rrm.rvsid = rm.type, rm.rvsid
        return

    def collect_re_confusion_matrices(self, docset, rconfusions, confusions):
        """
        collect confusion matrices for RE
        :param docset: DocSet with rtypedict and rlabeldict
        :param rconfusions: confusion matrix with reverse relationships
        :param confusions: confusion matrix without reverse relationships
        :return:
        """
        for _, rm in self.rmdict.items():
            rm.collect_re_confusion_matrices(docset, rconfusions, confusions, gold=True)
        for _, rrm in self.rrmdict.items():
            rrm.collect_re_confusion_matrices(docset, rconfusions, confusions, gold=False)


# Doc class for unary relation extraction
class ureDoc(reDoc):

    def transfer_relation_mentions(self, snt):
        """
        transfer unary relation mentions to the sentence
        :param snt: the target sentence
        :return: set the transferred relation mentions
        """
        for rid, rm in self.rmdict.items():
            if rm.visited:  continue
            em1 = snt.get_entity_mention(rm.emid1)
            if em1:
                snt.append_relation_mention(rm.__copy__(hsno1=em1.hsno, heno1=em1.heno))  # text=snt.text
                rm.visited = True   # set the visited mark
        return

    def generate_sentence_instances(self, tcfg):
        """
        generate unary relation instances for a sentence
        :param tcfg:
        :return:
        """
        if len(self.emlist) < 1: return []  # one entity is OK
        tokens = self.tokens.__copy__()
        tokens.blind_entity_mention_placeholders(tcfg.bld_ent_mode)
        #
        instances = []
        for em1 in self.emlist:
            urm = self.generate_relation_mention_candidate(tokens, em1, mark_mode=tcfg.mark_ent_pair)
            instances.append(urm)
        return instances

