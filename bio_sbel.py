"""
"""

from ie_docsets import *

class sBelStatement(RelationMention):
    def __init__(self,
                 id=None,
                 type=None,
                 stype=None,
                 rvsid=False,  # reverse or not
                 name=None,    # type name
                 sname=None,   # subtype name
                 emid1=None,
                 emid2=None,
                 text='',
                 func1=None,
                 func2=None
                 ):

        super(sBelStatement, self).__init__(id=id, type=type, stype=stype, rvsid=rvsid, name=name,
                                            sname=sname, emid1=emid1, emid2=emid2, text=text)
        self.func1 = func1
        self.func2 = func2
        self.pfunc1 = None
        self.pfunc2 = None

    def __str__(self):
        str = super(sBelStatement, self).__str__()
        return '{}|{} {}'.format(self.func1, self.func2, str)

    def __copy__(self, text=None):
        bs = sBelStatement()
        for att in self.__dict__.keys(): bs.__dict__[att] = self.__dict__[att]
        if text is not None: bs.text = text
        return bs


# Doc class for sbel extraction
class sbelDoc(reDoc):
    #
    def generate_relation_mention_candidate(self, sent, em1, em2=None):
        rid = '{}-{}'.format(em1.id, em2.id)
        if rid in self.rmdict:  # positive
            rm = self.rmdict[rid].__copy__(text=sent)
        else:  # negative or predicted instances, default type is None
            rm = sBelStatement(id=rid, type=None, emid1=em1.id, emid2=em2.id, text=sent)
        # pmid-lineno-eid1-eid2
        rm.id = '{}-{}-{}'.format(self.id, self.no, rid)
        # GENE_n-->GENE, CHEM_n-->CHEM
        # rm.blind_relation_entity_mentions(em1, em2, bertID=bertID)
        # GENE_n--># GENE #, CHEM_n-->@ CHEM @
        rm.insert_entity_mentions_delimiters(em1, em2, recover=False)
        return rm

    def convert_sbel_statements_bre_mentions(self, snt):
        snt.emdict, snt.emlist = self.emdict, self.emlist
        snt.rmdict = {}
        for _, sbs in self.rmdict.items():
            # RelationMention
            if sbs.emid1 == sbs.emid2:  print(sbs)
            rm = super(sBelStatement, sbs).__copy__()
            snt.append_relation_mention(rm, gold=True)
        return

    def convert_sbel_statements_ure_mentions(self, snt):
        snt.emdict, snt.emlist = self.emdict, self.emlist
        snt.rmdict = {}
        for _, sbs in self.rmdict.items():
            # 10003274,200685643,T1,T2,inc,increases,None,tloc
            func1, func2 = sbs.func1, sbs.func2
            if func1 in ('cat', 'kin', 'tscript'):  func1 = 'act'
            if func2 in ('cat', 'kin', 'tscript'):  func2 = 'act'
            # get the first function for an entity
            if func1 != 'None' and sbs.emid1 not in snt.rmdict:
                rm = RelationMention(id=sbs.emid1, type=func1, emid1=sbs.emid1)
                snt.append_relation_mention(rm, gold=True)
            if func2 != 'None' and sbs.emid2 not in snt.rmdict:
                rm = RelationMention(id=sbs.emid2, type=func2, emid1=sbs.emid2)
                snt.append_relation_mention(rm, gold=True)
        return


# DocSet class for SBEL
class sbelDocSet(reDocSet):
    def load_docset_relation_mentions(self, verbose=0):
        rfilename = '{}/{}.sbel'.format(self.wdir, self.id)
        rlines = file_line2array(rfilename, verbose=verbose)
        self_cnt, slines = 0, []
        for rline in rlines:
            # 10003274,200685643,T1,T2,inc,increases,None,tloc
            pmid, belid, emid1, emid2, type, name, func1, func2 = rline
            doc = self.contains_doc(pmid)
            if doc is None:  print('DocPmidErr: {}'.format(pmid))
            #
            if emid1 not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, emid1))
            if emid2 not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, emid2))
            #
            if emid1 == emid2:  # skip self relationship
                slines.extend(['', doc.emlist[doc.emdict[emid1]].__str__(), '\t'.join(rline), doc.text])
                self_cnt += 1
                continue
            # set rvsID is necessary
            rvsID = False
            if doc.emlist[doc.emdict[emid2]].heno <= doc.emlist[doc.emdict[emid1]].hsno:  # if protein precceeds chemical
                emid1, emid2 = emid2, emid1
                rvsID = True
            # append BEL statement
            id = '{}-{}'.format(emid1, emid2)
            #
            if type == 'dinc':  type = 'inc'
            elif type == 'ddec':  type = 'dec'
            sbs = sBelStatement(id=id, type=type, name=name, rvsid=rvsID, emid1=emid1, emid2=emid2,
                                func1=func1, func2=func2)
            doc.append_relation_mention(sbs)
        print('Num of self-relation is {}'.format(self_cnt))
        file_list2line(slines, '{}/{}.self'.format(self.wdir, self.id))
        return slines

    # overwrite temporarily
    def prepare_docset_dicts_features(self, cfg_dict=None, word_dict=None, verbose=0):
        pass

    # if two entities have multiple relations, then how?
    # convert sbel statements (docset.docdict) to relations
    def convert_docset_sbel_re(self, tcfg, DocSet, Doc, task=None, stask=None, verbose=0):
        docs = self.docdict.items()
        if verbose:
            print('\nConverting {}/{} extraction to {} ...'.format(self.wdir, self.id, stask))
            docs = tqdm(docs)
        # DocSet is reDocSet, since sbelDocSet hasn't copy method while its parent class has.
        docset = self.__copy__(task=task, stask=stask, fmt='i')
        for _, doc in docs:    # sbelDoc
            for snt in doc.sntlist:  # only one sentence in a BEL statement
                # convert sentence's sBEL statements to bre relation mentions
                bsnt = Doc(id=snt.id, no=snt.no, title=snt.title, text=snt.text)
                if task == 're':
                    snt.convert_sbel_statements_bre_mentions(bsnt)
                else:
                    snt.convert_sbel_statements_ure_mentions(bsnt)
                rmcs = bsnt.generate_sentence_instances(tcfg)
                docset.insts.extend(rmcs)
        return docset

# DocSets class for sbel
class sbelDocSets(reDocSets):
    # convert sbelDocSets to reDocSets
    def convert_docsets_sbel_re(self, tcfg, DocSets, DocSet, Doc, task='re', stask='sbel',verbose=0):
        resets = self.__copy__(DocSets, task=task, stask=stask)
        for fileset in self.filesets:
            reds = fileset.convert_docset_sbel_re(tcfg, DocSet, Doc, task=task, stask=stask, verbose=verbose)
            reds.prepare_docset_dicts_features(self.cfg_dict, self.word_dict, verbose=verbose)
            resets.filesets.append(reds)
        return resets


def review_sbel_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=0):
    sbelsets = sbelDocSets(task, wdir, cpsfiles, cpsfmts)
    sbelsets.prepare_corpus_filesets(op, tcfg, sbelDocSet, sbelDoc, verbose=verbose)
    # relation statistics as RE
    # crcounts, ccounts = sbelsets.calculate_docsets_relation_statistics()
    # sbelsets.output_docsets_relation_statistics(crcounts, ccounts, logfile='relations.cnt')
    # binary relation statistics
    bresets = sbelsets.convert_docsets_sbel_re(tcfg, reDocSets, reDocSet, reDoc, 're', 'sbel', verbose=verbose)
    crcounts, ccounts = bresets.calculate_docsets_relation_statistics()
    bresets.output_docsets_relation_statistics(crcounts, ccounts, logfile='brelations.cnt')
    # # unary relation statistics
    uresets = sbelsets.convert_docsets_sbel_re(tcfg, reDocSets, ureDocSet, ureDoc, 'ure', 'sbel', verbose=verbose)
    crcounts, ccounts = uresets.calculate_docsets_relation_statistics()
    uresets.output_docsets_relation_statistics(crcounts, ccounts, logfile='urelations.cnt')
    return

# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bert_path=None, model_name='LstmCrf', avgmode='micro',
                      epo=0, fold=0, folds=None):
    #
    #elif task == 've':  DocSets = veDocSets
    iesets = create_corpus_filesets(sbelDocSets, task, wdir, cpsfiles, cpsfmts, model_name, bert_path=bert_path, avgmode=avgmode)

    iesets.prepare_corpus_filesets(op, tcfg, sbelDocSet, sbelDoc, verbose=1)
    bresets = iesets.convert_docsets_sbel_re(tcfg, reDocSets, reDocSet, reDoc, 're', 'sbel', verbose=1)
    bresets.train_eval_docsets(op, tcfg, reDocSet, epo, fold, folds)
    uresets = iesets.convert_docsets_sbel_re(tcfg, reDocSets, ureDocSet, ureDoc, 'ure', 'sbel', verbose=1)
    uresets.train_eval_docsets(op, tcfg, ureDocSet, epo, fold, folds)

    clear_gpu_processes()
    return


def main(op, task, wdir, cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', mdlname='Bert', tcfg=None, epo=0, fold=0, folds=None):
    bert_path = './bert-model/biobert-pubmed-v1.1'

    if 'r' in op: # prepare word vocabulary
        review_sbel_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=1)
    else:
        train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bert_path=bert_path,
                          model_name=mdlname, avgmode='micro', epo=epo, fold=fold, folds=folds)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main('b', '', sys.argv[1])
    else:   # f-format conversion, c-corpus preprocess, t-train, v-validate, p-predict
        elist = ('GENE', 'CHEM', 'DISE', 'BPRO')    # entity types to be replaced/blinded with PLACEHOLDER
        tcfg = TrainConfig(epochs=3, valid_ratio=0, fold_num=10, fold_num_run=1,
                           max_seq_len=100, batch_size=32,
                           bld_ent_types=elist, sent_simplify=1, diff_ent_type=0)
        # VE & SBEL
        main('r', 'sbel', 'BEL', ('train', 'test'), 'ss', 'Bert', tcfg, epo=2)  # SBEL
        #