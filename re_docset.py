
from ie_docset import *
from re_doc import *


# return meta and features of a relation instance in *.wrd
def get_re_instance(flines):
    i = 0   # line no
    while i < len(flines):
        #print(flines[i])
        hlen, slen = int(flines[i][-2]), int(flines[i][-1])
        fpos = i + hlen
        yield flines[i:fpos], flines[fpos:fpos+slen]
        i += hlen + slen + 1
    yield None, None


#  confusion matrix for relation classification
#     0   1   2   3   4   5   6   7   8 None
#  [139   0   8  13   2   0   0   1   1  28]    0
#  [  0 309   0   0   4   0   0   1   2  12]    1
#  [  2   1 265   0   1  12   5   2   0  24]    2
#  [  0   0   0 274   0   0   0   0   0  18]    3
#  [  0   2   1   2 226   0   0   1   2  24]    4
#  [  0   0   2   3   1 124   0   0   2  24]    5
#  [  0   0   3   1   2   1 209   0   0  17]    6
#  [  0   0   4   0   0   0   1 233   0  23]    7
#  [  0   3   0   0   7   2   0   1 197  21]    8
#  [  6  15  26  15  17  16  29  25  16 289]  None
#

# document set for NER
class reDocSet(ieDocSet):
    def __init__(self,
                 task = None,
                 stask = None,
                 wdir = None,
                 id = None,
                 fmt = 'i',
                 ):
        super(reDocSet, self).__init__(task, stask, wdir, id, fmt)
        # relation
        self.noneID = False  # whether type 'None' exists
        self.rvsID = False   # whether reverse relations exist
        self.rtypedict = {}   # type dict from 3 to index
        self.rlabeldict = {}  # label dict from 3.R to index

    def __copy__(self, task=None, stask=None, fmt=None):
        ds = reDocSet()
        for att in self.__dict__.keys():
            if att not in ('docdict', 'insts', 'data'):
                ds.__dict__[att] = self.__dict__[att]
        if task:  ds.task = task
        if stask:  ds.stask = stask
        if fmt:  ds.fmt = fmt
        return ds

    # get RE instances from file like *.wrd
    def prepare_docset_instance(self, op, tcfg, Doc):
        ffilename = '{}/{}.wrd'.format(self.wdir, self.id)  # feature file
        flines = file_line2array(ffilename, verbose=tcfg.verbose)
        #
        inst_get = get_re_instance(flines)
        metas, feats = next(inst_get)
        while metas:
            tokens = [feat[1] for feat in feats]
            rid = '{}-{}-{}-{}'.format(metas[0][1], metas[0][2], metas[0][3], metas[0][4])
            rvsid = (metas[1][1] == 'R')
            type = metas[1][2] if rvsid else metas[1][1]
            #
            inst = RelationMention(id=rid, emid1=metas[0][3], emid2=metas[0][4], type=type, rvsid=rvsid,
                                   text=' '.join(tokens))
            #self.append_relation_candidate(inst)
            self.append_instance(inst)
            metas, feats = next(inst_get)
        return

    # prepare doc set for biomedical relation extraction
    def prepare_docset_abstract(self, op, tcfg, Doc):
        self.load_docset_bio_text(Doc, verbose=tcfg.verbose)
        efilename = '{}/{}.ent'.format(self.wdir, self.id)
        self.load_docset_entity_mentions(efilename, verbose=tcfg.verbose)
        self.check_docset_entity_mentions(verbose=tcfg.verbose)
        if op != 'p':  # not only prediction
            self.load_docset_relation_mentions(verbose=tcfg.verbose)
        self.preprocess_docset_entity_mentions(tcfg)
        self.generate_docset_sentences(tcfg, Doc)
        self.collect_docset_instances(tcfg)
        return

    def load_docset_relation_mentions(self, verbose=0):
        rfilename = '{}/{}.rel'.format(self.wdir, self.id)
        if not os.path.exists(rfilename):  return False
        rlines = file_line2array(rfilename, verbose=verbose)
        for pmid, rid, emid1, emid2, type, name in rlines:
            # 23538162,T5-T19,T5,T19,4,DOWNREGULATOR
            if emid1 == emid2:  continue  # self-relationship
            doc = self.contains_doc(pmid)
            if doc is None:  continue
            # check entity
            em1, em2 = doc.get_entity_mention(emid1), doc.get_entity_mention(emid2)
            if not em1 and verbose:  print('EntIdErr: {} {}'.format(pmid, emid1))
            if not em2 and verbose:  print('EntIdErr: {} {}'.format(pmid, emid2))
            # reverse the relationship if necessary
            rvsID = False
            if em1 and em2 and em2.heno <= em1.hsno:  # if protein preceeds chemical
                emid1, emid2 = emid2, emid1
                rvsID = True
            # append relation mention
            rid = '{}-{}'.format(emid1, emid2)
            types = type.split('|')
            if len(types) == 1:  types = [type, None]
            #
            rm = RelationMention(id=rid, type=types[0], stype=types[1], rvsid=rvsID, name=name, emid1=emid1, emid2=emid2)
            doc.append_relation_mention(rm)
        return True

    # create dict from relation instances
    def create_type_label_dict(self, tcfg, filename=None):
        # collect relation type
        self.create_relation_type_dict()
        self.create_relation_label_dict()
        # save to a json file
        if filename is not None:
            if tcfg.verbose > 0: print('\nSaving config to {}'.format(filename))
            with open(filename, 'w') as outfile:
                cfg_dict = {'noneID':self.noneID, 'rvsID':self.rvsID, 'rtypedict':self.rtypedict, 'rlabeldict':self.rlabeldict}
                json.dump(cfg_dict, outfile, indent=2, sort_keys=False)
        return

    def update_relation_type_dict(self, rmc, rtypedict):
        if rmc.rvsid and not self.rvsID: self.rvsID = True  # auto detect reverse relations
        rtype=rmc.type
        if rtype == 'None' or rtype is None:
            if not self.noneID: self.noneID = True
        else:
            if rtype not in rtypedict:   rtypedict[rtype] = len(rtypedict)
        return

    # create relation type dict from different levels
    def create_relation_type_dict(self, level='inst'):
        # set to default values
        rtypedict, self.rtypedict = {}, {}
        self.noneID = False
        self.rvsID = False
        #
        if level == 'docu':
            for _, doc in self.docdict.items():
                for _, rmc in doc.rmdict.items():  # relation mention
                    self.update_relation_type_dict(rmc, rtypedict)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    for _, rmc in snt.rmdict.items():  # relation mention
                        self.update_relation_type_dict(rmc, rtypedict)
        else:  # 'inst'
            for rmc in self.insts:  # relation mention
                self.update_relation_type_dict(rmc, rtypedict)
        # sort
        for i, key in enumerate(sorted(rtypedict)):    self.rtypedict[key] = i
        self.rtypedict['Avg.'] = len(self.rtypedict)
        return

    # create relation labels from relation types
    def create_relation_label_dict(self):
        self.rlabeldict = {}
        for key, idx in sorted(self.rtypedict.items(), key=lambda d: d[1]):
            if idx < len(self.rtypedict) - 1:
                self.rlabeldict[key] = len(self.rlabeldict)
                if self.rvsID:  # add the reverse relation label
                    self.rlabeldict[key+'.R'] = len(self.rlabeldict)
            elif self.noneID:   # the last one
                self.rlabeldict['None'] = len(self.rlabeldict)
        return

    def set_type_label_dict(self, tdict):
        self.rlabeldict = tdict['rlabeldict']
        self.rtypedict = tdict['rtypedict']
        self.noneID = tdict['noneID']
        self.rvsID = tdict['rvsID']

    def get_label_dict(self):  return self.rlabeldict

    def get_type_dict(self):  return self.rtypedict

    # output relation candidates
    def output_docset_instance_candidates(self, exams, verbose=0):
        # get label type dict
        labeldict = self.get_label_dict()
        num_classes = [len(labeldict)]

        # [[X1, X2],[Y1,Y2]] while Xn are lists, but Yn are not necessary
        data = [[[[self.get_word_idx(word) for word in words], [0]*len(words),[len(words)]],
                 [labeldict[label]]] for words, label in exams]

        # find a positive example to demonstrate
        exno = 0
        if verbose:
            for i, rmc in enumerate(self.insts):
                if rmc.type and rmc.type != 'None':
                    exno = i
                    break
        return data, num_classes, labeldict, exno

    def assign_docset_predicted_results(self, pred_nos, bertID=False):
        idx2label = {idx:label for label, idx in self.rlabeldict.items()}  # no-->type
        insts = self.insts
        #
        if len(pred_nos) != len(insts):
            print('Predicted: {} does not equal to candidates: {}'.format(len(pred_nos), len(insts)))
        #
        for i, rmc in enumerate(insts):
            if pred_nos[i] not in idx2label:
                print('Predicted type: {} is out of scope!'. format(pred_nos[i]))
            #
            rtype = idx2label[pred_nos[i]]
            prvsid = rtype.endswith('.R')
            ptype = rtype[:-2] if prvsid else rtype
            rmc.set_predicted_type(ptype=ptype, prvsid=prvsid)
        return

    # 0-Avg, 1-per type, 2-confusion matrix
    # level: 'inst'-instance, 'docu'-document
    def collect_docset_performance(self, level='inst', avgmode='micro', verbose=0):
        typedict = self.rtypedict
        # initialize confusion matrix
        rconfusions = np.zeros([len(self.rlabeldict), len(self.rlabeldict)], dtype=int)
        confusions = np.zeros([len(typedict), len(typedict)], dtype=int)
        rstlines = []
        #
        if level == 'inst':  # instance
            for rmc in self.insts:
                rmc.collect_re_confusion_matrices(self, rconfusions, confusions, gold=True)
                rmc.get_re_labeled_instance(rstlines, verbose=verbose)
        else:                # 'document'
            for _, doc in self.docdict.items():
                doc.match_gold_pred_instances()
                doc.collect_re_confusion_matrices(self, rconfusions, confusions)
        # sum gold and predictions
        prfs = np.zeros([len(typedict), 6], dtype=float)
        if self.rvsID:
            rprfs = np.zeros([len(self.rlabeldict), 3], dtype=float)
            sum_confusion_matrix_to_prfs(rconfusions, rprfs, self.noneID)
            # merge reverse relations to non-reverse relations in prfs
            for rtype, ridx in self.rlabeldict.items():
                if rtype == 'None':  continue
                if rtype.endswith('.R'):  rtype = rtype[:-2]
                idx = self.rtypedict[rtype]
                prfs[idx, :3] += rprfs[ridx, :3]
            #sum_confusion_matrix_to_prfs(confusions, prfs, self.noneID)
        else:
            sum_confusion_matrix_to_prfs(rconfusions, prfs, self.noneID)
        # calculate PRFs
        calculate_classification_prfs(prfs, avgmode=avgmode)
        return [rconfusions, confusions, prfs, rstlines]


    # 0-Avg, 1-per type performance, confusion matrix, 2-erroneous, 3-all
    # level: 'inst'-instance, 'docu'-document
    def calculate_docset_performance(self, level='inst', mdlfile=None, logfile=None, rstfile=None, avgmode='micro', verbose=0):
        performance = self.collect_docset_performance(level, avgmode, verbose)
        aprf = self.output_docset_performance(performance, level, mdlfile, logfile, rstfile, verbose)
        return aprf

    def output_docset_performance(self, performance, level='inst', mdlfile=None, logfile=None, rstfile=None, verbose=0):
        rconfusions, confusions, prfs, rstlines = performance
        olines = []
        if verbose >= 1:
            # output header
            sdt = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            olines.append('\n{}/{}({}), {}, {}'.format(self.wdir, self.id, level, mdlfile, sdt))
            # output reverse confusion matrix
            olines.append('\nConfusion matrix for reverse relation types:')
            olines.extend(output_confusion_matrix(cm=rconfusions, ldict=self.rlabeldict))
            # output the confusion matrix
            if self.rvsID:
                olines.append('\nConfusion matrix for non-reverse relation types:')
                olines.extend(output_confusion_matrix(cm=confusions, ldict=self.rtypedict))
            # output P/R/F1
            olines.append('\nPRF performance for non-reverse relation types:')
            #print(sorted(self.rtypedict.items(), key=lambda d: d[1]))
            olines.extend(output_classification_prfs(prfs, self.rtypedict, verbose))
        # output result lines
        if verbose >= 2 and level == 'inst' and rstfile:
            #rlines = ['{}-->{}\t{}\t{}'.format(r[0], r[1], r[3], r[2]) for r in sorted(rstlines)]
            file_list2line(rstlines, rstfile)
        #
        if logfile is not None:
            flog = open(logfile, 'a', encoding='utf8')
            print('\n'.join(olines), file=flog)
            flog.close()
        else:  print('\n'.join(olines))
        # always return the overall performance
        print('{:>6}({}): {}'.format(self.id, level, format_row_prf(prfs[-1])))
        return prfs[-1]

    # for RE/URE
    def dispatch_docset_predicted_results(self, level='docu', target_docset=None):
        # docset.insts --> rmc.prvsid, rmc.ptype -->
        # docset.docdict --> doc.rmdict/doc.rrmdict
        # clear the predicted results
        tds = target_docset if target_docset else self
        for _, doc in tds.docdict.items():
            for _, rm in doc.rmdict.items():
                rm.set_predicted_type(ptype=None, prvsid=False)
            doc.rrmdict = {}
        # dispatch results to documents
        for rmc in self.insts:
            if rmc.ptype == 'None':  continue          # negative is neglected
            #print(rmc)
            if self.task == 'ure':  # URE
                did, _, emid1 = rmc.id.split('-')       # docid-lineno-emid1
                rid, emid2 = emid1, None
            else:  #  RE
                did, _, emid1, emid2 = rmc.id.split('-')  # docid-lineno-emid1-emid2
                rid = '{}-{}'.format(emid1, emid2)  # id for relation mention in an abstract
            # check whether did exists
            doc = tds.contains_doc(did)
            if doc is None:  continue
            # add the recognized relation mention to rrm
            rm = RelationMention(id=rid, emid1=emid1, emid2=emid2)
            rm.set_predicted_type(ptype=rmc.ptype, prvsid=rmc.prvsid)
            doc.append_relation_mention(rm, gold=False)
        return

    # output predicted RE/URE results
    # RE: pmid emid1 emid2 3
    # URE: pmid emid1 3
    def output_docset_predicted_results(self, rstfile=None, verbose=0):
        # collect
        rmlines = []
        for rrm in self.insts:  # inst is RelationMention
            if rrm.ptype == 'None':  continue
            ids = rrm.id.split('-')  # the 1st token is pmid by default
            if self.task == 'ure':
                rmlines.append([ids[0], rrm.emid1, rrm.ptype])
            else:
                emid1, emid2 = rrm.emid1, rrm.emid2
                if rrm.prvsid:  emid1, emid2 = emid2, emid1
                rmlines.append([ids[0], emid1, emid2, rrm.ptype])
        # sort the output
        rmlines = ['\t'.join(rmline) for rmline in sorted(rmlines)]
        # output
        if rstfile is not None:
            file_list2line(rmlines, rstfile)
        else:
            print(rmlines)
        #
        if verbose:
            dstr = ' to {}'.format(rstfile) if rstfile else '.'
            print('\nOutput totally {} relation mentions{}'.format(len(rmlines), dstr))
        return


# DocSet class for unary relation extraction
class ureDocSet(reDocSet):
    # load unary relation mentions
    def load_docset_relation_mentions(self, verbose=0):
        rfilename = '{}/{}.urel'.format(self.wdir, self.id)
        rlines = file_line2array(rfilename, verbose=verbose)
        for pmid, emid1, type, name in rlines:
            # 23538162, T5, 4, DOWNREGULATOR
            doc = self.contains_doc(pmid)
            if doc is None:  continue
            # check entity id
            if emid1 not in doc.emdict and verbose:  print('EntIdErr: {} {}'.format(pmid, emid1))
            types = type.split('|')
            if len(types) == 1:  types = [type, None]
            #
            urm = RelationMention(id=emid1, type=types[0], stype=types[1], name=name, emid1=emid1)
            doc.append_relation_mention(urm)
        return
