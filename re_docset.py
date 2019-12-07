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
def sum_re_confusion_matrix_to_prfs(confusions, prfs):
    prfs[:-1, 0] = confusions.sum(axis=-1)    # gold
    prfs[:-1, 1] = confusions.sum(axis=0)     # predicted
    for i in range(len(prfs)-1):  prfs[i, 2] = confusions[i, i]


# document set for NER
class reDocSet(ieDocSet):
    def __init__(self,
                 task = None,
                 stask = None,
                 wdir = None,
                 id = None,
                 fmt = 'i',
                 model_name = 'Lstm',
                 avgmode = 'micro',
                 elabel_schema = 'SBIEO',
                 tokenizer = None
                 ):
        super(reDocSet, self).__init__(task, stask, wdir, id, fmt, model_name, avgmode,
                                       elabel_schema, tokenizer)
        # relation
        self.noneID = False  # whether type 'None' exists
        self.rvsID = False   # whether reverse relations exist
        self.rrtypedict = {}  # dict from 3.R to index
        self.rtypedict = {}   # dict from 3 to index

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
    def prepare_docset_instance(self, op, tcfg, Doc, verbose=0):
        ffilename = '{}/{}.wrd'.format(self.wdir, self.id)  # feature file
        flines = file_line2array(ffilename, verbose=verbose)
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
    def prepare_docset_abstract(self, tcfg, Doc, verbose=0):
        self.load_docset_bio_text(Doc, verbose=verbose)
        efilename = '{}/{}.ent'.format(self.wdir, self.id)
        self.load_docset_entity_mentions(efilename, verbose=verbose)
        self.check_docset_entity_mentions(verbose=verbose)
        self.load_docset_relation_mentions(verbose=verbose)
        self.preprocess_docset_entity_mentions(tcfg)
        self.generate_docset_sentences(tcfg, Doc, verbose=verbose)
        self.collect_docset_instances(tcfg)
        return

    def load_docset_relation_mentions(self, verbose=0):
        rfilename = '{}/{}.rel'.format(self.wdir, self.id)
        if not os.path.exists(rfilename):  return False
        rlines = file_line2array(rfilename, verbose=verbose)
        for pmid, rid, emid1, emid2, type, name in rlines:
            # 23538162,T5-T19,T5,T19,4,DOWNREGULATOR
            doc = self.contains_doc(pmid)
            if doc is None:  print('DocPmidErr: {}'.format(pmid))
            # check entity id
            if emid1 not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, emid1))
            if emid2 not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, emid2))
            if emid1 == emid2:  continue    # self-relationship
            # reverse the relationship if necessary
            rvsID = False
            if doc.emlist[doc.emdict[emid2]].heno <= doc.emlist[doc.emdict[emid1]].hsno:  # if protein precceds chemical
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
    def create_label_type_dict(self, filename=None, verbose=0):
        # collect relation type
        self.create_reverse_relation_type_dict()
        self.create_non_reverse_relation_type_dict()
        # save to a json file
        if filename is not None:
            if verbose > 0: print('\nSaving config to {}'.format(filename))
            with open(filename, 'w') as outfile:
                cfg_dict = {'noneID':self.noneID, 'rvsID':self.rvsID, 'rrtypedict':self.rrtypedict, 'rtypedict':self.rtypedict}
                json.dump(cfg_dict, outfile, indent=2, sort_keys=True)
        return

    def update_relation_type_dict(self, rmc):
        if rmc.rvsid and not self.rvsID: self.rvsID = True  # auto detect reverse relations
        rtype = relation_type(type=rmc.type, rvsid=rmc.rvsid)
        if rtype == 'None':
            if not self.noneID: self.noneID = True
        else:
            if rtype not in self.rrtypedict:   self.rrtypedict[rtype] = len(self.rrtypedict)
        return

    # create relation type dict from different levels
    def create_reverse_relation_type_dict(self, level='inst'):
        # set to default values
        self.rrtypedict = {}
        self.noneID = False
        self.rvsID = False
        #
        if level == 'docu':
            for _, doc in self.docdict.items():
                for _, rmc in doc.rmdict.items():  # relation mention
                    self.update_relation_type_dict(rmc)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    for _, rmc in snt.rmdict.items():  # relation mention
                        self.update_relation_type_dict(rmc)
        else:  # 'inst'
            for rmc in self.insts:  # relation mention
                self.update_relation_type_dict(rmc)
        # sort
        for i, key in enumerate(sorted(self.rrtypedict)):    self.rrtypedict[key] = i
        if self.noneID: self.rrtypedict['None'] = len(self.rrtypedict)
        self.rrtypedict['Avg.'] = len(self.rrtypedict)
        return

    # create non-reverse relation types from reverse types
    def create_non_reverse_relation_type_dict(self):
        # type dict without reverse relations
        self.rtypedict = {}
        for key,idx in sorted(self.rrtypedict.items(), key=lambda d: d[1]):
            if key.endswith('.R'):  key = key[:-2]
            if key not in self.rtypedict: self.rtypedict[key] = len(self.rtypedict)
        return

    def set_label_type_dict(self, tdict):
        self.rrtypedict = tdict['rrtypedict']
        self.rtypedict = tdict['rtypedict']
        self.noneID = tdict['noneID']
        self.rvsID = tdict['rvsID']

    def get_label_type_dict(self):  # num of classes for relation extraction
        return self.rrtypedict

    # return sequence data for NER
    def get_docset_data(self, verbose=0):
        # get string-type examples
        insts, exams = self.insts, []
        if verbose > 0:
            print('\nGenerating {} candidates ...'.format(self.get_task_fullname()))
            insts = tqdm(insts)
        for inst in insts:
            exams.append(inst.output_instance_candidate(bertID=self.bertID))
        #
        doc_data, num_classes = self.output_docset_instance_candidates(exams, verbose=verbose)
        return doc_data, num_classes

    # output relation candidates
    def output_docset_instance_candidates(self, exams, verbose=0):
        # get label type dict
        labeldict = self.get_label_type_dict()
        num_classes = len(labeldict) - 1
        # save to the instance file
        # if filename is not None:
        #     dlines = ['{}\t{}'.format(' '.join(words), rtype) for words, rtype in insts]
        #     file_list2line(dlines, filename, verbose=verbose)

        # [[X1, X2],[Y1,Y2]] while Xn are lists, but Yn are not necessary
        data = [[[[self.get_word_idx(word) for word in words], [0]*len(words),[len(words)]],
                 [labeldict[rtype]]] for words, rtype in exams]

        # find a positive example to demonstrate
        if verbose:
            for i, rmc in enumerate(self.insts):
                if rmc.type is not None:
                    print(rmc)
                    for j, exam in enumerate(exams[i]):  print('I{}:'.format(j), exam)
                    for j, feature in enumerate(data[i][0]):  print('F{}:'.format(j), feature)
                    for j, label in enumerate(data[i][1]):  print('L{}:'.format(j), label)
                    break
            print('\n{:,} {} instances with {} classes'.format(len(data), self.id, num_classes))
            print(sorted(labeldict.items(), key=lambda d: d[1]))
        return data, num_classes

    def assign_docset_predicted_results(self, pred_nos):
        rmclist = self.insts
        rrdict = {no:type for type, no in self.rrtypedict.items()}  # no-->type
        #
        if len(pred_nos) != len(rmclist):
            print('Predicted: {} does not equal to candidates: {}'.format(len(pred_nos), len(rmclist)))
        for i, rmc in enumerate(rmclist):
            if pred_nos[i] not in rrdict:
                print('Predicted type: {} is out of scope!'. format(pred_nos[i]))
            rtype = rrdict[pred_nos[i]]
            prvsid = rtype.endswith('.R')
            ptype = rtype[:-2] if prvsid else rtype
            rmc.set_predicted_type(ptype=ptype, prvsid=prvsid)
        return


    # 0-Avg, 1-per type, 2-confusion matrix
    # level: 'inst'-instance, 'docu'-document
    def calculate_docset_f1(self, level='inst', mdlfile=None, logfile=None, rstfile=None, verbose=0):
        flog = sys.stdout
        if logfile: flog = open(logfile, 'a', encoding='utf8')
        if verbose > 0:
            sdt = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            print('\n{}/{}({}), {}, {}'.format(self.wdir, self.id, level, mdlfile, sdt), file=flog)
        # initialize confusion matrix
        rconfusions = np.zeros([len(self.rrtypedict)-1, len(self.rrtypedict)-1], dtype=int)
        confusions = np.zeros([len(self.rtypedict)-1, len(self.rtypedict)-1], dtype=int)
        # collect gold and predictions
        rprfs = np.zeros([len(self.rrtypedict), 6], dtype=float)
        rstlines = []
        if level == 'inst':  # instance
            #for rmc in self.rmclist:
            for rmc in self.insts:
                self.collect_re_confusion_matrices(rconfusions, confusions, rmc)
                rst = rmc.get_re_result()
                if rst and (verbose < 3 and rst[0] != rst[1]):  rstlines.append(rst)
        else:       # 'document'
            for _, doc in self.docdict.items():
                for _, rm in doc.rmdict.items():
                    self.collect_re_confusion_matrices(rconfusions, confusions, rm)
                for _, rrm in doc.rrmdict.items():
                    self.collect_re_confusion_matrices(rconfusions, confusions, rrm)
        #
        if verbose >= 2 and rstfile:
            rlines = ['{}-->{}\t{}\t{}'.format(r[0], r[1], r[3], r[2]) for r in sorted(rstlines)]
            file_list2line(rlines, rstfile)
        #
        sum_re_confusion_matrix_to_prfs(rconfusions, prfs=rprfs)
        # output confusion matrix
        #print(sorted(self.rrtypedict.items(), key=lambda d: d[1]))
        if verbose >= 1:
            print('Confusion matrix for relation types:', file=flog)
            output_confusion_matrix(cm=rconfusions, ldict=self.rrtypedict, file=flog)
        # calculate and output prf
        calculate_classification_prfs(rprfs, noneID=self.noneID, avgmode=self.avgmode)
        print('\nPRF performance for relation types:', file=flog)
        output_classification_prfs(rprfs, self.rrtypedict, flog=flog, verbose=verbose)
        if not self.rvsID:  # non-reverse relationships
            if logfile is not None: flog.close()
            print('{:>6}({}): {}'.format(self.id, level, format_row_prf(rprfs[-1])))
            return rprfs[-1]
        #print(np.array_str(rprfs, precision=2))
        # output confusion matrix
        if verbose >= 1:
            print('\nConfusion matrix for non-reverse relation types:', file=flog)
            #print(np.array_str(confusions), file=flog)
            output_confusion_matrix(cm=confusions, ldict=self.rtypedict, file=flog)
        # calculate and output PRFs without reverse relationships
        prfs = np.zeros([len(self.rtypedict), 6], dtype=float)
        # merge reverse rprfs to non-reverse prfs
        for rtype, ridx in self.rrtypedict.items():
            if rtype.endswith('.R'):  rtype = rtype[:-2]
            idx = self.rtypedict[rtype]
            prfs[idx,:3] += rprfs[ridx,:3]
        # calculate and output PRFs
        calculate_classification_prfs(prfs, noneID=self.noneID, avgmode=self.avgmode)
        if verbose > 0:  print('\nPRF performance for non-reverse relation types:', file=flog)
        output_classification_prfs(prfs, self.rtypedict, flog=flog, verbose=verbose)
        if logfile: flog.close()
        print('{:>6}({}): {}'.format(self.id, level, format_row_prf(prfs[-1])))
        return prfs[-1]

    # for RE/URE
    def dispatch_docset_predicted_results(self):
        # docset.insts --> rmc.prvsid, rmc.ptype -->
        # docset.docdict --> doc.rmdict/doc.rrmdict
        # clear the predicted results
        for _, doc in self.docdict.items():
            for _, rm in doc.rmdict.items():
                rm.set_predicted_type(ptype=None, prvsid=False)
            doc.rrmdict = {}
        #
        for rmc in self.insts:
            if rmc.ptype == 'None':  continue          # negative is neglected
            #print(rmc)
            if self.task == 're':
                did, _, emid1, emid2 = rmc.id.split('-')     # docid-lineno-emid1-emid2
                rid = '{}-{}'.format(emid1, emid2)  # id for relation mention in an abstract
            else:  #  URE
                did, _, emid1 = rmc.id.split('-')  # docid-lineno-emid1
                rid, emid2 = emid1, None
            #
            doc = self.contains_doc(did)
            if not doc:
                print('PmidIdErr: {} {}'.format(did, rmc))
                continue
            #
            if rid in doc.rmdict:   # annotated
                doc.rmdict[rid].set_predicted_type(ptype=rmc.ptype, prvsid=rmc.prvsid)
            else:
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
        # for _, doc in self.docdict.items():
        #     # pick the results from rmdict and rrmdict respectively
        #     for rmdict in (doc.rmdict, doc.rrmdict):
        #         for _, rrm in sorted(rmdict.items(), key=lambda x: x[0]):
        #             if self.task == 'ure':
        #                 rmlines.append([doc.id, rrm.emid1, rrm.ptype])
        #             else:
        #                 emid1, emid2 = rrm.emid1, rrm.emid2
        #                 if rrm.prvsid:  emid1, emid2 = emid2, emid1
        #                 rmlines.append([doc.id, emid1, emid2, rrm.ptype])
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

    # level: 'inst', 'sent', 'docu'
    def generate_docset_relation_statistics(self, level='inst'):
        # initialize counts
        rcounts = np.zeros([len(self.rrtypedict)], dtype=int)
        counts = np.zeros([len(self.rtypedict)], dtype=int)
        #
        # collect statistics
        if level == 'inst':  # instance
            #for rmc in self.rmclist:
            for rmc in self.insts:
                self.collect_relation_statistics(rcounts, counts, rmc)
        elif level == 'docu':       # 'document'
            for _, doc in self.docdict.items():
                for _, rm in doc.rmdict.items():
                    self.collect_relation_statistics(rcounts, counts, rm)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    for _, rm in snt.rmdict.items():
                        self.collect_relation_statistics(rcounts, counts, rm)
        # sum
        #print(rcounts)
        aidx = -2 if self.noneID else -1  # idx for average, None is always at the bottom
        rcounts[-1] = np.sum(rcounts[:aidx], axis=0)
        counts[-1] = np.sum(counts[:aidx], axis=0)
        return rcounts, counts

    def collect_relation_statistics(self, rcounts, counts, rmc):
        # gold type & no
        gtype = relation_type(type=rmc.type, rvsid=rmc.rvsid)  # gold type
        gno = self.rrtypedict[gtype]
        rcounts[gno] += 1
        # calculate confusion matrix without reverse relations
        if self.rvsID:
            gtype = relation_type(type=rmc.type, rvsid=False)
            gno = self.rtypedict[gtype]
            counts[gno] += 1
        return

    def collect_re_confusion_matrices(self, rconfusions, confusions, rmc):
        # gold & predicted type
        gtype = relation_type(type=rmc.type, rvsid=rmc.rvsid)  # gold type
        ptype = relation_type(type=rmc.ptype, rvsid=rmc.prvsid)  # predicted type
        # gold & predicted no
        gno = self.rrtypedict[gtype]
        pno = self.rrtypedict[ptype]
        rconfusions[gno][pno] += 1
        # calculate confusion matrix without reverse relations
        if self.rvsID:
            gtype = relation_type(type=rmc.type, rvsid=False)
            ptype = relation_type(type=rmc.ptype, rvsid=False)
            gno = self.rtypedict[gtype]
            pno = self.rtypedict[ptype]
            confusions[gno][pno] += 1
        return

# DocSet class for unary relation extraction
class ureDocSet(reDocSet):
    # load unary relation mentions
    def load_docset_relation_mentions(self, verbose=0):
        rfilename = '{}/{}.urel'.format(self.wdir, self.id)
        rlines = file_line2array(rfilename, verbose=verbose)
        for pmid, emid1, type, name in rlines:
            # 23538162	T5	4	DOWNREGULATOR
            doc = self.contains_doc(pmid)
            if doc is None:  print('DocPmidErr: {}'.format(pmid))
            # check entity id
            if emid1 not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, emid1))
            types = type.split('|')
            if len(types) == 1:  types = [type, None]
            #
            urm = RelationMention(id=emid1, type=types[0], stype=types[1], name=name, emid1=emid1)
            doc.append_relation_mention(urm)
        return
