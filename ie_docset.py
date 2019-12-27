"""
processing document set
"""
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ie_doc import *


def npdiv0(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def output_confusion_matrix(cm, ldict):
    rows = [k for k, v in sorted(ldict.items(), key=lambda x: x[1])]
    crange = range(len(cm[0]))
    rows = [rows[c] for c in crange]
    #
    cols = [['TYPE', 6, 0]]
    cols.extend([[k, 4, 0] for k in rows])
    #
    lines = output_matrix(cm, cols, rows)
    #print('\n'.join(lines), file=file)
    return lines

#      0    1    2 None
#  [1662   29    5  139]    0
#  [   8 1107   13  210]    1
#  [  14    4 1702  121]    2
#  [  94  208   85    0] None
# noneID: whether label 'None' exists in the confusion matrix, which is True for NER
def sum_confusion_matrix_to_prfs(confusions, prfs, noneID=False):
    end = confusions.shape[0] - 1 if noneID else confusions.shape[0]
    prfs[:-1, 0] = confusions[range(end)].sum(axis=-1)  # gold
    prfs[:-1, 1] = confusions[:, range(end)].sum(axis=0)  # predicted
    for i in range(len(prfs) - 1):  prfs[i, 2] = confusions[i, i]

# 1..
# ...
# Avg.
def calculate_classification_prfs(prfs, noneID=False, avgmode='micro'):
    aidx = -2 if noneID else -1 # idx for average, None is always at the bottom
    prfs[-1,:3] = np.sum(prfs[:aidx,:3], axis=0)
    prfs[:,3] = npdiv0(prfs[:,2], prfs[:,1]) * 100
    prfs[:,4] = npdiv0(prfs[:,2], prfs[:,0]) * 100
    prfs[:,5] = npdiv0(2 * prfs[:,3] * prfs[:,4], prfs[:,3] + prfs[:,4])
    if avgmode == 'macro':
        prfs[-1,3:] = np.average(prfs[:aidx,3:], axis=0)
    return

def format_row_prf(prf):
    sprf = '{:6.0f}\t{:6.0f}\t{:6.0f}\t{:6.2f}\t{:6.2f}\t{:6.2f}'.format(prf[0], prf[1], prf[2], prf[3], prf[4], prf[5])
    return sprf

def output_classification_prfs(prfs, rtypedict, verbose=0):
    cols = (('TYPE', 6, 0), ('GOLD', 6, 0), ('TP+FP', 6, 0), ('TP', 6, 0), ('P', 6, 2), ('R', 6, 2), ('F1', 6, 2))
    rows = [key for key, idx in sorted(rtypedict.items(), key=lambda x: x[1])]
    lines = output_matrix(prfs, cols, rows)
    #print('\n'.join(lines) if verbose >= 1 else lines[-1], file=flog)
    return lines if verbose >= 1 else [lines[-1]]

# cols=[['TYPE', 6, 0],...], rows=['0',...'Avg.']
# output matrix like:
#
# TYPE      0    1    2 None
#  0    [1662   29    5  139]
#  1    [   8 1107   13  210]
#  2    [  14    4 1702  121]
#  Avg. [  94  208   85    0]
def output_matrix(m, cols, rows=None, colsepch='\t'):
    if m is None or cols is None:   return None
    if rows and len(cols) != len(m[0])+1 or not rows and len(cols) != len(m[0]):
        print('Columns {} and matrix {} do not match!'.format(len(cols), len(m[0])))
        return ''
    # make the title
    lines, heads = [], []
    for col in cols:
        fmt = make_matrix_cell_format(col, titleID=True)
        heads.append(fmt.format(col[0]))
    lines.append(colsepch.join(heads))
    # make the matrix
    for i, row in enumerate(m):
        vals = []
        for j,col in enumerate(cols):
            k = j-1 if rows else j
            fmt = make_matrix_cell_format(col, titleID=(k < 0))
            vals.append(fmt.format(rows[i] if k < 0 else m[i][k]))
        lines.append(colsepch.join(vals))
    return lines

def make_matrix_cell_format(col, titleID=False):
    if titleID:    # the title or the row header
        fmt = '{}:>{}{}'.format('{', col[1], '}')
    else:
        fmt = '{}:>{},.{}f{}'.format('{', col[1], col[2], '}')
    return fmt

# return features for a sentence in *.ann
def get_ner_instance(flines):
    i = 0   # line no
    while i < len(flines):
        if flines[i][0] != '':
            eos = False
            for j in range(i+1, len(flines)):
                if flines[j][0] == '':
                    yield flines[i:j]
                    i = j
                    eos = True
                    break
            if not eos:
                yield flines[i:]
                i = len(flines)
        else:
            i += 1
    yield None


# document set for NER
class ieDocSet(object):
    def __init__(self,
                 task = None,
                 stask = None,
                 wdir = None,
                 id = None,
                 fmt = 'i',
                 elabel_schema = 'SBIEO'  # elabel_schema can be set by tcfg, but loaded from cfg file
                 ):
        self.task = task    # standard NLP task, such as re, ure, ner
        self.stask = stask  # source NLP task to be cast as task
        self.fmt = fmt      # i-instance, s-sentence, a-title/abstract, f-full-text
        self.id = id        # train/dev/text
        self.wdir = wdir    # GE09/test

        # the following fields can't be copied
        self.docdict = {}    # doc dict from pmid to document
        self.insts = []      # string-type instances
        self.data = []       # interger-type data

        # entity and word dicts
        self.elabel_schema = elabel_schema
        self.etypedict = {}
        self.elabeldict = {}
        self.worddict = {}

    def __str__(self):
        docs = [doc[1].__str__() for doc in sorted(self.docdict.items(), key=lambda d:d[0])]
        return '\n{}/{} task={} stask={} fmt={} \n{}'.format(self.wdir, self.id, self.task, self.stask, self.fmt, '\n'.join(docs))

    def __copy__(self, task=None, stask=None, fmt=None):
        ds = ieDocSet()
        for att in self.__dict__.keys():
            if att not in ('docdict', 'insts', 'data'):
                ds.__dict__[att] = self.__dict__[att]
        if task:  ds.task = task
        if stask:  ds.stask = stask
        if fmt:  ds.fmt = fmt
        return ds

    # get task fullname
    def get_task_fullname(self):
        if not self.stask: return self.task
        return '{}_{}'.format(self.stask, self.task)

    def append_doc(self, did, doc): # doc id and doc instance
        self.docdict[did] = doc

    def append_instance(self, inst):
        self.insts.append(inst)

    def extend_instances(self, insts):
        self.insts.extend(insts)

    def extend_data(self, data):
        self.data.extend(data)

    # return doc instance if docset contain doc, else None
    def contains_doc(self, did):
        if did in self.docdict:
            return self.docdict[did]
        print('DocPmidErr: {}'.format(did))
        return None

    def load_docset_bio_sentence(self, Doc, tfilename, verbose=0):
        tlines = file_line2array(tfilename, verbose=verbose)
        for pmid, tline in tlines:
            self.append_doc(pmid, Doc(id=pmid, text=tline))
        return

    def load_docset_bio_abstract(self, Doc, tfilename, verbose=0):
        tlines = file_line2array(tfilename, verbose=verbose)
        for tline in tlines:
            if len(tline) < 3:  print(tline)
            pmid, title, tline = tline
            tabs = tline    # for BioASQ
            if self.task in ('re', 'ner', 've'):    # for RE and NER, URE?
                tabs = ' '.join([title, tline])
            self.append_doc(pmid, Doc(id=pmid, title=title, text=tabs))
        return

    def load_docset_bio_fulltext(self, Doc, verbose=0):
        # read full-test
        mypath = os.path.join(self.wdir, self.id)
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.txt')]
        if verbose:
            print('\nLoading full-text docs from {}'.format(mypath))
            files=tqdm(files)
        for file in files:
            tfilename = os.path.join(mypath, file)
            text = file_line2string(tfilename)
            self.append_doc(did=file[:-4], doc=Doc(id=file[:-4], text=text))
        return

    def load_docset_bio_text(self, Doc, verbose=0):
        # read abstracts
        tfilename = '{}/{}.txt'.format(self.wdir, self.id)
        if self.fmt == 's':
            self.load_docset_bio_sentence(Doc, tfilename, verbose=verbose)
        elif self.fmt == 'a':
            self.load_docset_bio_abstract(Doc, tfilename, verbose=verbose)
        elif self.fmt == 'f':
            self.load_docset_bio_fulltext(Doc, verbose=verbose)

    def load_docset_entity_mentions(self, efilename, verbose=0):
        if not os.path.exists(efilename):  return False
        elines = file_line2array(efilename, verbose=verbose)
        for pmid, eid, type, spos, epos, name, link in elines:
            # prepare type and links
            types, links = type.split('|'), link.split(':')
            if len(types) == 1:  types = [type, None]
            if len(links) == 1:  links = [None, link]
            # generate entity mention
            em = EntityMention(did=pmid, id=eid, type=types[0][:4], stype=types[1], name=name,
                               linkdb=links[0], linkid=links[1], hsno=int(spos), heno=int(epos))
            doc = self.contains_doc(pmid)
            if doc: doc.append_entity_mention(em)
        return True

    # check location consistency between entity annotation and text
    def check_docset_entity_mentions(self, pmid=None, verbose=0):
        for did, doc in self.docdict.items():
            if pmid and did != pmid:  continue
            #print(did)
            doc.sort_entity_mentions()
            for em in doc.emlist:
                if verbose and doc.otext[em.hsno:em.heno] != em.name.replace('_', ' '):
                    print('EntTxtCon: {}\t{}\t{}'.format(doc.id, doc.text[em.hsno:em.heno], em))

    #
    def preprocess_docset_entity_mentions(self, tcfg):
        for _, doc in self.docdict.items():
            doc.preprocess_document_entity_mentions(tcfg)

    def generate_docset_sentences(self, tcfg, Doc):
        dictdocs = self.docdict.items()
        if tcfg.verbose:
            print('\nGenerating {} sentences from documents ...'.format(self.get_task_fullname()))
            dictdocs = tqdm(dictdocs)
        for _, doc in dictdocs:
            doc.generate_document_sentences(tcfg, Doc, self.task, self.fmt)
            #print(doc)

    def collect_docset_instances(self, tcfg):
        for _, doc in sorted(self.docdict.items(), key=lambda d:d[0]):
            for snt in doc.sntlist:
                insts = snt.generate_sentence_instances(tcfg)
                self.extend_instances(insts)

    # prepare NER docset for training or prediction
    # must be overwritten due to ieDoc
    def prepare_docset_abstract(self, op, tcfg, Doc):
        self.load_docset_bio_text(Doc, verbose=tcfg.verbose)
        if op != 'p':   # not only prediction
            efilename = '{}/{}.ent'.format(self.wdir, self.id)
            self.load_docset_entity_mentions(efilename, verbose=tcfg.verbose)
            self.check_docset_entity_mentions(verbose=tcfg.verbose)
        self.generate_docset_sentences(tcfg, Doc)  # no entity, relations, events
        self.collect_docset_instances(tcfg)
        return

    # preparing NER docset from sentence-style file *.iob like CONLL2003, JNLPBA-2004
    # convert spaces to tabs in CoNLL2003
    def prepare_docset_instance(self, op, tcfg, Doc):
        ffilename = '{}/{}.iob'.format(self.wdir, self.id)  # feature file
        self.load_docset_instance(op, Doc, ffilename, verbose=tcfg.verbose)
        self.collect_docset_instances(tcfg)
        return

    def load_docset_instance(self, op, Doc, ffilename, verbose=0):
        flines = file_line2array(ffilename, verbose=verbose)
        # make vocabularies
        sent_get = get_ner_instance(flines)
        feats = next(sent_get)
        # a virtual document for all sentences
        doc = Doc(id=self.id)
        self.append_doc(self.id, doc)
        sent_no = 0
        while feats:
            # print_list(feats)
            text = ' '.join([feat[0] for feat in feats])
            # add a sentence to the doc
            snt = Doc(id=self.id, no=sent_no, text=text)
            if 'r' in op or 't' in op or 'v' in op:  # train/test
                labels = [feat[-1] for feat in feats]
                # recover entity mentions
                ems = get_entity_mentions_from_labels(labels)
                for eno, spos, epos, type in ems:
                    snt.add_entity_mention_to_sentence(self.id, sent_no, feats, eno, spos, epos, type)
            # print(feats)
            doc.sntlist.append(snt)
            # print(snt)
            feats = next(sent_get)
            sent_no += 1
        return

    def update_entity_type_dict(self, typedict, emlist):
        for em in emlist:
            if em.type not in typedict:
                typedict[em.type] = len(typedict)

    # collect label types from instances, sentences, or docs
    def create_entity_type_dict(self, level='inst'):
        # collect
        typedict, self.etypedict = {}, {}
        if level == 'docu':
            for _, doc in self.docdict.items():
                self.update_entity_type_dict(typedict, doc.emlist)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    self.update_entity_type_dict(typedict, snt.emlist)
        else:   #  'inst'
            for inst in self.insts:  # inst is SequenceLabel
                self.update_entity_type_dict(typedict, inst.emlist)
        # sort
        for i, key in enumerate(sorted(typedict)):    self.etypedict[key] = i
        self.etypedict['Avg.'] = len(self.etypedict)
        return

    # create label dict from entity types
    def create_entity_label_dict(self, label_schema):
        self.elabel_schema = label_schema
        self.elabeldict = {'PAD':0, 'O':1}
        for type in sorted(self.etypedict):
            if type != 'Avg.':
                labeldict = {'{}-{}'.format(label, type):len(self.elabeldict)+i for i, label in enumerate(label_schema[:-1])}
                self.elabeldict.update(labeldict)
        return

    # create type dict from entity mentions
    def create_type_label_dict(self, tcfg, filename=None):
        # entity type dict
        self.create_entity_type_dict()
        self.create_entity_label_dict(label_schema=tcfg.elabel_schema)
        # save to config file
        if filename is not None:
            if tcfg.verbose:  print('Saving config to {}'.format(filename))
            with open(filename, 'w') as outfile:
                cfg_dict ={'eschema':self.elabel_schema, 'etype':self.etypedict, 'elabel':self.elabeldict}
                json.dump(cfg_dict, outfile, indent=2, sort_keys=False)
        return

    def set_type_label_dict(self, tdict):
        # entity
        self.elabel_schema = tdict['eschema']
        self.etypedict = tdict['etype']
        self.elabeldict = tdict['elabel']
        return

    def get_type_dict(self):  return self.etypedict

    def get_label_dict(self):  return self.elabeldict

    def set_word_vocab_dict(self, wdict):  self.worddict = wdict

    def generate_docset_instance_features(self, tcfg):
        insts = self.insts
        if tcfg.verbose:
            print('Generating {} {} features'.format(self.get_task_fullname(), tcfg.model_name))
            insts = tqdm(insts)
        for inst in insts:
            inst.generate_instance_features(tcfg, self)
        return

    def create_docset_word_vocab(self, filename=None, verbose=0):
        self.worddict = {'[PAD]':0, '[UNK]':1, '[NUM]':2}
        snt_len = 0
        #for ist in self.sntlist if self.task == 'ner' else self.rmclist: # 're' or 'ner'
        for inst in self.insts:
            if len(inst.words) > snt_len:  snt_len = len(inst.words)
            for word in inst.words:
                if word not in self.worddict: self.worddict[word] = len(self.worddict)
        #
        if filename is not None:
            print('The longest sentence is {} in length.'.format(snt_len))
            save_word_voc_file(self.worddict, filename, verbose=verbose)
        return

    def set_docset_word_vocab(self, word_dict=None):
        if word_dict is not None: self.worddict = word_dict

    def prepare_docset_dicts_features(self, tcfg, cfg_dict=None, word_dict=None):
        task, wdir, cps_file = self.get_task_fullname(), self.wdir, self.id
        if cfg_dict is None:
            cfilename = '{}/{}_{}_cfg.json'.format(wdir, task, cps_file)  # save config file
            if os.path.exists(cfilename):  cfg_dict=load_json_file(cfilename)
            else:
                self.create_type_label_dict(tcfg, filename=cfilename)
        # set label type dict
        if cfg_dict:
            self.set_type_label_dict(tdict=cfg_dict)
        # generate docset re features
        self.generate_docset_instance_features(tcfg)
        # save or set word vocabulary
        if word_dict is None:
            vfilename = '{}/{}_{}_voc.txt'.format(wdir, task, cps_file)  # word vocab file
            self.create_docset_word_vocab(filename=vfilename, verbose=tcfg.verbose)
        else:
            self.set_docset_word_vocab(word_dict=word_dict)  # share word dict across RE & NER
        return

    def get_word_idx(self, word):
        if word not in self.worddict: word = '[UNK]'
        return self.worddict[word]

    # # return sequence data for NER
    def get_docset_data(self, tcfg):
        # get the examples
        insts, exams = self.insts, []
        if tcfg.verbose > 0:
            print('Generating {} candidates ...'.format(self.get_task_fullname()))
            insts = tqdm(insts)
        for inst in insts:
            exams.append(inst.output_instance_candidate(bertID=tcfg.bertID))
        #
        data, num_classes, labeldict, exno = self.output_docset_instance_candidates(exams, verbose=tcfg.verbose)
        # display an example
        if tcfg.verbose:
            print('\n{}\n'.format(self.insts[exno]))
            for j, exam in enumerate(exams[exno]):  print('I{}:'.format(j), exam)
            for j, feature in enumerate(data[exno][0]):  print('F{}:'.format(j), feature)
            for j, label in enumerate(data[exno][1]): print('L{}:'.format(j), label)
            # display the label dictionary
            print('\n{:,} {} instances with {} classes'.format(len(data), self.id, num_classes))
            print(sorted(labeldict.items(), key=lambda d: d[1]))
        return data, num_classes

    # output ner candidates for a docset
    def output_docset_instance_candidates(self, exams, verbose=0):
        """
        :param exams:
        :param verbose:
        :return:
        """
        # get the label dictionary
        labeldict = self.get_label_dict()
        num_classes = [len(labeldict)]
        # get integer-type data
        data = [[[[self.get_word_idx(word) for word in words], [0]*len(words), [len(words)]],
                [[labeldict[label] for label in labels]]] for words, labels in exams]
        # display an example
        exno = 0
        if verbose:
            for i, inst in enumerate(self.insts):
                if len(inst.emlist) > 1:  # at least two entity mentions
                    exno = i
                    break
        return data, num_classes, labeldict, exno

    # assign predicted to the instances
    def assign_docset_predicted_results(self, pred_nos, bertID=False):
        # create dict of idx to tag
        idx2tag = {i:tag for tag, i in self.elabeldict.items()}
        pred_labels = [[idx2tag[i] for i in row] for row in pred_nos]
        # generate remlist for every sentence
        for i, inst in enumerate(self.insts):
            # get predicted labels for a sentence
            #labels_len = len(inst.blabels if self.bertID else inst.labels)
            plabels = pred_labels[i][:len(inst.labels)]
            # generate remlist for a sentence
            inst.recognize_entity_mentions_from_labels(plabels=plabels, lineno=i, bertID=bertID)
            # convert recognized entities to original word positions for bert sequence
            if bertID:  inst.convert_bert_entity_mention_positions()
        return

    # docset.insts --> inst.remlist --> docset.docdict --> doc.remlist
    def dispatch_docset_predicted_results(self, level='docu', target_docset=None):
        odoc, mno = None, 1
        tds = target_docset if target_docset else self
        for inst in self.insts:
            doc = tds.contains_doc(inst.id)
            if not doc:  continue
            #
            if doc != odoc: odoc, mno = doc, 1  # if doc changes, set mno to 1
            for rem in inst.remlist:
                if rem.hsno >= rem.heno:  continue  # bert recognized, but ill-formed for word
                if '[CLS]' in rem.name:
                    print(inst)
                    # print(rem)
                    # print(snt.boffsets)
                if level == 'docu':
                    sno = inst.offsets[rem.hsno][0]
                    eno = inst.offsets[rem.heno-1][1]
                    ename = doc.text[sno:eno]
                    em = EntityMention(did=inst.id, id='T{}'.format(mno), type=rem.type, name=ename, hsno=sno, heno=eno)
                    doc.append_entity_mention(em, gold=False)
                else:       # 'sent'
                    ename = ' '.join(inst.words[rem.hsno:rem.heno])
                    em = EntityMention(did=inst.id, id='T{}'.format(mno), type=rem.type, name=ename,
                                       lineno=inst.no, hsno=rem.hsno, heno=rem.heno)
                    doc.sntlist[inst.no].append_entity_mention(em, gold=False)
                mno += 1
        return

    def collect_docset_performance(self, level='inst', avgmode='micro', verbose=0):
        typedict = self.etypedict
        # confusion matrix for NER
        confusions = np.zeros([len(typedict), len(typedict)], dtype=int)
        rstlines = []
        #
        if level == 'inst': # instance
            for inst in self.insts:
                inst.match_gold_pred_instances()
                inst.collect_ner_confusion_matrix(confusions, typedict)
                inst.get_ner_labeled_sentences(rstlines, verbose=verbose)
        else:   # document
            for _, doc in self.docdict.items():
                doc.match_gold_pred_instances()
                doc.collect_ner_confusion_matrix(confusions, typedict)
        # sum confusions to prfs, noneID is always True for NER
        prfs = np.zeros([len(typedict), 6], dtype=float)
        sum_confusion_matrix_to_prfs(confusions, prfs, noneID=True)
        # calculate prfs
        calculate_classification_prfs(prfs, avgmode=avgmode)
        return [confusions, prfs, rstlines]

    def calculate_docset_performance(self, level='inst', mdlfile=None, logfile=None, rstfile=None, avgmode='micro', verbose=0):
        performance = self.collect_docset_performance(level, avgmode, verbose)
        aprf = self.output_docset_performance(performance, level, mdlfile, logfile, rstfile, verbose)
        return aprf

    # verbose: 0-only consile overall performance, 1-confusion/PRF, 2-erroneous instances, 3-all instances
    def output_docset_performance(self, performance, level='inst', mdlfile=None, logfile=None, rstfile=None, verbose=0):
        confusions, prfs, rstlines = performance
        olines = []
        if verbose >= 1:
            # output header
            sdt = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            olines.append('\n{}/{}({}), {}, {}'.format(self.wdir, self.id, level, mdlfile, sdt))
            # output confusion matrix
            olines.append('Confusion matrix for entity types:')
            olines.extend(output_confusion_matrix(cm=confusions, ldict=self.etypedict))
            # output P/R/F
            olines.append('\nPRF performance for entity types:')
            #print(sorted(self.etypedict.items(), key=lambda d: d[1]))
            olines.extend(output_classification_prfs(prfs, self.etypedict, verbose=verbose))
        # output result files
        if verbose >= 2 and level=='inst' and rstfile:
            file_list2line(rstlines, rstfile)
        #
        if logfile is not None:
            flog = open(logfile, 'a', encoding='utf8')
            print('\n'.join(olines), file=flog)
            flog.close()
        else:  print('\n'.join(olines))
        # always output the overall performance to the console
        print('{:>6}({}): {}'.format(self.id, level, format_row_prf(prfs[-1])))
        return prfs[-1]

    # output predicted NER results
    # pmid eid type spos epos name
    def output_docset_predicted_results(self, rstfile=None, verbose=0):
        # collect
        remlines = []
        if self.fmt == 'i':
            for inst in self.insts:
                for rem in sorted(inst.remlist):
                    sem = '{} {}|{}|{} {} {}|{}'.format(inst.id, rem.id, rem.type, rem.name, inst.no, rem.hsno, rem.heno)
                    remlines.append(sem)
        else:
            for _, doc in self.docdict.items():
                for rem in sorted(doc.remlist):
                    sem = '{}\t{}\t{}\t{}\t{}\t{}'.format(doc.id, rem.id, rem.type, rem.hsno, rem.heno, rem.name)
                    remlines.append(sem)
        # output
        if rstfile is not None:
            file_list2line(remlines, rstfile)
        else:
            print(remlines)
        #
        if verbose:
            dstr = ' to {}'.format(rstfile) if rstfile else '.'
            print('\nOutput totally {} entity mentions{}'.format(len(remlines), dstr))
        return

    # level: 'inst', 'sent', 'docu'
    def generate_docset_instance_statistics(self, level='inst'):
        # initialize counts
        typedict = self.get_type_dict()
        counts = np.zeros([len(typedict)], dtype=int)
        # collect statistics
        if level == 'inst':  # instance
            # Inst may be SequenceLabel, RelationMentions, sBelStatements
            for inst in self.insts:
                inst.collect_instance_statistics(counts, typedict)
        elif level == 'docu':       # 'document'
            for _, doc in self.docdict.items():
                doc.collect_instance_statistics(counts, typedict)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    snt.collect_instance_statistics(counts, typedict)
        # sum
        counts[-1] = np.sum(counts[:-1], axis=0)
        return counts

    # statistics for entity mentions specifically
    # level: 'sent', 'docu'
    def generate_docset_entity_mention_statistics(self, level='sent'):
        # initialize counts
        typedict = self.etypedict
        counts = np.zeros([len(typedict)], dtype=int)
        # collect statistics
        if level == 'docu':       # 'document'
            for _, doc in self.docdict.items():
                collect_entity_mention_statistics(doc, counts, typedict)
        elif level == 'sent':
            for _, doc in self.docdict.items():
                for snt in doc.sntlist:
                    collect_entity_mention_statistics(snt, counts, typedict)
        # sum
        counts[-1] = np.sum(counts[:-1], axis=0)
        return counts

    #
    def evaluate_docset_model(self, op, tcfg, pred_classes, mdlfile=None):
        task = self.get_task_fullname()
        # instance-level performance
        lfilename = '{}/{}_{}.log'.format(self.wdir, self.id, task)
        rfilename = '{}/{}_{}.rst'.format(self.wdir, self.id, task)
        pfilename = '{}/{}_{}.prd'.format(self.wdir, self.id, task)
        #
        self.assign_docset_predicted_results(pred_nos=pred_classes, bertID=tcfg.bertID)
        #
        if 'v' in op:   # instance-level validation
            self.calculate_docset_performance(level='inst', mdlfile=mdlfile, logfile=lfilename, rstfile=rfilename,
                                              avgmode=tcfg.avgmode, verbose=tcfg.verbose)
        # document-level performance
        if self.fmt in 'saf':  # sentence/abstract/full text
            self.dispatch_docset_predicted_results()
            if 'v' in op:   # document-level validation
                self.calculate_docset_performance(level='docu', mdlfile=mdlfile, logfile=lfilename,
                                                  avgmode=tcfg.avgmode, verbose=tcfg.verbose)
        # predict
        if 'p' in op:
            self.output_docset_predicted_results(rstfile=pfilename, verbose=tcfg.verbose)
        return

# DocSet for segmented sequence labeling
class sslDocSet(ieDocSet):
    def __init__(self,
                task = None,
                stask = None,
                wdir = None,
                id = None,
                fmt = 'i',
                elabel_schema='SBIEO',
                slabel_schema='IO',
                ):
        super(sslDocSet, self).__init__(task, stask, wdir, id, fmt, elabel_schema)
        self.slabel_schema = slabel_schema
        self.stypedict = {}
        self.slabeldict = {}

    def create_segment_label_dict(self, label_schema):
        self.slabel_schema = label_schema
        self.slabeldict = {'O':0}
        for type in sorted(self.stypedict):
            labeldict = {'{}-{}'.format(label, type):len(self.slabeldict)+i for i, label in enumerate(label_schema[:-1])}
            self.slabeldict.update(labeldict)
        return

    # collect label types from the sentences in every doc
    def create_segment_type_dict(self):
        typedict, self.stypedict = {}, {}
        for inst in self.insts:  # inst is SequenceLabel
            for em in inst.smlist:
                if em.type not in typedict:
                    typedict[em.type] = len(typedict)
        # sort
        for i, key in enumerate(sorted(typedict)):    self.stypedict[key] = i
        return

    # create type dict from entity mentions
    def create_type_label_dict(self, tcfg, filename=None):
        # entity type dict
        super(sslDocSet, self).create_entity_type_dict()
        super(sslDocSet, self).create_entity_label_dict(tcfg.elabel_schema)
        # segment type dict
        self.create_segment_type_dict()
        self.create_segment_label_dict(label_schema=tcfg.slabel_schema)
        # save to config file
        if filename is not None:
            if tcfg.verbose:  print('Saving config to {}'.format(filename))
            with open(filename, 'w') as outfile:
                cfg_dict ={'eschema':self.elabel_schema, 'etype':self.etypedict, 'elabel':self.elabeldict,
                           'sschema':self.slabel_schema, 'stype':self.stypedict, 'slabel':self.slabeldict}
                print(cfg_dict)
                json.dump(cfg_dict, outfile, indent=2, sort_keys=False)
        return

    def set_type_label_dict(self, tdict):
        # entity
        super(sslDocSet, self).set_type_label_dict(tdict)
        # segment
        self.slabel_schema = tdict['sschema']
        self.stypedict = tdict['stype']
        self.slabeldict = tdict['slabel']
        return

    def get_segment_label_dict(self):
        return self.slabeldict

    def generate_docset_instance_features(self, tcfg):
        insts = self.insts
        if tcfg.verbose:
            print('Generating {} {} features'.format(self.get_task_fullname(), tcfg.model_name))
            insts = tqdm(insts)
        for inst in insts:
            inst.generate_instance_features(tcfg, self)
        return

    # output ner candidates for a docset
    def output_docset_instance_candidates(self, exams, verbose=0):
        # get the label dictionary
        labeldict = self.get_label_dict()
        num_classes = [len(labeldict)]
        segdict = self.get_segment_label_dict()
        # get integer-type data
        data = [[[[self.get_word_idx(word) for word in words],               # word_idx
                 [(1 if slabel != 'O' else 0) for slabel in slabels],        # seg_ids
                 [segdict[slabel] for slabel in slabels if slabel != 'O']],  # seg type
                [[labeldict[label] for label in labels]]] for words, labels, slabels in exams]
        # find an example
        exno = 0
        if verbose:
            for i, inst in enumerate(self.insts):
                if len(inst.emlist) > 1:  # at least two entity mentions
                    exno = i
                    break
        return data, num_classes, labeldict, exno