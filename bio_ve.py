
from ie_docsets import *

# instance class for VE
class EventArgument(object):
    def __init__(self,
                 id=None,
                 type=None,
                 role=None):
        self.id = id
        self.type = type    # GENE, EVENT, ENTITY(optional)
        self.role = role

    def __str__(self):
        return '|'.join([self.role, self.id, self.type])

class EventMention(object):
    def __init__(self,
                 id=None,
                 type=None,
                 triggerid=None):
        self.id = id
        self.type = type            # event type
        self.triggerid = triggerid  # trigger word id
        self.arglist = []           # args list

        self.visited = False

    def __str__(self):
        sevt = '|'.join([self.id, self.type, self.triggerid])
        sarg = '\t'.join([arg.__str__() for arg in self.arglist])
        return '{}\t{}'.format(sevt, sarg)

    def append_argument(self, va):
        self.arglist.append(va)


# Doc class for VE
class veDoc(ieDoc):
    def __init__(self,
                 id=None,
                 no=None,
                 title='',
                 text=''
                 ):

        super(veDoc, self).__init__(id, no, title, text)
        self.vmdict = {}  # event mention dictionary from vid --> EventMention
        self.rvmdict = {}  # recognized relation mentions

    def __str__(self):
        sdoc = super(veDoc, self).__str__()
        vlist = '\n'.join(['{}'.format(vm) for _, vm in self.vmdict.items()])
        svlist = '\nEVENTS:\n{}'.format(vlist) if len(vlist) > 0 else ''
        rvlist = '\n'.join(['{}'.format(vm) for _, vm in self.rvmdict.items()])
        srvlist = '\nRECOGNIZED EVENTS:\n{}'.format(rvlist) if len(rvlist) > 0 else ''
        return '{}{}{}'.format(sdoc, svlist, srvlist)

    def append_event_mention(self, vm, gold=True):
        if gold:  self.vmdict[vm.id] = vm
        else:  self.rvmdict[vm.id] = vm

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
            # make a sentence, is this reDoc extended to ureDoc/sbelDoc?
            snt = Doc(id=self.id, no=i, title=self.title, text=dline)
            self.recover_entity_mentions(tcfg, snt)
            self.transfer_event_mentions(snt)
            self.sntlist.append(snt)

        # check event completeness
        if task == 've' and verbose >= 2:  # always exists
            if not all([vm.visited for vid, vm in self.vmdict.items()]):
                # print(self)
                for _, vm in self.vmdict.items():
                    if not vm.visited:
                        print('\nMissing event in {}:\n{}'.format(self.id, vm))
        return

    def transfer_event_mentions(self, snt):
        if all([self.vmdict[vid].visited for vid in self.vmdict]):  return
        # build event mention list for the sentence
        for vid, vm in self.vmdict.items():
            if vm.visited:  continue
            if vm.triggerid in snt.emdict:  # trigger word found
                snt.append_event_mention(vm)  # args may span multiple sentences
                vm.visited = True
        return

    # create the trigger-argument relation mentions in the sentence
    def convert_ve_arguments_relation_mentions(self, Doc):
        # create a new sentence
        snt = Doc(self.id, self.no, self.title, self.text)
        for vid, vm in self.vmdict.items():  # for all events
            # event for out of the sentence
            if vm.triggerid not in self.emdict:   continue
            # for all arguments
            for va in vm.arglist:
                argid = va.id
                if va.type == 'EVENT':  # another valid event
                    if argid not in self.vmdict:  continue
                    argid = self.vmdict[argid].triggerid
                # invalid trigger id, or inter-sentence arguments
                if argid not in self.emdict:   continue
                # construct a relation mention between the trigger and the argument (maybe another event!)
                emid1, emid2, rvsID = vm.triggerid, argid, False
                # if the trigger is after the argument, reverse the relationship
                if self.get_entity_mention(emid1).hsno > self.get_entity_mention(emid2).hsno:
                    emid1, emid2 = emid2, emid1
                    rvsID = True
                rid = '{}-{}'.format(emid1, emid2)
                rm = RelationMention(id=rid, type=va.role, emid1=emid1, emid2=emid2, rvsid=rvsID)
                # append the relationship
                snt.append_relation_mention(rm, gold=True)
                if emid1 not in snt.emdict: snt.append_entity_mention(self.get_entity_mention(emid1).__copy__())
                if emid2 not in snt.emdict: snt.append_entity_mention(self.get_entity_mention(emid2).__copy__())
        snt.sort_entity_mentions()
        return snt

    # return new sentences
    def convert_ve_arguments_entity_mentions(self, Doc):
        nsnts = []
        for vid, vm in self.vmdict.items():  # for all events
            if vm.triggerid not in self.emdict:   continue
            # make a sentence of ieDoc
            nsnt = Doc(self.id, self.no, self.title, self.text)
            # append segment list
            sm = self.get_entity_mention(vm.triggerid).__copy__()  # set the trigger word
            sm.type, sm.stype = sm.stype, None  # move stype to type
            nsnt.smlist.append(sm)
            # append arguments
            for va in vm.arglist:  # for all arguments
                argid = va.id
                if va.type == 'EVENT':  # another valid event
                    if argid not in self.vmdict:  continue
                    argid = self.vmdict[argid].triggerid
                # valid trigger id, or inter-sentence arguments
                if argid not in self.emdict:   continue
                em = self.get_entity_mention(argid).__copy__()
                em.type, em.stype = va.role, va.type  # GENE/EVENT/ENTITY
                nsnt.append_entity_mention(em, gold=True)
            nsnt.sort_entity_mentions()
            nsnts.append(nsnt)
        return nsnts


# used for relation extraction for even role classification
class ve_arg_reDoc(reDoc):
    # generate relation candidates for a sentence
    # triggerID: relation candidates from event argument roles
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
                # An event argument role is between a trigger word and other entitiy
                if not (em1.type == 'TRIG' or em2.type == 'TRIG'):
                    continue
                rm = self.generate_relation_mention_candidate(sent, em1, em2)
                # print()
                # print(rm)
                candidates.append(rm)
        return candidates


# DocSet class for event extraction
class veDocSet(ieDocSet):
    # prepare doc set for biomedical relation extraction
    def prepare_docset_abstract(self, tcfg, Doc, verbose=0):
        self.load_docset_bio_text(Doc, verbose=verbose)
        efilename = '{}/{}.ent'.format(self.wdir, self.id)
        self.load_docset_entity_mentions(efilename, verbose=verbose)
        # read trigger words and optional entities (to be ignored)
        efilename = '{}/{}.trg'.format(self.wdir, self.id)
        self.load_docset_entity_mentions(efilename, verbose=verbose)
        self.check_docset_entity_mentions(verbose=1)
        vfilename = '{}/{}.evt'.format(self.wdir, self.id)
        self.load_docset_event_mentions(vfilename, verbose=verbose)
        self.preprocess_docset_entity_mentions(tcfg)
        self.generate_docset_sentences(tcfg, Doc, verbose=verbose)
        self.collect_docset_instances(tcfg)  # collect sequence labels
        return

    def load_docset_event_mentions(self, vfilename, verbose=0):
        if not os.path.exists(vfilename):  return
        vlines = file_line2array(vfilename, verbose=verbose)
        # 7929104, E7, Localization:T30, Theme:T6, ToLoc:T31
        for vline in vlines:
            pmid, vid, tid = vline[0], vline[1], vline[2].split(':')[-1]
            doc = self.contains_doc(pmid)
            if doc is None:  print('DocPmidErr: {}'.format(pmid))
            # check trigger id
            if tid not in doc.emdict:  print('EntIdErr: {} {}'.format(pmid, tid))
            trg = doc.emlist[doc.emdict[tid]]
            # generate event mention
            vm = EventMention(id=vid, type=trg.stype, triggerid=tid)
            for varg in vline[3:]:
                # Theme:T4
                args = varg.split(':')
                if len(args) == 1:  continue
                type = 'EVENT' if args[1].startswith('E') else doc.emlist[doc.emdict[args[1]]].type
                va = EventArgument(id=args[1], role=args[0], type=type)
                vm.append_argument(va)
            doc.append_event_mention(vm)
        return

    # overwrite temporarily
    # def prepare_docset_dicts_features(self, cfg_dict=None, word_dict=None, verbose=0):
    #     pass

    # convert ve trigger (docset.insts) identification to ner
    # veDocSet.insts have the list of SequenceLabel for all entity mentions, including TRIGGER
    def convert_docset_ve_trg_ner(self, tcfg, DocSet, Doc, task=None, stask=None, verbose=0):
        insts = self.insts
        if verbose:
            print('\nConverting VE trigger identification to NER ...')
            insts = tqdm(insts)
        # docset is an instance of ieDocSet for NER
        docset = self.__copy__(task=task, stask=stask, fmt='i')
        for inst in insts:  # SequenceLabel
            emlist = [em for em in inst.emlist if em.type=='TRIG']
            if not emlist:  continue
            ninst = SequenceLabel(id=inst.id, no=inst.no, text=inst.text, emlist=[], offsets=[])
            for em in inst.emlist:
                # copy trigger words to the new sentence as entity mentions
                if em.type == 'TRIG':
                    nem = em.__copy__()
                    nem.type, nem.stype = em.stype, None
                    ninst.emlist.append(nem)
            docset.insts.append(ninst)
            # print(inst)
            # print(ninst)
        return docset

    # convert docset ve argument role identification to ner
    def convert_docset_ve_arg_ner(self, tcfg, DocSet, Doc, task=None, stask=None, verbose=0):
        docs = self.docdict.items()
        if verbose:
            print('\nConverting VE argument role identification to NER ...')
            docs = tqdm(docs)
        # docset is set to ieDocSet
        docset = DocSet(task, stask, self.wdir, self.id, 'i', self.model_name, tokenizer=self.tokenizer)
        for _, doc in docs:  # doc is veDoc
            for snt in doc.sntlist:  # snt is also veDoc
                # get new sentences with segments and arguments as entity mentions
                nsnts = snt.convert_ve_arguments_entity_mentions(Doc)
                # generate instances from these sentences
                insts = [nsnt.generate_sentence_instances(tcfg)[0] for nsnt in nsnts]
                docset.insts.extend(insts)
        return docset

    # convert docset ve argument identification to re
    def convert_docset_ve_arg_re(self, tcfg, DocSet, Doc, task=None, stask=None, verbose=0):
        docs = self.docdict.items()
        if verbose:
            print('\nConverting VE argument role identification to RE ...')
            docs = tqdm(docs)
        # docset is set to reDocSet
        docset = DocSet(task, stask, self.wdir, self.id, 'i', self.model_name, tokenizer=self.tokenizer)
        for _, doc in docs:  # doc is veDoc
            for snt in doc.sntlist: # snt is veDoc
                # convert snt's event arguments to nsnt's relation mentions, Doc is reDoc
                nsnt = snt.convert_ve_arguments_relation_mentions(Doc)
                # generate relation candidates, redesign
                rmcs = nsnt.generate_sentence_instances(tcfg)
                # arg_cnt = np.array([len(vm.arglist) for _, vm in snt.vmdict.items()]).sum(axis=-1)
                # if len(snt.rmdict) != arg_cnt:
                #     print(snt)
                #     exit(0)
                # print_list(rmcs)
                docset.insts.extend(rmcs)
        return docset

# DocSets class for ve
class veDocSets(ieDocSets):
    # convert veDocSets to ieDocSets for NER
    def convert_docsets_ve_trg_ner(self, tcfg, DocSets, DocSet, Doc, task='ner', stask='ve_trg', verbose=0):
        trgsets = self.__copy__(DocSets, task=task, stask=stask)
        for fileset in self.filesets:
            trgds = fileset.convert_docset_ve_trg_ner(tcfg, DocSet, Doc, task=task, stask=stask, verbose=verbose)
            trgds.prepare_docset_dicts_features(self.cfg_dict, self.word_dict, verbose=verbose)
            trgsets.filesets.append(trgds)
        return trgsets

    # convert veDocSets to reDocSets/ieDocSets for RE/NER
    def convert_docsets_ve_arg(self, tcfg, DocSets, DocSet, Doc, task='re', stask='ve_arg', verbose=0):
        argsets = self.__copy__(DocSets, task=task, stask=stask)
        for fileset in self.filesets:
            if task == 're':
                argds = fileset.convert_docset_ve_arg_re(tcfg, DocSet, Doc, task=task, stask=stask, verbose=verbose)
            else:
                argds = fileset.convert_docset_ve_arg_ner(tcfg, DocSet, Doc, task=task, stask=stask, verbose=verbose)
            argds.prepare_docset_dicts_features(self.cfg_dict, self.word_dict, verbose=verbose)
            argsets.filesets.append(argds)
        return argsets

# convert BioNLP'09 ST-Gene Event to *.txt, *.ent, *.evt
# GE09: *.a1-protein, *.a2-trigger/entity/event, *.txt-title+abstract(two lines)
def convert_ge09_corpus(wdir, cps_file, verbose=0):
    mypath = os.path.join(wdir, cps_file)
    print('\nProcessing {} ...'.format(mypath))
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.a1')]
    # entities, text, events, trigger words and non-core roles
    elines, tlines, vlines, glines = [], [], [], []
    for file in files:  # filelist
        #print(file)
        fid = file[:-3]
        # merge text lines like pmid, title, text
        lines = file_line2list(os.path.join(mypath, fid+'.txt'), stripID=True)
        title_len = len(lines[0])   # used to determine if an entity mention is in the title or not
        tlines.append('{}\t{}'.format(fid, '\t'.join(lines)))
        # merge entities
        lines = file_line2array(os.path.join(mypath, file), sepch='\t', stripID=True)
        slines = []
        # T3\tProtein 134 137\tp65
        for eid, etype, ename in lines:
            sline = [fid, eid]
            tokens = etype.split(' ')
            if tokens[0] == 'Protein':    tokens[0] = 'GENE'
            if int(tokens[1]) >= title_len: # dec position by 1 for those in the abstract
                tokens[1], tokens[2] = str(int(tokens[1])-1), str(int(tokens[2])-1)
            sline.extend(tokens+[ename, 'None'])  # name, link
            slines.append(sline)
        elines.extend(slines)
        # merge events
        vfilename = os.path.join(mypath, fid + '.a2')
        if not os.path.exists(vfilename):  continue
        lines = file_line2array(vfilename, sepch='\t', stripID=True)
        slines = []
        for line in lines:
            if line[0] == '*' or line[0].startswith('M'):  continue # *: entity equivalence，M: negation, speculation?
            sline = [fid, line[0]]
            # trigger-type|Entity spos epos, or Binding:T35 Theme:T14 Theme2:T15
            tokens = line[1].split(' ')
            if line[0].startswith('T'):  # trigger words or non-core entities
                if tokens[0] != 'Entity':   # event type
                    types = tokens[0].split('_')
                    if len(types) >= 2:  type = '{}{}{}'.format(types[0][:2], types[1][0].upper(), types[1][1])
                    else:   type = types[0][:4]
                    tokens[0] = 'TRIGGER|{}'.format(type)
                else:   # trigger or entity
                    tokens[0] = tokens[0].upper()
                if int(tokens[1]) >= title_len:  # dec position by 1 for those in the abstract
                    tokens[1], tokens[2] = str(int(tokens[1]) - 1), str(int(tokens[2]) - 1)
                tokens.extend(line[2:]+['None'])
                sline.extend(tokens)
                glines.append(sline)
            else:
                sline.extend(tokens)
                slines.append(sline)
        vlines.extend(slines)
    # output text
    file_list2line(tlines, '{}{}'.format(mypath, '.txt'), verbose=verbose)
    elines = ['\t'.join(eline) for eline in elines]
    # output entities
    file_list2line(elines, '{}{}'.format(mypath, '.ent'), verbose=verbose)
    # output trigger words and non-entities
    if len(glines) > 0:
        glines = ['\t'.join(gline) for gline in glines]
        file_list2line(glines, '{}{}'.format(mypath, '.trg'), verbose=verbose)
    # output events
    if len(vlines) == 0:  return
    vlines = ['\t'.join(vline) for vline in vlines]
    file_list2line(vlines, '{}{}'.format(mypath, '.evt'), verbose=verbose)
    return


def convert_bio_corpus(wdir, cpsfiles, verbose=0):
    if wdir == 'GE09':
        for cf in cpsfiles:  convert_ge09_corpus(wdir, cf, verbose=verbose)
    return


def review_ve_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=0):
    argtask = 'ner'
    vesets = veDocSets(task, wdir, cpsfiles, cpsfmts)
    vesets.prepare_corpus_filesets(op, tcfg, veDocSet, veDoc, verbose=verbose)
    # entity statistics as NER
    # ccounts = vesets.calculate_docsets_entity_statistics()
    # vesets.output_docsets_entity_statistics(ccounts, logfile='events.cnt')
    # trigger statistics
    # trgsets = vesets.convert_docsets_ve_trg_ner(ieDocSets, ieDocSet, ieDoc, 'ner', 've_trg', verbose=verbose)
    # ccounts = trgsets.calculate_docsets_entity_statistics()
    # trgsets.output_docsets_entity_statistics(ccounts, logfile='triggers.cnt')
    # argument statistics
    if argtask == 're':
        argsets = vesets.convert_docsets_ve_arg(tcfg, reDocSets, reDocSet, ve_arg_reDoc, 're', 've_arg', verbose=verbose)
        crcounts, ccounts = argsets.calculate_docsets_relation_statistics()
        argsets.output_docsets_relation_statistics(crcounts, ccounts, logfile='argument_relations.cnt')
    else:   # ssl-segmented sequence labeling
        argsets = vesets.convert_docsets_ve_arg(tcfg, ieDocSets, sslDocSet, sslDoc, 'ner', 've_arg', verbose=verbose)
        ccounts = argsets.calculate_docsets_entity_statistics()
        argsets.output_docsets_entity_statistics(ccounts, logfile='arguments.cnt')
    return


# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bert_path=None, model_name='LstmCrf', avgmode='micro',
                      epo=0, fold=0, folds=None):
    #
    iesets = create_corpus_filesets(veDocSets, task, wdir, cpsfiles, cpsfmts, model_name, bert_path=bert_path, avgmode=avgmode)

    ve_trg, ve_arg_re= 0, 0
    iesets.prepare_corpus_filesets(op, tcfg, veDocSet, veDoc, verbose=1)
    if ve_trg:
        trgsets = iesets.convert_docsets_ve_trg_ner(tcfg, ieDocSets, ieDocSet, ieDoc, 'ner', 've_trg', verbose=1)
        trgsets.train_eval_docsets(op, tcfg, ieDocSet, epo, fold, folds)
    if ve_arg_re:
        argsets = iesets.convert_docsets_ve_arg(tcfg, reDocSets, reDocSet, ve_arg_reDoc, 're', 've_arg', verbose=1)
        argsets.train_eval_docsets(op, tcfg, reDocSet, epo, fold, folds)
    else:
        argsets = iesets.convert_docsets_ve_arg(tcfg, ieDocSets, sslDocSet, sslDoc, 'ner', 've_arg', verbose=1)
        argsets.train_eval_docsets(op, tcfg, sslDocSet, epo, fold, folds)

    clear_gpu_processes()
    return


def main(op, task, wdir, cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', mdlname='Bert', tcfg=None, epo=0, fold=0, folds=None):
    bert_path = './bert-model/biobert-pubmed-v1.1'

    if 'f' in op:     # format the corpus
        convert_bio_corpus(wdir, cpsfiles, verbose=1)
    elif 'r' in op: # prepare word vocabulary
        review_ve_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=1)
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
        # VE
        main('r', 've', 'GE09', ('train', 'dev'), 'aa', 'Bert', tcfg, epo=0)
        #
