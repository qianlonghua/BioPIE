import random
import os
import copy

from keras.utils import to_categorical

from re_docset import *
from ie_models import *


COUNT_LEVELS = ('inst', 'sent', 'docu')

def set_random_seed(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    #K.set_random_seed(seed_value)

# pad for a batch of sequences for a maximal length
def batch_seq_padding(X, padding=0, max_len=100):
    if not isinstance(X[0], list):  # X is not a list of lists
        return X
    L = [len(x) for x in X]
    ML = max(L)
    if max_len: ML = min(ML, max_len)
    #ML = max_len
    return [np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X]
    #pad_sequences(maxlen=ML, sequences=X, padding='post', value=padding)

# data generator for multi-label multi-class classification
class data_generator:
    def __init__(self, data=None, max_len=120, batch_size=32, num_classes=None):
        self.data = data
        if data is None:  return
        self.max_len = max_len
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.num_features = len(data[0][0])
        self.num_labels = len(data[0][1])
        self.num_classes = num_classes

    def __len__(self):
        return self.steps

    def __iter__(self, shuffleID=False):
        while True:
            idxs = np.arange(len(self.data))
            if shuffleID: np.random.shuffle(idxs)
            X = [[] for _ in range(self.num_features)] # X shape is batch_size*num_features
            Y = [[] for _ in range(self.num_labels)]   # Y shape is batch_size*num_labels
            for i in idxs:
                Xs, Ys = self.data[i]    # [X1,X2],[Y1,Y2]
                # text = d[0][:max_len] # only valid for Chinese
                for j in range(self.num_features):  X[j].append(Xs[j])  # multiple feature like BERT
                for j in range(self.num_labels):    Y[j].append(Ys[j])
                if len(X[0]) == self.batch_size or i == idxs[-1]: # the last one in the batch or the file
                    X = [np.array(batch_seq_padding(X[j], max_len=self.max_len)) for j in range(self.num_features)]
                    Y = [[to_categorical(ys, num_classes=self.num_classes[j]) for ys in batch_seq_padding(Y[j], max_len=self.max_len)]
                         for j in range(self.num_labels)]
                    # print_list(X)
                    # print_list(Y)
                    yield X, Y
                    X = [[] for _ in range(self.num_features)]
                    Y = [[] for _ in range(self.num_labels)]

# predict on batches with different sequence length
def predict_on_batch_keras(model, doc_D, verbose=0):
    doc_gen = doc_D.__iter__()
    pred_cs, pred_ps = [], []
    trange = range(doc_D.steps)
    if verbose:
        print('\nPredicting ...')
        trange = tqdm(trange)
    for _ in trange:
        X, Y = next(doc_gen)
        #print(X, Y)
        pred_p = model.predict_on_batch(X)      # probability for each class
        pred_ps.extend(pred_p)
        pred_c = np.argmax(pred_p, axis=-1)     # class no
        pred_cs.extend(pred_c)
    return pred_cs, pred_ps

def test_with_model(model, model_file, docdata, cfg):
    # Loading
    print('\nLoading {} ...'.format(model_file))
    model.load_weights(model_file)
    # Testing
    doc_D = data_generator(docdata, cfg.max_seq_len, batch_size=cfg.batch_size, num_classes=[cfg.num_classes])
    pred_cs, pred_ps = predict_on_batch_keras(model, doc_D=doc_D, verbose=1)
    return pred_cs, pred_ps


class ieDocSets(object):
    def __init__(self,
                 task = None,
                 wdir = None,
                 cpsfiles = None,
                 cpsfmts = None,
                 model_name = 'Lstm',
                 elabel_schema = 'SBIEO',
                 bert_path = None,
                 stask = None,
                 cfg_dict = None,
                 word_dict=None,
                 tokenizer = None,
                 embed_matrix = None,
                 avgmode = 'micro',
                 ):

        self.task = task
        self.stask = stask
        self.wdir = wdir
        self.cpsfiles = cpsfiles
        self.cpsfmts = cpsfmts

        self.model_name = model_name
        self.bert_path = bert_path

        self.elabel_schema = elabel_schema

        self.cfg_dict = cfg_dict
        self.word_dict = word_dict
        self.tokenizer = tokenizer
        self.embed_matrix = embed_matrix
        self.avgmode = avgmode

        self.filesets = []
        self.datasets = []

    def __str__(self):
        cpsinfo = '{} {} {} {}'.format(self.get_task_fullname(), self.wdir, self.cpsfiles, self.cpsfmts)
        return '{} {} {} {}'.format(cpsinfo, self.model_name, self.bert_path, self.avgmode)

    # different DocSets have the same properties, but different methods
    def __copy__(self, DocSets, task=None, stask=None):
        sets = DocSets()
        for att in self.__dict__.keys():
            if att not in ('filesets', 'datasets'):
                sets.__dict__[att] = self.__dict__[att]
        if task:  sets.task = task
        if stask:  sets.stask = stask
        return sets

    def browse_corpus(self, wdir, cpsfiles, verbose=0):
        pass

    def copy2docset(self, DocSet, cno):
        # cfg_dict and word_dict are processed in get_docset_[re|ner]_dicts_features
        docset = DocSet(task=self.task, wdir=self.wdir, id=self.cpsfiles[cno], fmt=self.cpsfmts[cno], avgmode=self.avgmode,
                          model_name=self.model_name, elabel_schema=self.elabel_schema, tokenizer=self.tokenizer)
        return docset

    # get task fullname
    def get_task_fullname(self):
        if not self.stask: return self.task
        return '{}_{}'.format(self.stask, self.task)

    def get_model_filename(self, epo, fold):
        task = self.task
        if self.stask:  task = '{}_{}'.format(self.stask, task)
        return '{}/{}_{}_e{}_f{}.hdf5'.format(self.wdir, task, self.model_name, epo, fold)

    def create_entity_type_dicts(self, level='inst'):
        for fileset in self.filesets:
            fileset.create_entity_type_dict(level)
        return

    # prepare NER docset with entity type, features/labels, and word vocabulary
    def prepare_corpus_fileset(self, op, tcfg, DocSet, Doc, cno, verbose=0):
        #
        docset = self.copy2docset(DocSet, cno)
        # generate docset
        if self.cpsfmts[cno] == 'i':  # sentence or instance-level
            docset.prepare_docset_instance(op, tcfg, Doc, verbose=verbose)
        elif self.cpsfmts[cno] in 'saf':  # abstract level, include sentence, abstract, fulltext
            docset.prepare_docset_abstract(tcfg, Doc, verbose=verbose)
        # save or set entity type dict
        docset.prepare_docset_dicts_features(self.cfg_dict, self.word_dict, verbose=verbose)
        return docset

    def prepare_corpus_filesets(self, op, tcfg, DocSet, Doc, verbose=0):
        # prepare docset creation functions
        sno = 0 if ('t' in op or 'r' in op) else len(self.cpsfiles)-1
        for i in range(sno, len(self.cpsfiles)):
            fileset = self.prepare_corpus_fileset(op, tcfg, DocSet, Doc, cno=i, verbose=verbose)
            self.filesets.append(fileset)
        return

    # prepare train/dev/test sets
    # return num of classes
    def prepare_tdt_docsets(self, op, DocSet, valid_ratio=0, verbose=0):
        # prepare total data and docsets
        total_data, total_inst, num_classes = [], [], 0
        self.datasets = [DocSet(task=self.task)] * 3
        #
        # the last one for test/predict, others for training
        sno = 0 if 't' in op else len(self.filesets)-1
        for i in range(sno, len(self.filesets)):
            # prepare training/dev/test data
            doc_dat, num_classes = self.filesets[i].get_docset_data(verbose=verbose)
            # when the last one for validation or prediction, the whole fileset is retained
            if i == len(self.filesets)-1 and ('v' in op or 'p' in op):
                self.datasets[2] = self.filesets[i]
                self.datasets[2].extend_data(data=doc_dat)
            elif 't' in op:
                total_inst.extend(self.filesets[i].insts)
                total_data.extend(doc_dat)

        # recombine training and validation data
        if 't' in op:  # generate training data and set
            if valid_ratio == 0:
                # doc_data[0] = total_data
                self.datasets[0] = self.filesets[0].__copy__()  # copy parameters except data
                self.datasets[0].extend_instances(insts=total_inst)
                self.datasets[0].extend_data(data=total_data)
            else:  # validation data
                idxs = np.arange(len(total_data))
                valid_len = int(len(total_data) * valid_ratio)
                np.random.shuffle(idxs)
                # prepare doc_sets
                for i in range(2):
                    self.datasets[i] = self.filesets[0].__copy__()  # copy parameters except data
                    trange = range(valid_len, len(total_data)) if i == 0 else range(valid_len)
                    self.datasets[i].extend_instances(insts=[total_inst[idxs[j]] for j in trange])
                    self.datasets[i].extend_data(data=[total_data[idxs[j]] for j in trange])
        return num_classes


    # train and evaluate docsets
    def train_eval_docsets(self, op, tcfg, DocSet, epo=0, fold=0, folds=None):
        bertID = ('Bert' in self.model_name)
        # cross-validation if only one file exists and operations are training/validation
        cvID = (len(self.cpsfiles) == 1 and 't' in op and 'v' in op)
        if epo == 0 or epo > tcfg.epochs:  epo = tcfg.epochs
        set_random_seed(fold)
        #
        if bertID: tcfg.batch_size = 4    # should be 2 for CHEMD BertCrf, otherwise 4

        # split into train/dev/test docsets
        if cvID:
            tcfg.num_classes = self.prepare_cv_tdt_docsets(tcfg.fold_num, verbose=1)
        else:
            tcfg.num_classes = self.prepare_tdt_docsets(op, DocSet, tcfg.valid_ratio, verbose=1)

        # create models for BERT/LSTM-CRF
        model = create_training_model(self, tcfg, verbose=1)

        # cross-validation
        if cvID:
            self.cv_train_eval_model(model, cfg=tcfg, verbose=1)
            return

        # normal training
        if self.datasets[0].data:    # train
            model_file = self.train_eval_model(model, cfg=tcfg, fold=fold)
        else:     # validation of a specific model
            model_file = self.get_model_filename(epo, fold)

        # validation or predicting
        if not self.datasets[2].data:  return
        if folds is None or 't' in op:
            pred_cs, pred_ps = test_with_model(model, model_file, docdata=self.datasets[2].data, cfg=tcfg)
        else:   # ensemble classification
            pred_cs, pred_ps = self.ensemble_classify(model, docset=self.datasets[2], cfg=tcfg, epo=epo, folds=folds)
        self.datasets[2].evaluate_docset_model(op, pred_classes=pred_cs, mdlfile=model_file, verbose=3)
        return

    # train the model using datasets[0].data, evaluate on the datasets[1,2].data
    def train_eval_model(self, model, cfg, fold):
        model_file, vprfs, tprfs = None, [], []
        for i in range(cfg.epochs):
            print('\nTraining epoch {}/{}'.format(i + 1, cfg.epochs))
            model_file = self.get_model_filename(epo=i+1, fold=fold)
            # training
            doc_D = data_generator(self.datasets[0].data, cfg.max_seq_len, cfg.batch_size, num_classes=[cfg.num_classes])
            model.fit_generator(doc_D.__iter__(shuffleID=True), steps_per_epoch=len(doc_D), epochs=1, verbose=True)
            model.save_weights(model_file)
            # evaluate on validation/test
            for j in range(1, 3):
                if not self.datasets[j].data:  continue
                doc_D = data_generator(self.datasets[j].data, cfg.max_seq_len, cfg.batch_size, num_classes=[cfg.num_classes])
                # loss, acc = model.evaluate_generator(doc_D.__iter__(), steps=len(doc_D))
                # print('{:>6} loss: {:6.4f}, accuracy: {:6.4f}'.format('valid' if j==1 else 'test', loss, acc))
                pred_cs, _ = predict_on_batch_keras(model, doc_D=doc_D, verbose=0)
                set_name = self.datasets[j].id if j == 2 else 'valid'
                self.datasets[j].assign_docset_predicted_results(pred_nos=pred_cs)
                aprf = self.datasets[j].calculate_docset_f1(level='inst', mdlfile=model_file, verbose=0)
                if j == 1:
                    vprfs.append(aprf)  # validation
                else:
                    tprfs.append(aprf)  # test
        # aprfs = np.array(aprfs)
        print()
        for prf in vprfs:  print(format_row_prf(prf))
        print()
        for prf in tprfs:  print(format_row_prf(prf))
        return model_file

    def ensemble_classify(self, model, docset, cfg, epo, folds):
        pred_pt = None
        for foldno in folds:
            model_file = self.get_model_filename(epo, foldno)
            pred_cs, pred_ps = test_with_model(model, model_file, docdata=docset.data, cfg=cfg)
            if pred_pt is None:
                pred_pt = np.array(pred_ps)
            else:
                pred_pt += np.array(pred_ps)
        pred_pt /= len(folds)
        pred_cs = np.argmax(pred_pt, axis=-1)
        return list(pred_cs), list(pred_pt)

    # cross-validation
    # op=='tv', len(doc_files)==1
    def prepare_cv_tdt_docsets(self, fold_num=5, verbose=0):
        # generate_document_sentences
        # fold_no = FOLD_NUM
        self.datasets = [ieDocSet(task=self.task)] * fold_num
        # prepare docset
        doc_dat, num_classes = self.filesets[0].get_docset_data(verbose=verbose)
        doc_inst = self.filesets[0].insts
        # reshuffle data and instances
        idxs = np.arange(len(doc_inst))
        np.random.shuffle(idxs)
        doc_inst = [doc_inst[i] for i in idxs]
        doc_dat = [doc_dat[i] for i in idxs]
        # divide
        fold_len = int(len(doc_inst) / fold_num)
        for i in range(fold_num):
            # the ith training/test data
            self.datasets[i] = self.filesets[0].__copy__()
            self.datasets[i].data.append(doc_dat[:i * fold_len] + doc_dat[(i + 1) * fold_len:])  # 0: training
            self.datasets[i].data.append([])
            self.datasets[i].data.append(doc_dat[i * fold_len:(i + 1) * fold_len])  # 2: testing
            self.datasets[i].extend_instances(doc_inst[i * fold_len:(i + 1) * fold_len])
            #print(len(test_sets[i].data))
        return num_classes

    # train and validate with cross-validation
    def cv_train_eval_model(self, model, cfg, verbose=0):
        #
        epochs, fold_num = cfg.epochs, cfg.fold_num
        old_weights = model.get_weights()
        lfilename = '{}/{}_{}.log'.format(self.wdir, self.datasets[0].id, self.task)
        #
        aprfs = np.zeros([fold_num+2, 6], dtype=float)
        fold_num_valid = fold_num if cfg.fold_num_run == 0 else cfg.fold_num_run
        for i in range(fold_num_valid):
            model.set_weights(old_weights)
            for j in range(epochs):
                #model_file = '{}/{}_{}_e{}_f{}.hdf5'.format(wdir, task, model_name, j + 1, i + 1)
                model_file = self.get_model_filename(epo=j+1, fold=i+1)
                rfilename = '{}/{}_{}_e{}_f{}.rst'.format(self.wdir, self.datasets[0].id, self.task, j+1, i+1)
                #
                if verbose:
                    print('\nTraining epoch {}/{} for fold {}/{} ...'.format(j + 1, epochs, i + 1, fold_num))
                doc_D = data_generator(self.datasets[i].data[0], cfg.max_seq_len, cfg.batch_size,
                                       num_classes=[cfg.num_classes])
                model.fit_generator(doc_D.__iter__(shuffleID=True), steps_per_epoch=len(doc_D), epochs=1, verbose=True)
                model.save_weights(model_file)
                # evaluate on test
                doc_D = data_generator(self.datasets[i].data[2], cfg.max_seq_len, cfg.batch_size,
                                       num_classes=[cfg.num_classes])
                pred_cs, _ = predict_on_batch_keras(model, doc_D=doc_D, verbose=1)
                self.datasets[i].assign_docset_predicted_results(pred_nos=pred_cs)
                aprf = self.datasets[i].calculate_docset_f1(level='inst', mdlfile=model_file, logfile=lfilename,
                                                            rstfile=rfilename, verbose=3)
                aprfs[i] = aprf
        # calculate average and standard deviations
        aprfs[-2, :] = np.average(aprfs[:-2, :], axis=0)
        aprfs[-1, :] = np.std(aprfs[:-2, :], axis=0)
        # output average performance
        flog = open(lfilename, 'a', encoding='utf8')
        print('\nAverage PRF performance across {} folders:'.format(fold_num), file=flog)
        tdict = {str(i): i for i in range(fold_num)}
        tdict.update({'Avg.': fold_num, 'Std.': fold_num + 1})
        output_classification_prfs(aprfs, tdict, flog=flog, verbose=1)
        flog.close()
        return

    def calculate_docsets_entity_statistics(self, levels=COUNT_LEVELS):
        ccounts = []
        for fileset in self.filesets:
            lcounts = []
            for level in levels:  # exclude the level -- 'inst'
                lcounts.append(fileset.generate_docset_entity_statistics(level))
                if fileset.fmt == 'i':  break  # 'saf': sent-level/docu-level for sentence, abstract, full-text
            ccounts.append(lcounts)
        return ccounts

    # output relation statistics from one aspect: 'level' or 'file'
    def output_aspect_statistics(self, aspect, object, typedict, levels, ccounts, flog):
        num_of_cols, num_of_loops = ccounts.shape[:2]  # num of files, num of levels
        if aspect == 'file': num_of_loops, num_of_cols = ccounts.shape[:2]
        totalID = (aspect == 'level' and num_of_cols > 1)
        for i in range(num_of_loops):
            # num of entity/relation types, num of files plus one total
            icounts = np.zeros([len(typedict), num_of_cols + (1 if totalID else 0)], dtype=int)
            for j in range(num_of_cols):
                icounts[:, j] = ccounts[j, i] if aspect == 'level' else ccounts[i, j]
            # totol on files
            if totalID:  icounts[:, -1] = np.sum(icounts[:, :-1], axis=-1)
            # calculate on reverse relations
            cols = [['TYPE', 6, 0]]
            cols.extend([[(self.filesets[j].id if aspect == 'level' else levels[j]), 6, 0] for j in range(num_of_cols)])
            if totalID:  cols.append(['total', 6, 0])
            rows = [k for k, v in sorted(typedict.items(), key=lambda x: x[1])]
            lines = output_matrix(icounts, cols=cols, rows=rows)
            # output relation statistics
            sdt = datetime.now().strftime("%y-%m-%d, %H:%M:%S")
            tname = (levels[i] if aspect == 'level' else self.filesets[i].id)
            print('\n{} {} ({}), {}'.format(self.wdir, object, tname, sdt), file=flog)
            print('\n'.join(lines), file=flog)
        return

    # counts: [[d1[l1], [l2]], [d2[l1], [l2], ...], ...]
    def output_docsets_entity_statistics(self, ccounts, levels=COUNT_LEVELS, logfile=None):
        wdir, etypedict = self.wdir, self.filesets[0].etypedict
        flog = sys.stdout
        if logfile:  flog = open(os.path.join(wdir, logfile), 'a', encoding='utf8')
        # 3*2*14
        ccounts = np.array(ccounts)  # num of files, num of levels, num of types
        # aspect with different levels, total on files
        if ccounts.shape[0] > 1 or ccounts.shape[1] == 1:  # multiple files or only one level
            self.output_aspect_statistics('level', 'entities', etypedict, levels, ccounts, flog)
        # aspect with different files
        if ccounts.shape[1] > 1:  # multiple levels
            self.output_aspect_statistics('file', 'entities', etypedict, levels, ccounts, flog)
        if logfile:  flog.close()
        return


# DocSets class for RE
class reDocSets(ieDocSets):
    def create_relation_type_dicts(self):
        for fileset in self.filesets:
            fileset.create_reverse_relation_type_dict()
            fileset.create_non_reverse_relation_type_dict()
        return
    #
    def calculate_docsets_relation_statistics(self, levels=COUNT_LEVELS):
        crcounts, ccounts = [], []
        for fileset in self.filesets:
            lrcounts, lcounts = [], []
            for level in levels:
                rcounts, counts = fileset.generate_docset_relation_statistics(level=level)
                batch_list_append(lrcounts, rcounts, lcounts, counts)
                if fileset.fmt == 'i':  break  # 'saf': sent-level/docu-level for sentence, abstract, full-text
            batch_list_append(crcounts, lrcounts, ccounts, lcounts)
        return crcounts, ccounts

    # counts: [[d1[l1], [l2]], [d2[l1], [l2], ...], ...]
    def output_docsets_relation_statistics(self, crcounts, ccounts, levels=COUNT_LEVELS, logfile=None):
        rrtypedict, rtypedict, rvsID = self.filesets[0].rrtypedict, self.filesets[0].rtypedict, self.filesets[0].rvsID
        flog = sys.stdout
        if logfile:  flog = open(os.path.join(self.wdir, logfile), 'a', encoding='utf8')
        #
        crcounts = np.array(crcounts)  # num of files, num of levels, num of types
        ccounts = np.array(ccounts)  # 3*2*14
        # aspect with different levels, total on files
        if crcounts.shape[0] > 1 or crcounts.shape[1] == 1:  # multiple files or only one level
            self.output_aspect_statistics('level', 'reverse relations', rrtypedict, levels, crcounts, flog)
            if rvsID:
                self.output_aspect_statistics('level', 'non-reverse relations', rtypedict, levels, ccounts, flog)
        # aspect with different files
        if crcounts.shape[1] > 1:  # multiple levels
            self.output_aspect_statistics('file', 'reverse relations', rrtypedict, levels, crcounts, flog)
            if rvsID:
                self.output_aspect_statistics('file', 'non-reverse relations', rtypedict, levels, ccounts, flog)
        if logfile:  flog.close()
        return


def create_corpus_filesets(DocSets, task, wdir, cpsfiles, cpsfmts, model_name,
                           bert_path=None, avgmode='micro'):
    iesets = DocSets(task, wdir, cpsfiles, cpsfmts, model_name, bert_path=bert_path, avgmode=avgmode)
    # specific task config: elabelschema, etypedict, elabeldict for NER
    iesets.cfg_dict = load_json_file(os.path.join(wdir, '{}_cfg.json'.format(task)))
    word_dict = load_word_voc_file(os.path.join(wdir, '{}_voc.txt'.format(task)))   # default for LSTM-CRF
    #
    if 'Bert' in model_name:   # for BERT model
        word_dict, iesets.tokenizer = load_bert_tokenizer(bert_path)
    else:   # non-bert model, using pre-trained word vectors
        pretrained_filename = 'glove.6B.100d.txt'
        word_dict, iesets.embed_matrix = load_pretrained_embedding_from_file(pretrained_filename, word_dict, EMBED_DIM=100)
    iesets.word_dict = word_dict
    return iesets

