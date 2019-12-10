#!/usr/bin/python
# -*- coding: utf-8 -*-
# routines for processing biomedical NER and RE

import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ie_conversion import *
from ie_docsets import *


def review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=0):
    resets = reDocSets(task, wdir, cpsfiles, cpsfmts)
    DocSet = reDocSet if task == 're' else ureDocSet
    Doc = reDoc if task == 're' else ureDoc
    resets.prepare_corpus_filesets(op, tcfg, DocSet, Doc, verbose=verbose)
    # statistics on entities
    levels = ('sent', 'docu')
    resets.create_entity_type_dicts(level=levels[-1])
    ccounts = resets.calculate_docsets_entity_statistics(levels)
    resets.output_docsets_entity_statistics(ccounts, levels, logfile='re_entities.cnt')
    #
    crcounts, ccounts = resets.calculate_docsets_relation_statistics()
    logfile = '{}relations.cnt'.format('u' if task == 'ure' else '')
    resets.output_docsets_relation_statistics(crcounts, ccounts, logfile=logfile)
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    return


def review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=0):
    iesets = ieDocSets(task, wdir, cpsfiles, cpsfmts)
    iesets.prepare_corpus_filesets(op, tcfg, ieDocSet, ieDoc, verbose=verbose)
    #
    ccounts = iesets.calculate_docsets_entity_statistics()
    iesets.output_docsets_entity_statistics(ccounts, logfile='entities.cnt')
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    return


# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bert_path=None, model_name='LstmCrf', avgmode='micro',
                      epo=0, fold=0, folds=None):
    #
    DocSets = ieDocSets
    if task in ('re', 'ure'): DocSets = reDocSets
    iesets = create_corpus_filesets(DocSets, task, wdir, cpsfiles, cpsfmts, model_name, bert_path=bert_path, avgmode=avgmode)

    if task == 'ner':
        iesets.prepare_corpus_filesets(op, tcfg, ieDocSet, ieDoc, verbose=1)
        iesets.train_eval_docsets(op, tcfg, ieDocSet, epo, fold, folds)

    elif task == 're':
        iesets.prepare_corpus_filesets(op, tcfg, reDocSet, reDoc, verbose=1)
        iesets.train_eval_docsets(op, tcfg, reDocSet, epo, fold, folds)

    elif task == 'ure':
        iesets.prepare_corpus_filesets(op, tcfg, ureDocSet, ureDoc, verbose=1)
        iesets.train_eval_docsets(op, tcfg, ureDocSet, epo, fold, folds)

    clear_gpu_processes()
    return


def main(op, task, wdir, cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', mdlname='Bert', tcfg=None, epo=0, fold=0, folds=None):
    bert_path = './bert-model/biobert-pubmed-v1.1'
    if wdir in ('SMV', 'CONLL2003'):  bert_path = './bert-model/bert-base-uncased'
    # two entities with different types are required for some relations
    if wdir in ('CPR', 'CDR'):  tcfg.diff_ent_type = 1
    # sentence simplification for RE/URE/SBEL
    if task in ('re', 'ure'):  tcfg.sent_simplify = 1

    if 'f' in op:     # format the corpus
        convert_bio_corpus(wdir, cpsfiles, verbose=1)
    elif 'r' in op: # prepare word vocabulary
        if task in ('ner'):  # NER
            #prepare_docset_ner_file(func, wdir, 'train', datfmts[0], verbose=1)
            review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=1)
        elif task in ('re', 'ure'):
            review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, verbose=1)
    else:
        avgmode = 'macro' if wdir == 'SMV' else 'micro'
        train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, bert_path=bert_path, model_name=mdlname,
                          avgmode=avgmode, epo=epo, fold=fold, folds=folds)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main('b', '', sys.argv[1])
    else:   # f-format conversion, c-corpus preprocess, t-train, v-validate, p-predict
        elist = ('GENE', 'CHEM', 'DISE', 'BPRO')    # entity types to be replaced/blinded with PLACEHOLDER
        tcfg = TrainConfig(epochs=3, valid_ratio=0, fold_num=10, fold_num_run=1,
                           max_seq_len=100, batch_size=32,
                           bld_ent_types=elist, diff_ent_type=0)
        # RERE
        #main('v', 're', 'SMV', ('train','test'), 'ii', 'Bert', tcfg, epo=0, fold=0, folds=range(3))     # SMV
        #main('v', 're', 'SMV', ('train', 'test'), 'ii', 'Bert', tcfg, epo=0)  # SMV
        #main('tv', 're', 'PPI', ['train'], 'i', 'Bert', tcfg, epo=0)       # PPI
        #main('r', 're', 'CPR', ('train', 'dev', 'test'), 'aaa', 'Bert', tcfg, epo=0)     # CPR, diff_ent_type=1
        #main('r', 're','GAD', ['train'], 'i', 'Bert', tcfg)
        main('r', 'ure', 'BEL', ('train', 'test'), 'ss', 'Bert', tcfg, epo=2)  # BEL
        #main('vp', 're', 'BEL', ('train', 'test'), 'ss', 'Bert', tcfg, epo=2)  # BEL
        #main('r', 're', 'EUADR', ['train'], 'i',  'Bert', tcfg,)
        # NER
        #main('v', 'ner', 'CONLL2003', ('train', 'dev', 'test'), 'iii', 'Bert', tcfg, epo=0)
        #main('tv', 'ner', 'JNLPBA', ('train', 'dev'), 'ii', 'Bert', tcfg, epo=1)
        #main('tv', 'ner', 'BC2GM', ('train', 'test'), 'ss', 'Bert', tcfg, epo=0)
        #main('tv', 'ner', 'CHEMD', ('train', 'dev', 'test'), 'aaa', 'Bert', tcfg, epo=0)
        #main('r', 'ner', 'NCBI', ('train', 'dev', 'test'), 'aaa', 'Bert', tcfg, epo=0)
        #main('r', 'ner', 'CDR', ('train', 'dev', 'test'), 'aaa', 'Bert', tcfg, epo=0)
        #main('v', 'ner', 'CEMP', ('train','dev'), 'aa', 'Bert', tcfg, epo=0)
        #main('v', 'ner', 'GPRO',('train','dev'), 'aa', 'Bert', tcfg, epo=0)
        #main('r', 'ner', 'LINN', ['train'], 'f', 'Bert', tcfg)
        #main('r', 'ner', 'S800', ['train'], 'a', 'Bert', tcfg)
        #
        """ data formats:
        i - instance-level like CoNLL2003, JNLPBA2004
        s - sentence-level like BC2GM, text: sntid, sentence 
        a - abstract level like CPR, text: pmid, title and abstract
        f - full text like LINNAEUS
        """
        """ func list:
        f - format, convert original corpus files to standard files like *.txt, *.ent, *.rel
        r - review the corpus, prepare json config file, combine word vocabularies
        t - train using the whole corpora , including train, dev and test sets.
        v - evaluate on the last file 
        p - predict on the last file, no PRF performance will be reported
        tv - train using the corpora except the last file, evaluate on the last one.
             if only one corpus file exists, cross-validation will be performed. (FOLD_NUM, FOLD_NUM_RUN) 
        tp - train using the corpora except the last file, predict on the last one
        """
        """ model_filename for non Cross-Validation, PS: for CV, these 3 parameters are ignored.
            '{}/{}_{}_e{}_f{}.hdf5'.format(wdir, task, model_name, epo, fold), like SMV/re_Bert_e3_f0.hdf5, or
            '{}/{}_{}_{}_e{}_f{}.hdf5'.format(wdir, stask, task, model_name, epo, fold), like SMV/ve_trg_ner_Bert_e3_f0.hdf5
        epo: valid only for funcs 'vp' to indicate which epoch model to apply, invalid for 't'
        fold: valid for funcs 'tvp' to indicate which fold model to apply, whence the seed is set to fold.
        folds: valid only for 'vp' when ensemble classification is applied to indicate models of folds, invalid for NER and func 't'
        """
