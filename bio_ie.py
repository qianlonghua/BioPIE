#!/usr/bin/python
# -*- coding: utf-8 -*-
# routines for processing biomedical NER and RE

from ie_conversion import *
from ie_docsets import *


def review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    resets = reDocSets(task, wdir, cpsfiles, cpsfmts)
    DocSet = reDocSet if task == 're' else ureDocSet
    Doc = reDoc if task == 're' else ureDoc
    resets.prepare_corpus_filesets(op, tcfg, DocSet, Doc)
    # statistics on entities
    # levels = ('sent', 'docu')
    # resets.create_entity_type_dicts(level=levels[-1])
    # ccounts = resets.calculate_docsets_entity_mention_statistics(levels)
    # typedict = resets.filesets[0].etypedict
    # resets.output_docsets_instance_statistics(ccounts, 'Entity Mentions', levels=levels, logfile='reEntityMentions.cnt', typedict=typedict)
    #
    ccounts = resets.calculate_docsets_instance_statistics()
    logfile = '{}RelationMentions.cnt'.format('u' if task == 'ure' else '')
    resets.output_docsets_instance_statistics(ccounts, 'Relation Mentions', logfile=logfile)
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    return


def review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    iesets = ieDocSets(task, wdir, cpsfiles, cpsfmts)
    iesets.prepare_corpus_filesets(op, tcfg, ieDocSet, ieDoc)
    #
    ccounts = iesets.calculate_docsets_instance_statistics()
    iesets.output_docsets_instance_statistics(ccounts, 'Entity Mentions', logfile='EntityMentions.cnt')
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    return


# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, epo=0, fold=0, folds=None):

    DocSets = ieDocSets
    if task in ('re', 'ure'): DocSets = reDocSets
    iesets = create_corpus_filesets(tcfg, DocSets, task, wdir, cpsfiles, cpsfmts)
    #print(tcfg)

    if task == 'ner':
        iesets.prepare_corpus_filesets(op, tcfg, ieDocSet, ieDoc)
        iesets.train_eval_docsets(op, tcfg, ieDocSet, epo, fold, folds)

    elif task == 're':
        iesets.prepare_corpus_filesets(op, tcfg, reDocSet, reDoc)
        iesets.train_eval_docsets(op, tcfg, reDocSet, epo, fold, folds)

    elif task == 'ure':
        iesets.prepare_corpus_filesets(op, tcfg, ureDocSet, ureDoc)
        iesets.train_eval_docsets(op, tcfg, ureDocSet, epo, fold, folds)

    clear_gpu_processes()
    return


def main(op, task, wdir, cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=None, epo=0, fold=0, folds=None):
    tcfg.bert_path = './bert-model/biobert-pubmed-v1.1'
    if wdir in ('SMV', 'CONLL2003'):  tcfg.bert_path = './bert-model/bert-base-uncased'
    # two entities with different types are required for some relations
    if wdir in ('CPR', 'CDR'):  tcfg.diff_ent_type = 1
    # sentence simplification for RE/URE/SBEL
    if task in ('re', 'ure'):  tcfg.sent_simplify = 1

    if 'f' in op:     # format the corpus
        convert_bio_corpus(wdir, cpsfiles, verbose=tcfg.verbose)
    elif 'r' in op: # prepare word vocabulary
        tcfg.model_name, tcfg.bertID = 'Lstm', False    # default model name for review
        if task in ('ner'):  # NER
            #prepare_docset_ner_file(func, wdir, 'train', datfmts[0], verbose=1)
            review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts)
        elif task in ('re', 'ure'):
            review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts)
    else:
        if wdir == 'SMV': tcfg.avgmode = 'macro'
        train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, epo, fold, folds)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main('b', '', sys.argv[1])
    else:   # f-format conversion, c-corpus preprocess, t-train, v-validate, p-predict
        elist = ('GENE', 'CHEM', 'DISE', 'BPRO')    # entity types to be replaced/blinded with PLACEHOLDER
        options = OptionConfig(model_name='Bert', epochs=3,
                               valid_ratio=0, verbose=3,
                               fold_num=10, fold_num_run=1,
                               bld_ent_types=elist, diff_ent_type=0)
        (tcfg, _) = options.parse_args()
        # RE
        main('tv', 're', 'CPR', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)     # CPR, diff_ent_type=1
        # NER
        main('tv', 'ner', 'CONLL2003', ('train', 'dev', 'test'), 'iii', tcfg, epo=0)
        #
        """ data formats:
        i - instance-level like CoNLL2003, JNLPBA2004
        s - sentence-level like BC2GM, text: sntid, sentence 
        a - abstract-level like CPR, text: pmid, title and abstract
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
