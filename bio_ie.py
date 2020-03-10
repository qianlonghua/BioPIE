#!/usr/bin/python
# -*- coding: utf-8 -*-
# routines for processing biomedical NER and RE
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ie_conversion import *
from re_utils.re_docsets import *
from re_utils.re_docset import *
#from el_utils import *
from tensorflow import set_random_seed


def set_my_random_seed(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    set_random_seed(seed_value)


def review_re_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    resets = reDocSets(task, wdir, cpsfiles, cpsfmts)
    #
    DocSet = reDocSet if task == 're' else ureDocSet
    Doc = reDoc if task == 're' else ureDoc
    resets.prepare_corpus_filesets(op, tcfg, DocSet, Doc)
    # statistics on entities
    levels = ('sent', 'docu')
    resets.create_entity_type_dicts(level=levels[-1])
    ccounts = resets.calculate_docsets_entity_mention_statistics(levels)
    etypedict = resets.filesets[0].etypedict
    resets.output_docsets_instance_statistics(ccounts, 'Entity Mentions', levels=levels, logfile='reEntityMentions.cnt', typedict=etypedict)
    # statistics on relation mention
    ccounts = resets.calculate_docsets_instance_statistics()
    logfile = '{}RelationMentions.cnt'.format('u' if task == 'ure' else '')
    resets.output_docsets_instance_statistics(ccounts, 'Relation Mentions', logfile=logfile)
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    # output the test instances
    resets.filesets[0].print_docset_instances(filename=wdir + '/test_rel.txt', level='docu', verbose=tcfg.verbose)
    return


def review_ner_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts):
    nersets = nerDocSets(task, wdir, cpsfiles, cpsfmts)
    nersets.prepare_corpus_filesets(op, tcfg, nerDocSet, nerDoc)
    # statistics on entity mentions
    ccounts = nersets.calculate_docsets_instance_statistics()
    nersets.output_docsets_instance_statistics(ccounts, 'Entity Mentions', logfile='EntityMentions.cnt')
    combine_word_voc_files(wdir, ['{}_{}_voc.txt'.format(task, cf) for cf in cpsfiles], '{}_voc.txt'.format(task), verbose=True)
    #
    nersets.filesets[0].print_docset_instances(filename=wdir+'/test_ent.txt', level='docu', verbose=tcfg.verbose)
    return


# train, evaluate and predict NER on a corpus
# op: t-train with training files, v-validate [-1], p-predict [-1]
def train_eval_corpus(op, tcfg, task, wdir, cpsfiles, cpsfmts, epo=0, fold=0, folds=None):
    set_my_random_seed(fold)
    DocSets = nerDocSets
    if task in ('re', 'ure'): DocSets = reDocSets
    iesets = create_corpus_filesets(tcfg, DocSets, task, wdir, cpsfiles, cpsfmts)
    #print(tcfg)

    if task == 'ner':
        iesets.prepare_corpus_filesets(op, tcfg, nerDocSet, nerDoc)
        iesets.train_eval_docsets(op, tcfg, nerDocSet, epo, fold, folds)

    elif task == 're':
        iesets.prepare_corpus_filesets(op, tcfg, reDocSet, reDoc)
        iesets.train_eval_docsets(op, tcfg, reDocSet, epo, fold, folds)

    elif task == 'ure':
        iesets.prepare_corpus_filesets(op, tcfg, ureDocSet, ureDoc)
        iesets.train_eval_docsets(op, tcfg, ureDocSet, epo, fold, folds)

    elif task == 'el':  # entity linking
        iesets.prepare_corpus_filesets(op, tcfg, elDocSet, nerDoc)

    clear_gpu_processes()
    return


def main(op, task, wdir, cpsfiles, cpsfmts, tcfg, epo=0, fold=0, folds=None):
    tcfg.word_vector_path = './glove/glove.6B.100d.txt'
    tcfg.bert_path = './bert-model/biobert-pubmed-v1.1'
    if wdir in ('SMV', 'CONLL2003'):  tcfg.bert_path = './bert-model/bert-base-uncased'
    # predicted file suffix
    tcfg.pred_file_suff = 'ent' if task == 'ner' else 'rel' if task == 're' else 'urel'
    # two entities with different types are required for some relations
    if wdir in ('CPR', 'CDR'):  tcfg.diff_ent_type = 1
    # sentence simplification for RE/URE
    if task in ('re', 'ure'):  tcfg.sent_simplify = 1
    #
    if tcfg.bertID: tcfg.batch_size = 2 if 'Crf' in tcfg.model_name else 4

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
    elist = ('GENE', 'CHEM', 'DISE', 'BPRO')    # entity types to be replaced/blinded with PLACEHOLDERS
    options = OptionConfig(model_name='Bert', epochs=20, batch_size=32,
                           valid_ratio=0.1, verbose=2,
                           fold_num=10, fold_num_run=1,
                           bld_ent_types=elist, diff_ent_type=0, mark_ent_pair=1,
                           case_sensitive=0, test_ent_pred=0, elabel_schema='BIEO')
    (tcfg, _) = options.parse_args()
    # NER
    #main('tv', 'ner', 'CONLL2003', ('dev', 'test'), 'ii', tcfg, epo=0)
    #main('tv', 'ner', 'CONLL2003', ('train', 'dev', 'test'), 'iii', tcfg, epo=0)
    #main('tv', 'ner', 'JNLPBA', ('train', 'dev'), 'ii', tcfg, epo=1)
    #main('tv', 'ner', 'BC2GM', ('train', 'test'), 'ss', tcfg, epo=0)
    #main('r', 'ner', 'CHEMD', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)
    main('tv', 'ner', 'NCBI', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)
    #main('r', 'ner', 'CDR', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)
    #main('v', 'ner', 'CEMP', ('train','dev'), 'aa', 'Bert', epo=0)
    #main('v', 'ner', 'GPRO',('train','dev'), 'aa', tcfg, epo=0)
    #main('r', 'ner', 'LINN', ['train'], 'f', tcfg)
    #main('tv', 'ner', 'S800', ['train'], 'a', tcfg)
    #main('vp', 'ner', 'CPR', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)
    # RE
    #main('v', 're', 'SMV', ('train','test'), 'ii', tcfg, epo=0, fold=0, folds=range(3))     # SMV
    #main('v', 're', 'SMV', ('train', 'test'), 'ii', tcfg, epo=10)  # SMV, mark_ent_pair=0
    #main('tv', 're', 'PPI', ['train'], 'i', tcfg, epo=0)       # PPI
    #main('tv', 're', 'CPR', ('train', 'dev', 'test'), 'aaa', tcfg, epo=0)     # CPR, diff_ent_type=1
    #main('r', 're','GAD', ['train'], 'i',  tcfg)
    #main('tv', 'ure', 'BEL', ('train', 'test'), 'ss', tcfg, epo=0)  # BEL
    #main('vp', 're', 'BEL', ('train', 'test'), 'ss', tcfg, epo=1)  # BEL
    #main('r', 're', 'EUADR', ['train'], 'i', tcfg,)
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
