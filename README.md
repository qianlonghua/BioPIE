# BioPIE：Platform for Biomedical Information Extraction
The repository provides the source code for BioPIE, a deep learning (DL)-based research and development platform designed for information extraction, i.e., NER (Named Entity Recognition) and RE (Relation Extraction) in biomedical domain. It can deal with biomedical corpora with different annotation levels in a unified way, train and validate multiple fundamental DL models, such as CNN, LSTM, Att-LSTM, LSTM-CRF, BERT and BERT-CRF etc.

## Corpus formats
For biomedical corpora, there are basically four kinds of annotation levels: instance(CONLL-2003, SemEval-2010, oooops, they are not biomedical-related), sentence(BC2GM, BEL), abstract(NCBI, CPR) and full-text (LINNAEUS). some of them are even mixed with two kinds of annotation levels, such as GE2011, which includes both abstract and full text-level annotations. They can be treated in a unfied way.

#### Instance level
The annotation unit is an instance, e.g. a sentence annotated with entity labels (CONLL-2003, JNLPBA-2004), or a relation instance with a relation type (SemEval-2010).

#### Sentence level
The annotation unit is a sentence, annotated with entity mentions with their starting and ending positions for NER, together with relations between any pair of entity mentions for RE.

#### Abstract level
Similar to sentence-level, in addition that the annotation unit is an abstract with multiple sentences.

#### Full-text level
Similar to abstract-level, in addition that the annotation unit is a full-text article with multiple paragraphs, with each paragraph consisting of multiple sentences.

## Application steps
For a specific biomedical corpus, several steps need to be followed in order to apply the platform to your task. First, the corpus is reviewed according to its annotation level and the IE task to be performed, then you can train and validate your  model, finally the derived model can be applied to predict entity mentions or relation mentions from a biomedical literature.

### Review a corpus
According to your task and corpus, the "Review" process makes a config file, a corpus statistic file, and a word dict file. The last one is used for non-BERT models.

### Train and Validate
Currently, NER and RE are two tasks that can be trained and validated on different-level corpora and various DL models. 

Before training and validation, if necessary, a full article is broken into several paragraphs, and a paragraph is splitted into several sentences. Thereafter sentences are tokenized and transformed to instances, such as sequences with continuous entity labels for NER and relation instances with discrete relation types for RE. Specially for RE, the entity mentions in a sentence can be blinded to placeholders and two involved entity mentions can also be marked out in different ways.

For NER, Lstm, LstmCrf, Bert, and BertCrf models can be used.
For RE, Cnn, Lstm, AttLstm, and Bert models can be used.

Currently, 
Best model are chosen according to performance on validation set. Also, early stopping can be applied to improve time efficiency.

### prediction
The final prediction result is the performance(f1) of the model on test set. 
There are two kinds of performance, instance-level and abstract-level.

### usage and example
```shell
options = OptionConfig(model_name='LstmCrf', epochs=15, batch_size=32,
                       valid_ratio=0.1, verbose=3,
                       fold_num=10, fold_num_run=1,
                       bld_ent_types=elist, diff_ent_type=0, mark_ent_pair=1,
                       case_sensitive=0, test_ent_pred=0)
(tcfg, _) = options.parse_args()
```
OptionConfig can pass some option parser during the training procedure: 
model_name, epochs and batch_size are model, epoches and batch size used in training; 

valid_ratio is the percentage used when randomly split data as validation set;

fold_num us the number of folder used in cross validation.
```shell
main(op='r', task='ner', wdir='NCBI', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
main(op='tv', task='ner', wdir='NCBI', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
```
op is the operation used in the function.

r: review operation can review the corpus files, prepare json config file and combine word vocabularies.

tv: train and validate operation can train on the the corpora except the last file and evaluate on the last one.

task is the biomedical information extraction task: ner or re.

cpsfiles is the corpus file used in the operation.

cpsfmts is the corpus formats: i for instance level; a for abstract level; s for sentence level; f for full text level.
