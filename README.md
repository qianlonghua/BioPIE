# BioPIE：Platform for Biomedical Information Extraction
The repository provides the source code for BioPIE, a deep learning (DL)-based research and development platform designed for information extraction, i.e., NER (Named Entity Recognition) and RE (Relation Extraction) in biomedical domain. It can deal with biomedical corpora with different annotation levels in a unified way, train and validate multiple fundamental DL models, such as CNN, LSTM, Att-LSTM, LSTM-CRF, BERT and BERT-CRF etc.

## corpus format
For biomedical corpora, there are basically four kinds of annotation levels: instance(CONLL-2003, SemEval-2010, oooops, they are not biomedical-related), sentence(BC2GM, BEL), abstract(NCBI, CPR) and full-text (LINNAEUS). some of them are even mixed with two kinds of annotation levels, such as GE2011, which includes both abstract and full text-level annotations.
Dataset can be reviewed and processed according to its labeling format. 
For different corpus, task config file, corpus statistic file and word dict file will be saved for further usage.

### train and validate
Named entity recognition and relation extraction are two default task that can be trained on different corpus and different models. 

For ner task, sentences are tokenized and each token is labeled with entity type in a sequence level. 
Data are trained in token level using Lstm, Lstm Crf and Bert model.

For re task, relation pair are blinded and each instance are labeled according to relation type. 
Data are trained in sentence level using Cnn, Lstm, Attention Lstm and Bert model.

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
