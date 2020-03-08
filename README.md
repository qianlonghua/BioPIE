# BioPIE：Platform for Biomedical Information Extraction
The repository provides the code for BioPIE, a deep learning research platform designed for biomedical text information extraction. 
It includes dataset processing and model training for various information extraction tasks such as named entity recognition, relation extraction and etc.

### data format
For each corpus, there are four kinds of labeling format: instance-level(CORLL2003, semeval2010), sentence-level(BC2GM, BEL), 
abstract-level(NCBI, CPR) and fulltext-level (LINNAEUS). Dataset can be reviewed and processed according to its labeling format. 
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
