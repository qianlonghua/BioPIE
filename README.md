# BioPIE：Platform for Biomedical Information Extraction
The repository provides the source code for BioPIE, a deep learning (DL)-based research and development platform designed for information extraction, i.e., NER (Named Entity Recognition) and RE (Relation Extraction) in biomedical domain. It can deal with biomedical corpora with different annotation levels in a unified way, train and validate multiple fundamental DL models, such as CNN, LSTM, Att-LSTM, LSTM-CRF, BERT and BERT-CRF etc.

## Corpus formats
For biomedical corpora, there are basically four kinds of annotation levels: instance(CONLL-2003[1], SemEval-2010 task 8[2], which are not biomedical-related), sentence(BC2GM, BEL), abstract(NCBI[3], CPR[4]) and full-text (LINNAEUS). some of them are even mixed with two kinds of annotation levels, such as GE2011, which includes both abstract and full text-level annotations. They can be treated in a unfied way.

#### Instance level
The annotation unit is an instance, e.g. a sentence annotated with entity labels (CONLL-2003, JNLPBA-2004), or a relation instance with a relation type (SemEval-2010).
#### Sentence level
The annotation unit is a sentence, annotated with entity mentions with their starting and ending positions for NER, together with relations between any pair of entity mentions for RE.
#### Abstract level
Similar to sentence-level, except that the annotation unit is an abstract with multiple sentences.
#### Full-text level
Similar to abstract-level, except that the annotation unit is a full-text article with multiple paragraphs, with each paragraph consisting of multiple sentences.

## Application steps
For a specific biomedical corpus, several steps need to be followed in order to apply the platform to your task. First, the corpus is reviewed according to its annotation level and the IE task to be performed, then you can train and validate your  model, finally the derived model can be applied to predict entity mentions or relation mentions from a biomedical literature.

### Review a corpus
According to your task and corpus, the "Review" process makes a config file, a corpus statistic file, and a word dict file. The last one is used for non-BERT models.

### Train and Validate
Currently, NER and RE are two tasks that can be trained and validated on different-level corpora and various DL models. 

Before training and validation, if necessary, a full article is broken into several paragraphs, and a paragraph is splitted into several sentences. Thereafter sentences are tokenized and transformed to instances, such as sequences with continuous entity labels for NER and relation instances with discrete relation types for RE. Specially for RE, the entity mentions in a sentence can be blinded to placeholders and two involved entity mentions can also be marked out in different ways.

For NER, Lstm, LstmCrf[5], Bert, and BertCrf[6] models can be used.
For RE, Cnn[7], Lstm[8], AttLstm[9], and Bert[10] models can be used.

Currently, the performance on the test set at the last epoch is used as the final results. Additionally during each training epoch, the performance on the validation and test set are displayed for reference. In order to improve time efficiency, "Best" and "early stoping" strategies in terms of performance on the validation set will be adopted later on.

### Prediction
The derived model can be used to recognize new entity mentions from a biomedical literature (NER), or to identify the relationship between two recognized entity mentions. In this scenario, no annotations are needed and no performance will be provided.

## Usage and Examples
### Options
Training parameters and data processing options are first initialized using the class of OptionConfig as follows:
```
elist = ('GENE', 'CHEM')
options = OptionConfig(model_name='LstmCrf', epochs=15, 
                       batch_size=32, valid_ratio=0.1, verbose=3,
                       bld_ent_types=elist, diff_ent_type=0, mark_ent_pair=1,
                       )
(tcfg, _) = options.parse_args()
tcfg.word_vector_path = './glove/glove.6B.100d.txt'
tcfg.bert_path = './bert-model/biobert-pubmed-v1.1'
```
Initialized options are then parsed into the variable *tcfg*, which is passed to the main() function. You can also add additional options before they are parsed.

*model_name*, *epochs*, *batch_size*, and *valid_ratio*: they are self-evident and used in training. When *valid_ratio* is set to 0, no validation will be performed; 

*verbose*: if 0, no prompt will be displayed, otherwise prompting messages at different levels will be displayed and saved;

*bld_ent_types*: the list of entity types to be blinded for RE;

*diff_ent_type*: whether a relation instance must involve two entity mentions with different types;

*mark_ent_pair*: how two entity mentions are marked out in RE with 0-no marking, 1-marking with # and @ for 1st and 2nd entity mentions respectively, and 3-marking with entity types; 

*word_vector_path*: the path and filename of the pre-trained word vectors;

*bert_path*: the path of the pre-train BERT model.

### NER
Take the NCBI disease corpus as an example.

#### Review
Invoke the following main() function in bio_ie.py to review the corpus, which is indicated in '**r**' operation:
```
main(op='r', task='ner', wdir='NCBI', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
```
It will make a file "ner_cfg.json" in the work directory "NCBI". The corpus has three text file (*train.txt*, *dev.txt*, *test.txt*) and entity files (*train.ent*, *dev.ent*, *test.ent*), and all the three files are annotated at abstract level (**i**nstance, **a**bstract, **s**entence and **f**ull-text).

#### Train and Validate
The operation of '**tv**' is invoked to train and validate.
```
main(op='tv', task='ner', wdir='NCBI', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
```
The NER model is trained on the first two files and evaluated on the last one.

### RE
Take the BioCreative VI CPR corpus as an example.

#### Review
```
main(op='r', task='re', wdir='CPR', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
```
It will make a file "re_cfg.json" in the work directory "CPR". In addition to three text file and three entity files, the corpus has three relation files (*train.rel*, *dev.rel*, *test.rel*).

#### Train and Validate
```
main(op='tv', task='re', wdir='CPR', cpsfiles=('train', 'dev', 'test'), cpsfmts='aaa', tcfg=tcfg)
```
Similarly, the RE model is trained on the first two files and evaluated on the last one.

### References
[1] Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In CoNLL.

[2] Hendrickx, S.N. Kim, Z. Kozareva, P. Nakov,D.´O S´eaghdha, S. Pad´o, M. Pennacchiotti, L. Ro-mano, and S. Szpakowicz. 2010. Semeval-2010 task8: Multi-way classiﬁcation of semantic relations be-tween pairs of nominals. In Proceedings of the 5thInternational Workshop on Semantic Evaluation.

[3] Dogan,R.I. et al. (2014) NCBI disease corpus: a resource for disease name recognition and concept normalization. J. Biomed. Inform., 47, 1–10.

[4] Taboureau O, Nielsen SK, Audouze K, Weinhold N, Edsgrd D, et al. Chemprot: a disease chemical biology database. Nucleic Acids Res. 2011;39:D367–D372.

[5] Habibi,M. et al. (2017) Deep learning with word embeddings improves biomedical named entity recognition. Bioinformatics, 33, i37–i48.

[6] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2019. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. arXiv:1901.08746 [cs]. ArXiv: 1901.08746.

[7] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. Relation classification via convolutional deep neural network. In Proceedings of COLING, pages 2335–2344.

[8] Shu Zhang, Dequan Zheng, Xinchen Hu, and Ming Yang. 2015. Bidirectional long short-term memory networks for relation classification.

[9] Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. 2016. Attention-based bidirectional long short-term memory networks for relation classification. In The 54th Annual Meeting of the Association for Computational Linguistics, page 207.

[10] Shanchan Wu and Yifan He. 2019. Enriching pretrained language model with entity information for relation classification. CoRR, abs/1905.08284.
