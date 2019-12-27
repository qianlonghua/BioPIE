"""
deal with deep learning models
"""
import os.path
from tqdm import tqdm

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras import backend as K
from keras.optimizers import Adam

from optparse import OptionParser

from ie_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_bert_tokenizer(bert_model_path):
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token_dict = load_word_voc_file(dict_path, verbose=True)
    tokenizer = Tokenizer(token_dict)
    return token_dict, tokenizer


def load_pretrained_embedding_from_file(embed_file, word_dict, EMBED_DIM=100):
    word2idx = {word:word_dict[word] for word in word_dict if word.isupper()}
    #
    lines = file_line2array(embed_file, sepch=' ', verbose=True)
    # filter the lines containing words in word_dict
    wlines = [values for values in lines if values[0] in word_dict]
    # update word dict with these words
    word2idx.update({values[0]:i+len(word2idx) for i, values in enumerate(wlines)})
    # build the embeddings matrix
    embed_matrix = np.zeros((len(word2idx), EMBED_DIM))
    for values in wlines:
        idx = word2idx[values[0]]
        embed_matrix[idx] = np.asarray(values[1:], dtype='float32')
    print('Found {:,} pretrained word vectors.'.format(len(word2idx)))
    iwords = [word for word in sorted(word_dict) if word not in word2idx]
    file_list2line(iwords, 'iwords.txt', verbose=True)
    return word2idx, embed_matrix


# predict on batches with different sequence length
def train_on_batch(model, doc_D, verbose=0, its=''):
    doc_gen = doc_D.__iter__()
    trange = range(doc_D.steps)
    if verbose:
        print('\nTraining {} ...'.format(its))
        trange = tqdm(trange)
    for _ in trange:
        X, Y = next(doc_gen)
        model.train_on_batch(X, Y)
    return

"""  option parameter as a list
def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()
parser.add_option('-f', '--foo',
                  type='string',
                  action='callback',
                  callback=get_comma_separated_args,
                  dest = foo_args_list)
"""

class OptionConfig(OptionParser):
    def __init__(self,
                 model_name='Lstm',     # model name, 'Bert', 'Lstm', 'LstmCrf'
                 epochs=3,
                 valid_ratio=0,
                 verbose=0,
                 fold_num=10,
                 fold_num_run=3,
                 max_seq_len=100,
                 batch_size=32,

                 bert_path = '',
                 avgmode = 'micro',
                 elabel_schema = 'SBIEO',   # set the new labeling schema
                 slabel_schema = 'IO',
                 tokenizer=None,

                 test_ent_pred = 0,         # using predicted test entity file

                 bld_ent_types = None,
                 bld_ent_mode = 0,

                 sent_simplify = 0,     # simplify sentence
                 ent_coref = 0,         # consider entity coreference
                 ent_nest = 0,          # consider nested entities
                 dbl_ent_type = 0,      # consider double typed entity
                 diff_ent_type = 0,
                 num_classes=0,
                 ):
        super(OptionConfig,self).__init__()

        self.add_option('--model_name', dest='model_name', type='str', default=model_name, help='model name')
        self.add_option('--bertID', dest='bertID', default=('Bert' in model_name))
        self.add_option('--epochs', dest='epochs', default=epochs)
        self.add_option('--valid_ratio', dest='valid_ratio', default=valid_ratio)
        self.add_option('--verbose', dest='verbose', default=verbose)

        self.add_option('--fold_num', dest='fold_num', default=fold_num)
        self.add_option('--fold_num_run', dest='fold_num_run', default=fold_num_run)
        self.add_option('--max_seq_len', dest='max_seq_len', default=max_seq_len)
        self.add_option('--batch_size', dest='batch_size', default=batch_size)

        self.add_option('--bert_path', dest='bert_path', default=bert_path)
        self.add_option('--avgmode', dest='avgmode', default=avgmode)
        self.add_option('--elabel_schema', dest='elabel_schema', default=elabel_schema)
        self.add_option('--slabel_schema', dest='slabel_schema', default=slabel_schema)
        self.add_option('--tokenizer', dest='tokenizer', default=tokenizer)

        self.add_option('--test_ent_pred', dest='test_ent_pred', default=test_ent_pred)

        self.add_option('--bld_ent_types', dest='bld_ent_types', default=bld_ent_types)
        self.add_option('--bld_ent_mode', dest='bld_ent_mode', default=bld_ent_mode)    # 0-type, 1-seq, 2-unique id
        self.add_option('--sent_simplify', dest='sent_simplify', default=sent_simplify)
        self.add_option('--ent_coref', dest='ent_coref', default=ent_coref)
        self.add_option('--ent_nest', dest='ent_nest', default=ent_nest)
        self.add_option('--dbl_ent_type', dest='dbl_ent_type', default=dbl_ent_type)

        self.add_option('--diff_ent_type', dest='diff_ent_type', default=diff_ent_type)  # 1-relation instances with different types of entities
        self.add_option('--num_classes', dest='num_classes', default=num_classes)

    def __str__(self):
        cfgs = ['{} = {}'.format(k, v) for k, v in sorted(self.__dict__.items(), key=lambda x:x[0])]
        return '\n'.join(cfgs)


class NerCrfConfig(object):
    def __init__(self,
                 model_name='LstmCrf',
                 vocab_size=1,
                 tag_size=1,
                 max_seq_len=100,
                 embedding_dim=100,
                 lstm_hidden_dim=100,
                 lstm_output_dim=100,
                 recurrent_dropout=0.1,
                 embedding_dropout=0.2,
                 lstm_output_dropout=0.2,
                 optimizer='rmsprop'
                 ):
        self.model_name=model_name
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.recurrent_dropout = recurrent_dropout
        self.embedding_dropout = embedding_dropout
        self.lstm_output_dropout = lstm_output_dropout
        self.optimizer = optimizer

from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

#from sklearn_crfsuite.metrics import flat_classification_report

# https://github.com/apogiatzis/NER-BERT-conll2003/blob/master/BERT_as_Keras_Layer_Example.ipynb
def build_lstm_crf_model(cfg, EMBED_MATRIX=None):
    input = Input(shape=(None,))
    model = Embedding(input_dim=cfg.vocab_size, output_dim=cfg.embedding_dim, weights=[EMBED_MATRIX],
                      input_length=None, mask_zero=True)(input)
    model = Dropout(cfg.embedding_dropout)(model)
    model = Bidirectional(LSTM(units=cfg.lstm_hidden_dim, return_sequences=True,
                               recurrent_dropout=cfg.recurrent_dropout))(model)  # variational biLSTM

    if 'Crf' in cfg.model_name:
        model = TimeDistributed(Dense(cfg.lstm_output_dim, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        output = CRF(cfg.tag_size)(model)  # CRF layer
    else:
        output = TimeDistributed(Dense(cfg.tag_size, activation="softmax"))(model)  # a dense layer as suggested by neuralNer

    # model = Dropout(cfg.lstm_output_dropout)(model)
    model = Model([input], [output])
    if 'Crf' in cfg.model_name:
        model.compile(optimizer=cfg.optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])
    else:
        model.compile(optimizer=cfg.optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def create_bert_sequence_labeling_model(bert_model_path, cfg, verbose=0):
    if verbose:  print('\nCreating {} model ...'.format(cfg.model_name))
    config_path = os.path.join(bert_model_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_model_path, 'bert_model.ckpt')

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers: l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    model = bert_model([x1_in, x2_in])
    if 'Lstm' in cfg.model_name:
        model = Bidirectional(LSTM(units=cfg.lstm_hidden_dim, return_sequences=True,
                                   recurrent_dropout=cfg.recurrent_dropout))(model)  # variational biLSTM

    if 'Crf' in cfg.model_name:
        model = TimeDistributed(Dense(cfg.lstm_output_dim, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        output = CRF(cfg.tag_size)(model)  # CRF layer
    else:   # Bert and BertLstm
        output = TimeDistributed(Dense(cfg.tag_size, activation="softmax"))(model)

    model = Model([x1_in, x2_in], [output])
    if 'Crf' in cfg.model_name:
        model.compile(loss=crf_loss, optimizer=Adam(1e-5), metrics=[crf_viterbi_accuracy])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['categorical_accuracy'])
    model.summary()
    return model


def load_bert_model(bert_path, verbose=0):
    if verbose:
        print('\nLoading the BERT model from {} ...'.format(bert_path))
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    return bert_model

# level: sent, token
def create_bert_classification_model(level='sent', bert_model=None, num_classes=None):

    for l in bert_model.layers: l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    if level == 'token':
        output = TimeDistributed(Dense(num_classes[0], activation="softmax"))(x)
    else:
        x = Lambda(lambda x: x[:, 0])(x)
        output = Dense(num_classes[0], activation='softmax')(x)

    model = Model([x1_in, x2_in, x3_in], [output])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['categorical_accuracy'])
    return model

# level: sent, token
def create_bert_ve_arg_classification_model(bert_model=None, num_classes=0):

    for l in bert_model.layers: l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    # 1-Bind, 2-GeEx, 3-Loca, 4-NeRe, 5-Phos, 6-PoRe, 7-PrCa, 8-Regu, 9-Tran

    x = bert_model([x1_in, x2_in])
    output = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    model = Model([x1_in, x2_in, x3_in], [output])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['categorical_accuracy'])
    return model


