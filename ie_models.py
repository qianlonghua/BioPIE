"""
deal with deep learning models
"""
from tqdm import tqdm

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras_attention import SelfAttention
import keras.backend as K

from optparse import OptionParser

from ie_utils import *


class piecewise_maxpool_layer(Layer):
    def __init__(self, filter_num, seq_len, **kwargs):
        self.filter_num = filter_num
        self.seq_len = seq_len
        super(piecewise_maxpool_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(piecewise_maxpool_layer, self).build(input_shape)

    def compute_mask(self, x, mask=None):
        return None

    def max_pool_piece1(self, x):
        conv_output = x[0]
        e1 = K.cast(x[1], 'int32')
        piece = K.slice(conv_output, [0, 0], [e1, self.filter_num])
        return K.max(piece, 0)

    def max_pool_piece2(self, x):
        conv_output = x[0]
        e1 = K.cast(x[1], 'int32')
        e2 = K.cast(x[2], 'int32')
        piece = K.slice(conv_output, [e1, 0], [e2 - e1, self.filter_num])
        return K.max(piece, 0)

    def max_pool_piece3(self, x):
        conv_output = x[0]
        # ~ e2 = tf.to_int32(x[1])
        e2 = K.cast(x[1], 'int32')
        piece = K.slice(conv_output, [e2, 0], [self.seq_len - e2, self.filter_num])
        return K.max(piece, 0)

    def call(self, inputs, **kwargs):
        assert (len(inputs) == 2)
        conv_output = inputs[0]
        e1 = inputs[1][:, 1]
        e2 = inputs[1][:, 3]
        conv_piece1 = K.map_fn(self.max_pool_piece1, (conv_output, e1), dtype='float32')
        conv_piece2 = K.map_fn(self.max_pool_piece2, (conv_output, e1, e2), dtype='float32')
        conv_piece3 = K.map_fn(self.max_pool_piece3, (conv_output, e2), dtype='float32')
        return K.concatenate([conv_piece1, conv_piece2, conv_piece3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2] * 3)

def Squeeze(x):
    return K.squeeze(x, axis=1)

def Max(x):
    return K.max(x, axis=1)

def load_bert_tokenizer(bert_model_path, verbose=0):
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token_dict = load_word_voc_file(dict_path, verbose)
    tokenizer = Tokenizer(token_dict)
    return token_dict, tokenizer


def load_pretrained_embedding_from_file(tcfg, word_dict):
    """
    initialize embedding matrix with pre-trained vectors from a file
    set to zeros for those not in pre-trained, is random initialization helpful?
    :param tcfg:
    :param word_dict:
    :return: word2idx, embed_matrix
    """
    # initialize word2idx
    word2idx = {word:word_dict[word] for word in word_dict if word.isupper()}
    # load pre-trained vector file
    lines = file_line2array(tcfg.word_vector_path, sepch=' ', verbose=tcfg.verbose)
    # filter the lines containing words in word_dict
    wlines = [values for values in lines if values[0] in word_dict]
    # update word dict with these words
    word2idx.update({values[0]:i+len(word2idx) for i, values in enumerate(wlines)})
    # initialize the embedding matrix
    #embed_matrix = np.zeros((len(word2idx), tcfg.embedding_dim))
    embed_matrix = np.random.normal(0, np.sqrt(3 / tcfg.embedding_dim), (len(word2idx), tcfg.embedding_dim))
    for values in wlines:
        idx = word2idx[values[0]]
        embed_matrix[idx] = np.asarray(values[1:], dtype='float32')
    #
    print('Found {:,} pretrained word vectors.'.format(len(word2idx)))
    iwords = [word for word in sorted(word_dict) if word not in word2idx]
    file_list2line(iwords, 'iwords.txt', verbose=True)
    return word2idx, embed_matrix


def load_pretrained_embedding_from_file_old(tcfg, word_dict):
    #
    lines = file_line2array(tcfg.word_vector_path, sepch=' ', verbose=tcfg.verbose)
    # filter the lines containing words in word_dict
    wlines = [values for values in lines if values[0] in word_dict]
    # build the embeddings matrix
    scope = np.sqrt(3 / tcfg.embedding_dim)
    embed_matrix = np.random.normal(0, scope, (len(word_dict), tcfg.embedding_dim))
    for values in wlines:
        idx = word_dict[values[0]]
        embed_matrix[idx] = np.asarray(values[1:], dtype='float32')
    print('Found {:,} pretrained word vectors.'.format(len(word_dict)))
    iwords = [word for word in sorted(word_dict) if word not in word_dict]
    file_list2line(iwords, 'iwords.txt', verbose=True)
    return embed_matrix


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
                 max_seq_len=128,
                 batch_size=32,

                 bert_path = '',            # for BERT model
                 tokenizer=None,

                 avgmode = 'micro',
                 elabel_schema = 'SBIEO',   # set the new labeling schema
                 slabel_schema = 'IO',

                 test_ent_pred = 0,         # using predicted test entity file
                 pred_file_suff = 'ent',    # predicted file suffix

                 bld_ent_types = None,
                 bld_ent_mode = 0,
                 mark_ent_pair = 0,

                 case_sensitive = 0,    # case-sensitive for text in instances, such as SequenceLabel etc.
                 sent_simplify = 0,     # simplify sentence
                 ent_coref = 0,         # consider entity coreference
                 ent_nest = 0,          # consider nested entities
                 dbl_ent_type = 0,      # consider double typed entity
                 diff_ent_type = 0,
                 num_classes=0,

                 word_vector_path='',  # pre-trained word vector path
                 vocab_size = 0,
                 embedding_dim = 100,
                 lstm_hidden_dim = 100,
                 lstm_output_dim = 50,
                 embedding_dropout=0.2,
                 lstm_recur_dropout = 0.1,
                 lstm_output_dropout = 0.2,
                 optimizer='rmsprop'
                 ):
        super(OptionConfig,self).__init__()

        self.add_option('--model_name', dest='model_name', type='str', default=model_name, help='model name')
        self.add_option('--bertID', dest='bertID', default=('Bert' in model_name))
        self.add_option('--bert_model', dest='bert_model', default=None)
        self.add_option('--epochs', dest='epochs', default=epochs)
        self.add_option('--valid_ratio', dest='valid_ratio', default=valid_ratio)
        self.add_option('--verbose', dest='verbose', default=verbose)

        self.add_option('--fold_num', dest='fold_num', default=fold_num)
        self.add_option('--fold_num_run', dest='fold_num_run', default=fold_num_run)
        self.add_option('--max_seq_len', dest='max_seq_len', default=max_seq_len)
        self.add_option('--batch_size', dest='batch_size', default=batch_size)

        self.add_option('--bert_path', dest='bert_path', default=bert_path)
        self.add_option('--tokenizer', dest='tokenizer', default=tokenizer)
        self.add_option('--avgmode', dest='avgmode', default=avgmode)
        self.add_option('--elabel_schema', dest='elabel_schema', default=elabel_schema)
        self.add_option('--slabel_schema', dest='slabel_schema', default=slabel_schema)

        self.add_option('--test_ent_pred', dest='test_ent_pred', default=test_ent_pred)
        self.add_option('--pred_file_suff', dest='pred_file_suff', default=pred_file_suff)

        self.add_option('--bld_ent_types', dest='bld_ent_types', default=bld_ent_types) # list of entity types to be blinded
        self.add_option('--bld_ent_mode', dest='bld_ent_mode', default=bld_ent_mode)    # 0-type, 1-seq, 2-unique id
        # how to mark entity mention pair in a relation instance, 0-None, 1-#@ around, like # E1 #, @ E2 @, 2-entity type
        self.add_option('--mark_ent_pair', dest='mark_ent_pair', default=mark_ent_pair)

        self.add_option('--case_sensitive', dest='case_sensitive', default=case_sensitive)
        self.add_option('--sent_simplify', dest='sent_simplify', default=sent_simplify)
        self.add_option('--repl_spec_char', dest='repl_spec_char', default=0)
        self.add_option('--ent_coref', dest='ent_coref', default=ent_coref)
        self.add_option('--ent_nest', dest='ent_nest', default=ent_nest)
        self.add_option('--dbl_ent_type', dest='dbl_ent_type', default=dbl_ent_type)
        self.add_option('--diff_ent_type', dest='diff_ent_type', default=diff_ent_type)  # 1-relation instances with different types of entities
        self.add_option('--num_classes', dest='num_classes', default=num_classes)
        self.add_option('--test_file_id', dest='test_file_id', default=False)  # the test file in the corpus

        # non-bert network parameters
        self.add_option('--word_vector_path', dest='word_vector_path', default=word_vector_path)
        self.add_option('--vocab_size', dest='vocab_size', default=vocab_size)
        self.add_option('--embedding_dim', dest='embedding_dim', default=embedding_dim)
        self.add_option('--lstm_hidden_dim', dest='lstm_hidden_dim', default=lstm_hidden_dim)
        self.add_option('--lstm_output_dim', dest='lstm_output_dim', default=lstm_output_dim)
        self.add_option('--embedding_dropout', dest='embedding_dropout', default=embedding_dropout)
        self.add_option('--lstm_recur_dropout', dest='lstm_recur_dropout', default=lstm_recur_dropout)
        self.add_option('--lstm_output_dropout', dest='lstm_output_dropout', default=lstm_output_dropout)
        self.add_option('--optimizer', dest='optimizer', default=optimizer)

    def __str__(self):
        cfgs = ['{} = {}'.format(k, v) for k, v in sorted(self.__dict__.items(), key=lambda x:x[0])]
        return '\n'.join(cfgs)

from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


#from sklearn_crfsuite.metrics import flat_classification_report
# https://github.com/apogiatzis/NER-BERT-conll2003/blob/master/BERT_as_Keras_Layer_Example.ipynb
def build_lstm_model(level, cfg, EMBED_MATRIX=None):
    # only x1_in is used
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    model = Embedding(input_dim=cfg.vocab_size, output_dim=cfg.embedding_dim, weights=[EMBED_MATRIX],
                      input_length=None, mask_zero=True)(x1_in)
    model = Dropout(cfg.embedding_dropout)(model)
    if level == 'token' or 'Att' in cfg.model_name:
        return_sequences = True
    else:
        return_sequences = False
    model = Bidirectional(LSTM(units=cfg.lstm_hidden_dim, return_sequences=return_sequences,
                               recurrent_dropout=cfg.lstm_recur_dropout))(model)  # variational biLSTM

    if 'Crf' in cfg.model_name:
        model = TimeDistributed(Dense(cfg.lstm_output_dim, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        output = CRF(cfg.num_classes[0])(model)  # CRF layer
    elif level == 'token':
        output = TimeDistributed(Dense(cfg.num_classes[0], activation="softmax"))(model)  # a dense layer as suggested by neuralNer
    else:
        if 'Att' in cfg.model_name:
            model = SelfAttention()(model)
        output = Dense(cfg.num_classes[0], activation="softmax")(model)

    # model = Dropout(cfg.lstm_output_dropout)(model)
    model = Model([x1_in, x2_in, x3_in], [output])
    if 'Crf' in cfg.model_name:
        model.compile(optimizer=cfg.optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])
    else:
        model.compile(optimizer=cfg.optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def build_cnn_model(level, cfg, EMBED_MATRIX=None):
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    word_embed = Embedding(input_dim=cfg.vocab_size, output_dim=cfg.embedding_dim, weights=[EMBED_MATRIX],
                           input_length=None, mask_zero=False)(x1_in)

    model = Conv1D(filters=cfg.embedding_dim, kernel_size=3)(word_embed)

    if 'Piece' in cfg.model_name:
        model = piecewise_maxpool_layer(cfg.embedding_dim, cfg.max_seq_len-2)([model, x3_in])
    else:
        #model = MaxPooling1D(pool_size=cfg.max_seq_len-2)(model)
        #model = Lambda(Squeeze)(model)
        model = Lambda(Max)(model)

    output = Dense(cfg.num_classes[0], activation='softmax')(model)

    model = Model([x1_in, x2_in, x3_in], [output])
    model.compile(optimizer=cfg.optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def build_simple_lstm_crf_model(level, model_name, num_classes, vocab_size, EMBED_MATRIX=None, optimizer='rmsprop'):
    """
    used out of the project
    :param level: 'sent'-sentence, 'token'-token
    :param model_name:
    :param num_classes:
    :param vocab_size:
    :param EMBED_MATRIX:
    :param optimizer:
    :return:
    """
    # only x1_in is used
    input = Input(shape=(None,))

    model = Embedding(input_dim=vocab_size, output_dim=100, weights=[EMBED_MATRIX],
                      input_length=None, mask_zero=False)(input)
    model = Dropout(0.2)(model)
    if level == 'sent' and 'Att' not in model_name:
        return_sequences = False
    else:
        return_sequences = False
    model = Bidirectional(LSTM(units=100, return_sequences=return_sequences, recurrent_dropout=0.1))(model)  # variational biLSTM

    if 'Crf' in model_name:
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        output = CRF(num_classes)(model)  # CRF layer
    else:
        output = TimeDistributed(Dense(num_classes, activation="softmax"))(model)  # a dense layer as suggested by neuralNer

    model = Model(input, output)
    if 'Crf' in model_name:
        model.compile(optimizer=optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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
def create_bert_classification_model(level='sent', tcfg=None):

    for l in tcfg.bert_model.layers: l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    x = tcfg.bert_model([x1_in, x2_in])
    if level == 'token':
        if 'Crf' in tcfg.model_name:
            x = TimeDistributed(Dense(tcfg.lstm_output_dim, activation="relu"))(x)  # a dense layer as suggested by neuralNer
            output = CRF(tcfg.num_classes[0])(x)  # CRF layer
        else:
            output = TimeDistributed(Dense(tcfg.num_classes[0], activation="softmax"))(x)
    else:
        x = Lambda(lambda x: x[:, 0])(x)
        output = Dense(tcfg.num_classes[0], activation='softmax')(x)

    model = Model([x1_in, x2_in, x3_in], [output])
    if 'Crf' in tcfg.model_name:
        model.compile(loss=crf_loss, optimizer=Adam(1e-5), metrics=[crf_viterbi_accuracy])
    else:
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


