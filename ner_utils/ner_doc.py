"""
processing documents and sentences
"""
from collections import defaultdict

from ner_utils.ner_instance import *

#BIO_SPECIAL_CHARS = ((0xa0, 0x20), (0xa0, 0x20))
# the first is for CHEMD/train.txt, the others are for CEMP/train/dev
BIO_SPECIAL_CHARS = ((chr(771), '~'),(chr(191),'-'), (chr(0xad),'-'), ('\ufeff', ''),
                     ('?', '-'), ('⁢','-'), ('`', "'"))
# "H-CO(X̃(2)A')" in CHEMD/train.txt

BIO_SPECIAL_TOKENS = (('&#8212;', '--'),
                      ('( +/+ )', '-wildtype '), ('(+/+)', '-wildtype '), ('+/+', '-wildtype '),
                      ('( +/- )', '-knockdown '), ('(+/-)', '-knockdown '),
                      ('( -/- )', '-knockout '), ('(-/-)', '-knockout '), ('-/-', '-knockout '),
                      ('--', ' -- '), ('+/-', ' +/- '))

# if a position is in a range
def bio_in_range(pos, ranges):
    for range in ranges:
        if pos > range[0] and pos < range[1]:   return True
        if pos > range[1]:  return False
    return False

# get the left word
def get_bio_leftword(line, pos):
    #while line[pos] == ' ':  pos -= 1
    while line[pos].isspace():  pos -= 1
    for i in range(pos, -1, -1):
        if not line[i].isalnum():   # separators, not alphabeta or digit
            return line[(i+1):(pos+1)]
    return line[:(pos+1)]

# get the right token
def get_bio_rightword(line, pos):
    #while line[pos] == ' ':  pos += 1
    if pos > len(line)-1: return ''
    while pos < len(line)-1 and line[pos].isspace():  pos += 1
    for i in range(pos, len(line)):
        #if line[i] in ' \t':   # separators
        if line[i].isspace():   # separators
            return line[pos:i]
    return line[pos:]

# return: the pair of L/R parens' locations in sentence
# lparen, rparen: left and right paren
def bio_sentence_paren(sline, lparen, rparen):
    stack = []  # stack for ()
    parens = []  # matched ()
    for i in range(len(sline)):
        if sline[i] == lparen:  # '('
            stack.append([sline[i], i])  # L/R parens and its location
        elif sline[i] == rparen:  # ')'
            if len(stack) > 0 and stack[-1][0] == lparen:  # match ()
                top = stack.pop()
                parens.append([top[1], i])  # locations of Left and Right parens
    if 0 == 1 and len(stack) > 0:  # unmatched (
        print(sline)
        for ch, i in stack:   print(ch, i)
    # unnested matched ()
    mparens = []
    if len(parens) > 0:
        while len(parens) > 0:
            top = parens.pop()
            if len(mparens) == 0 or not (top[0] > mparens[-1][0] and top[1] < mparens[-1][1]):  # nested
                mparens.append(top)
    if 0 == 1 and len(mparens) > 0:
        print(parens)
        print(mparens)
    return mparens

# split bio-text into sentences
def split_bio_sentence_old(sline):
    # do not split the string within parens and brackets
    mparens = bio_sentence_paren(sline, '(', ')')  # parens
    nparens = bio_sentence_paren(sline, '[', ']')  # brackets
    dlines = []
    line_no, line_pos = 0, 0
    for i in range(len(sline) - 1):
        # .|? and not in parenthesis
        # sline[i + 1] == ' ' --> sline[i+1].isspace()
        if sline[i] in '.?!' and sline[i+1].isspace() and not bio_in_range(i, mparens) and not bio_in_range(i, nparens):
            lword = get_bio_leftword(sline, i - 1)  # left word
            # the right word begins with a lowercase char.
            rword = get_bio_rightword(sline, i + 1)
            rInitialLowerID = (rword == '' or rword[0].islower())
            # the right word does not start with a lowercase
            # not single alphabeta , not in titles, or 'degrees C.'
            # if not (re.match('^[a-zA-Z]$', lword) or lword in titles) or \
            #         (sline[i - 9:i - 2] == 'degrees' and lword == 'C'):
            if not (re.match('^[a-zA-Z]$', lword) or lword in titles or rInitialLowerID) or \
                    (sline[i - 9:i - 2] == 'degrees' and lword == 'C'):
                dlines.append(sline[line_pos:(i + 1)].strip())
                line_no += 1
                line_pos = i + 2
    if line_pos < len(sline):  # last line
        dlines.append(sline[line_pos:].strip())
    return dlines

def tokenize_sentence_space(sline):
    tokens, locdict, j = [], {}, 0
    for i, ch in enumerate(sline):
        if ch.isspace():
            if i > j:  tokens.append(sline[j:i])
            j = i+1
        else:
            locdict[i] = len(tokens)
    # the last token
    if j < len(sline):  tokens.append(sline[j:])
    offsets = [[100000, -1] for _, _ in enumerate(tokens)]
    for i in locdict:
        tno = locdict[i]    # tno
        if i < offsets[tno][0]:  offsets[tno][0] = i
        if i > offsets[tno][1]:  offsets[tno][1] = i
    return tokens, offsets, locdict,

# split bio-text into sentences
def split_bio_sentence(sline):
    #tokens, offsets, locdict = tokenize_sentence_space(sline)
    # do not split the string within parens and brackets
    mparens = bio_sentence_paren(sline, '(', ')')  # parens
    nparens = bio_sentence_paren(sline, '[', ']')  # brackets
    dlines = []
    line_no, line_pos = 0, 0
    for i in range(len(sline) - 1):
        # .!? and not in parenthesis
        # sline[i + 1] == ' ' --> sline[i+1].isspace()
        # if sline[i] in '.?!' and i < len(sline)-1 and i == offsets[locdict[i]][-1] and \
        #         not bio_in_range(i, mparens) and not bio_in_range(i, nparens):
        #     tidx = locdict[i]
        #     token = tokens[tidx]
        #     #if len(token) == 2 and token[0].isupper() and tokens[tidx+1][0].isupper():
        #     # isolated . preceded by GET21, or the next word is initially capitalized, but not ending with .
        #     if len(token) == 1 or tokens[tidx+1][0].isupper() and not tokens[tidx+1].endswith('.'):
        #         if token[:-1] in titles:
        #             print()
        #             print(sline[i-20:i+50])
        #             print(token, tokens[tidx-1:tidx+1], tokens[tidx-2:tidx+3])
        if sline[i] in '.?!' and i < len(sline)-1 and sline[i+1].isspace() and \
                not bio_in_range(i, mparens) and not bio_in_range(i, nparens):
            lword = get_bio_leftword(sline, i - 1)  # left word
            # the right word begins with a lowercase char.
            rword = get_bio_rightword(sline, i + 1)
            rInitialLowerID = (rword == '' or rword[0].islower())
            # the right word does not start with a lowercase
            # not single alphabeta , not in titles, or 'degrees C.'
            # if not (re.match('^[a-zA-Z]$', lword) or lword in titles) or \
            #         (sline[i - 9:i - 2] == 'degrees' and lword == 'C'):
            # if not (re.match('^[a-zA-Z]$', lword) or lword in titles or rInitialLowerID) or \
            #         (sline[i - 9:i - 2] == 'degrees' and lword == 'C'):
            if not (lword in titles or rInitialLowerID or rword.endswith('.')):
                dlines.append(sline[line_pos:(i+1)].strip())
                line_no += 1
                line_pos = i + 2
    if line_pos < len(sline):  # last line
        dlines.append(sline[line_pos:].strip())
    return dlines

# titles
titles=('pp', 'vs','al', 'Mr', 'Ms', 'Mrs', 'Dr', 'St', 'No')
greekCH = ('alpha', 'beta', 'gamma')

wordSepCHs = '()[]{}:;",?*%\''

# sentence tokenization and normalization
def tokenize_bio_sentence(sline):
    pos, dline = 0, ''  # last position and new line
    for i in range(len(sline)):
        #
        if sline[i] == '-': # hyphen
            # search for the left entity from i-1, then separate -
            lword = get_bio_leftword(sline, i-1)
            rtoken = get_bio_rightword(sline, i)
            if re.match('-(.*)-$', rtoken): continue    # skip -to-, --
            #
            if is_bio_entity(lword) or sline[i-1] == ')':   # an entity mention or )
                dline += sline[pos:i] + sline[i] + ' '
                pos = i + 1
        elif sline[i:i+2] in ("'s", "'S") and sline[i+2] == ' ':    # 's 'S followed by a space
            dline += sline[pos:i] + ' ' + sline[i]
            pos = i + 1
        # elif sline[i] == "'" and sline[i-1].isdigit(): # omit ' following digit, e.g., 5'-flanking
        #     dline += sline[pos:i]
        #     pos = i + 1
        # word separators
        elif sline[i] in wordSepCHs:
            dline += sline[pos:i] + ' ' + sline[i] + ' '
            pos = i + 1
    dline += sline[pos:]
    dline = ' '.join(re.split('[ ]+', dline))
    return dline

# words are continuous strings of uppercase or lowercase alphabets or digits
def tokenize_ner_bio_sentence(sline):
    dline, lch = '', ''
    for i, ch in enumerate(sline):
        if ch.isdigit() and lch.isdigit() or \
                is_upper(ch) and is_upper(lch) or \
                is_lower(ch) and is_lower(lch) or \
                (is_upper(lch) and is_lower(ch) and (i==1 or not is_upper(sline[i-2]))):
            dline += ch
            lch = ch
        else:
            if ch.isspace():  ch=' '  # replace all kinds of spaces with ' '
            dline += ' ' + ch
            lch = ch
    dline = ' '.join(re.split('[ ]+', dline))
    return dline

hyphenRE = re.compile('[\(|\)|,|\.]')
# recombine hyphen
def bio_hyphen_recomb(words):
    tokens = []
    i = 0
    while (i < len(words)):
        if i < len(words)-2 and words[i+1] == '-':
            if not (is_bio_entity(words[i]) or hyphenRE.search(words[i]) or \
                    is_bio_entity(words[i+2]) or hyphenRE.search(words[i+2])):
                #print(words[i:i+3])
                tokens.append(''.join(words[i:i+3]))
                i += 3
                continue
        #if i < len(words)-1 and words[i] == '-' and not words[i-1].isdigit() and words[i+1].isdigit():
        #    print(words[i-1:i+2])
        tokens.append(words[i])
        i += 1
    return tokens

# combine parenthesis
# remove word sequence in parentheses not containing entities
# return False if no combination
def bio_paren_recomb(words):
    fpos, lpos = -1, -1    # the 1st and last left parentheses' position
    tokens, mentions, combID = [], [], False
    for i in range(len(words)):
        if words[i] == '(':
            if fpos < 0:   fpos = i    # the 1st (
            lpos = i    # the last (
            mentions = []
        elif words[i] == ')' and lpos >= 0:
            # entities between parentheses
            mwords = []
            if len(mentions) > 0: mwords = ['[[['] + words[lpos+1:i] + [']]]']    # entity in parentheses
            # nested parentheses
            combID = True
            if lpos == fpos:   # no nested parenthesis
                tokens.extend(mwords)
                fpos, lpos = -1, -1
            else:               # previous left parentheses
                tokens.extend(words[fpos:lpos] + mwords + words[i+1:])
                fpos, lpos = -1, -1
                break
        else:   # not ( )
            if lpos < 0:    # no left parenthesis
                if words[i] == ')' :    tokens.append(']]]')    # unmatched )
                else:   tokens.append(words[i])
            elif is_bio_entity(words[i]):  # is a bio entity mention
                mentions.append(words[i])  # save the innermost mentions
    if fpos >= 0:   # unmatched (
        if 0 == 1:
            print('\nUnmatched parenthesis:', fpos, words[fpos])
            print(words)
        tokens.extend(words[fpos:])
        #print(tokens)
    if 0 == 1 and combID:
        print(' '.join(words))
        print(' '.join(tokens))
    return combID, tokens


# process words sequence, such as hyphen recombination
# op: 'HYPHEN'-hyphen, 'PAREN'-parenthsis
def simplify_bio_sentence(sline, op):
    words = re.split('[\s]+', sline)
    if op == 'HYPHEN':
        words = bio_hyphen_recomb(words)
    elif op == 'PAREN':
        combID = True
        while combID:
            combID, words = bio_paren_recomb(words)
        for j in range(len(words)):
            if words[j] == '[[[':   words[j] = '('
            elif words[j] == ']]]': words[j] = ')'
    #
    dline = ' '.join(words)
    if dline.find('- -') >= 0:
        dline = dline.replace('- -', '--')  #
    return dline

# period tokenization at EOS
def tokenize_bio_eos(sline):
    sline = sline.strip()
    if len(sline) > 1 and sline[-1] in '.!?' and sline[-2] != ' ':
        sline = sline[:-1] + ' ' + sline[-1]
    return sline


# build coreference dict like ENT1(ENT2)
def bio_coref_build(ts):
    j, tokens, coref_pair = 0, [], []       # coreference pairs
    while j < len(ts):
        # coreference like ENT1( ENT2 ) or ENT1 [ ENT2 ]
        # omit the second in the sentence
        emid1 = is_bio_entity(ts[j])
        emid2 = is_bio_entity(ts[j + 2]) if j < len(ts)-3 else False
        # e1 and e2 must have the same type
        if j < len(ts)-3 and emid1 and ts[j+1] in '([' and emid2 and ts[j+3] in ')]' and emid1[:2] == emid2[:2]:
            tokens.append(ts[j])
            coref_pair.append([emid1[2:], emid2[2:]])
            j += 4
        tokens.append(ts[j])
        j += 1
    #if len(coref_pair) > 0: print(coref_pair)
    return tokens, coref_pair

ENT_BRACKET_ID = 0  # surrounded by []

# return a sentence with all bio entity mention placeholders replaced by CHEM_1, GENE_1,...
# bld_ent_seq: 0-type only, 1-type+seq_no, 2- entity id
def blind_entity_mention_placeholders(text, bld_ent_mode=0):
    words = text.split(' ')
    edict = {}
    for i, word in enumerate(words):
        etype, emid = is_bio_entity(word)
        if not emid: continue    # not an entity mention
        # increase entity sequence no like GENE_1, CHEM_1, DISE_1
        if bld_ent_mode == 0:   # 0-entity type, 1-type_seq, 2-entity id
            words[i] = etype
        else:
            if etype not in edict:  edict[etype] = 1
            eno = edict[etype]
            edict[etype] += 1
            words[i] = '{}_{}'.format(etype, eno)
        if ENT_BRACKET_ID: words[i] = '[' +words[i] + ']'
    return ' '.join(words)

# split biomedical text into sentences in terms of corpus format
def split_bio_sentences(fmt, text):
    lines = [text]  # fmt == 's' for one sentence
    if fmt == 'a':  # abstract, one paragraph
        lines = split_bio_sentence(text)
    elif fmt == 'f':  # full-text, multi-paragraph
        lines = []
        olines = text.split('\n')
        for i, oline in enumerate(olines):
            dlines = split_bio_sentence(oline)
            lines.extend(dlines)
    return lines


class nerDoc(object):
    # one document for information extraction
    def __init__(self,
                 id = None,
                 no = None,
                 title = '',
                 text = '',
                 tokensID = False     # whether generate tokens from text
                 ):
        self.id = id
        self.no = no
        self.title = title
        self.otext = text   # original text
        self.text = text    # preprocessed text
        #
        self.tokens = Tokens(text.split(' ') if tokensID else [])    # Tokens

        self.emdict = {}    # entity mention dict from id --> EntityMention
        self.emlist = []    # entity mention list
        self.remlist = []   # recognized entity mentions

        self.sntlist = []   # sentence list for a doc
        self.crfdict = defaultdict(list)    # coreference dictionary from an entity to its referents

    @ property
    def words(self):  return self.tokens.words

    @ property
    def offsets(self):  return self.tokens.offsets

    def __copy__(self):
        doc = nerDoc()
        for att in self.__dict__.keys():
            if type(self.__dict__[att]) not in (list, dict):  doc.__dict__[att] = self.__dict__[att]
        return doc

    def __str__(self):
        stext = '\nOTEXT:\n{}\nTEXT:\n{}'.format(self.otext, self.text)
        swords = '\nWORDS:\n{}'.format(self.words) if len(self.words) > 0 else ''
        elist = '\n'.join(['{}'.format(em) for em in self.emlist])
        selist = '\nGOLD ENTITIES:\n{}'.format(elist) if len(elist) > 0 else ''
        relist = '\n'.join(['{}'.format(em) for em in self.remlist])
        srelist = '\nRECOGNIZED ENTITIES:\n{}'.format(relist) if len(relist) > 0 else ''
        rstr = '\nID: {}\tNO: {} TITLE: {}{}{}{}{}'
        return rstr.format(self.id, self.no, self.title, stext, swords, selist, srelist)

    def build_tokens(self):   # for sentence
        self.tokens = Tokens(words=self.text.split(' '))
        return

    def build_sntlist_tokens(self):   # for sentence
        for snt in self.sntlist:  snt.build_tokens()

    def get_entity_mention(self, emid):
        if emid not in self.emdict:  return None
        return self.emdict[emid]

    def _index_entity_mentions(self):
        self.emdict = {em.id:i for i, em in enumerate(self.emlist)}

    def append_entity_mention(self, em, gold=True):
        if gold:
            self.emlist.append(em)
            self.emdict[em.id] = em
        else:
            self.remlist.append(em)

    def sort_entity_mentions(self):
        self.emlist.sort()
        #self._index_entity_mentions()  # reindex entity dict from id->index_no

    # add an entity to a sentence, move to ie_conversion.py
    def add_entity_mention_to_sentence(self, did, sent_no, feats, eno, spos, epos, type):
        # eid = '{}-{}-{}'.format(cps_file, sent_no, eno)
        eid = 'T{}'.format(eno)
        ename = '_'.join([feat[0] for feat in feats[spos:epos]])
        em = EntityMention(did=did, id=eid, type=type, name=ename, lineno=sent_no, hsno=spos, heno=epos)
        self.append_entity_mention(em)
        return

    def replace_bio_special_tokens(self, tokens):
        # &#8212;--> -- EM DASH
        for stoken, dtoken in tokens:
            self.text = self.text.replace(stoken, dtoken)
        return

    def replace_bio_special_chars(self, chars):
        for schar, dchar in chars:
            self.text = self.text.replace(schar, dchar)
        return

    def replace_entity_mention_with_placeholder(self):
        """
        start numbering from 0, transfer from otext to text
        :return: set self.text
        """
        tline = self.otext  #
        sline = ''
        lspos, lepos = -1, 0    # # last entity position
        for i, em in enumerate(self.emlist):
            # skip when visible is False
            if not em.visible:   continue
            sline += '{} {}{} '.format(tline[lepos:em.hsno], em.type[:4], em.id)  # type length is at most 4
            lspos, lepos = em.hsno, em.heno
        sline += tline[lepos:]
        self.text = ' '.join(re.split('[ ]+', sline))
        return

    def mask_nested_entity_mentions(self, nestedID=False, doubleID=False):
        """
        set visible to False for nested or duplicated entities
        :param nestedID: add nested entities to coreference list if set to True
        :param doubleID: add duplicated entities to coreference list if set to True
        :return:
        """
        ents = self.emlist
        eidx, spos, epos = -1, -1, -1  # the last entity index and start, end position
        #nextID, prevID, doubleID = True, True, True # nested in next or preceeding entities, or double typed
        nextdID, prevdID, doubledID = False, False, False
        #
        for i in range(len(ents)):
            # nested in the next entitiy
            if i < len(ents) - 1 and ents[i].hsno == ents[i+1].hsno and ents[i].heno < ents[i+1].heno:
                ents[i].visible = False
                if nestedID: self.crfdict[ents[i+1].id].append(ents[i].id) # link i to i+1
                if nextdID: print('Nested in the next entitiy:\n{}\n{}'.format(ents[i], ents[i+1]))
            # nested in the previous entity or duplicated entities
            elif ents[i].heno <= epos:
                ents[i].visible = False
                # double-typed entity
                if ents[i].hsno == spos and ents[i].heno == epos:
                    if doubleID: self.crfdict[ents[eidx].id].append(ents[i].id)  # link i to eidx
                    ents[i].linkid = ents[eidx].id    # refer to the previous entity
                    if doubledID: print('\nDuplicated entities: {}\n{}\n{}'.format(self.id, ents[eidx], ents[i]))
                    eidx = i
                # nested in the previous entity
                else:
                    if nestedID: self.crfdict[ents[eidx].id].append(ents[i].id)  # link i to eidx
                    if prevdID: print('Nested in the previous entity:\n{}\n{}'.format(ents[eidx], ents[i]))
            else:
                eidx, spos, epos = i, ents[i].hsno, ents[i].heno
        return

    def mask_outer_entity_mentions(self):
        """
        mask out-layer entities or duplicated entities
        :return: set them to invisible
        """
        ents = self.emlist
        for i, ent in enumerate(ents):
            # include or equal the preceding one
            if i > 0 and ents[i-1].hsno == ents[i].hsno and ents[i-1].heno <= ents[i].heno:
                ent.visible = False
            # include the following entity
            elif i < len(ents)-1 and ents[i].hsno < ents[i+1].hsno and ents[i].heno >= ents[i+1].heno:
                ent.visible = False
        return

    def preprocess_document_entity_mentions(self, tcfg):
        """
        mask nested entities, replace with placeholders and special tokens
        :param tcfg:
        :return:
        """
        self.sort_entity_mentions()
        self.mask_nested_entity_mentions(nestedID=tcfg.ent_nest, doubleID=tcfg.dbl_ent_type)
        self.replace_entity_mention_with_placeholder()
        if tcfg.sent_simplify:  # change the text length
            self.replace_bio_special_tokens(BIO_SPECIAL_TOKENS)
        return

    def align_document_with_sentences(self, verbose=0, original=False):
        """
        align the original/text document with sentences, generate Tokens with offsets for sentences
        :param verbose:
        :return: set words and their corresponding offsets in the tokens
        """
        i = 0  # start from 0 in the document
        text = self.otext if original else self.text
        for snt in self.sntlist:
            # prepare Tokens
            #snt.tokens = Tokens(snt.text.split())
            for k, word in enumerate(snt.words):
                etype, emid = is_bio_entity(word)
                if etype:
                    em = self.get_entity_mention(emid)
                    if em:
                        snt.tokens[k].offsets = [em.hsno, em.heno]
                        i = em.heno
                        continue
                # skip spaces
                while i < len(text) and text[i].isspace():  i += 1
                j = i
                # match the next word
                token = ''
                while i < len(text):
                    if not text[i].isspace():     # token --> text[i]
                        token += text[i]
                        if token == word:
                            i += 1
                            snt.tokens[k].offsets = [j, i]
                            break
                    i += 1
            # check whether all words are aligned
            if verbose > 0 and len(snt.words) != len(snt.offsets):
                print('\nWords and offsets do not match!')
                print(text)
                print(self.text)
                print()
                print(snt)
                print('words:', snt.words, len(snt.words))
                print('offsets:', snt.offsets, len(snt.offsets))
                for i in range(min(len(snt.words), len(snt.offsets))):
                    pos = snt.offsets[i]
                    print(i, snt.words[i], snt.offsets[i], text[pos[0]:pos[1]])
        return

    def transfer_document_entity_mentions(self, verbose=0):
        """
        transfer entity mention from a document to sentences
        :param verbose:
        :return: set snt.emdict
        """
        #
        self.mask_outer_entity_mentions()
        lineno  = 0  # start from line 0
        for em in self.emlist:
            if not em.visible:  continue  # nested or duplicated
            spos, epos = -1, -1
            while lineno < len(self.sntlist):
                offsets = self.sntlist[lineno].offsets
                # entity mention is in the sentence
                if len(offsets) == 0:   # empty sentence
                    i = 0   # null statement
                elif em.hsno >= offsets[0][0] and em.heno <= offsets[-1][1]:
                    for i, offset in enumerate(offsets):
                        # some erroneous annotation which omits one preceeding char
                        if spos < 0 and (offset[0] == em.hsno or (offset[0] == em.hsno-1 and offset[1]-offset[0] > 1)):
                            spos = i
                        if spos >=0 and epos < 0 and offset[1] >= em.heno:  epos = i
                        if spos >= 0 and epos >= 0:  break
                    break
                # entity mention across multiple sentences
                elif em.hsno < offsets[-1][1] and em.heno > offsets[-1][1]:
                    if verbose:
                        print('\nEntAcrossMultiSents: {} {}'.format(self.id, em))
                    break
                lineno += 1
            # valid entity mention
            if spos >= 0 and epos >= 0:
                nem = em.__copy__(lineno=lineno, hsno=spos, heno=epos+1)
                self.sntlist[lineno].append_entity_mention(nem)
        # debug purpose
        if verbose:  # the numbers of entity mentions in doc and snt are different
            dset = set(em.id for em in self.emlist if em.visible)
            sset = set(em.id for snt in self.sntlist for em in snt.emlist)
            dno, sno = len(dset), len(sset)
            if verbose >= 1 and dno > sno:
                print('\nEntNumDocSnts: {} {} {}'.format(self.id, dno, sno))
                if verbose >= 2:
                    #for em in self.emlist:  print(em)
                    eids = dset.difference(sset)
                    print('Diff:')
                    for eid in eids:  print(self.get_entity_mention(eid))
        return

    def generate_document_sentences(self, tcfg, Doc, task='ner', fmt='a'):
        """
        generate sentences for a document
        :param tcfg:
        :param Doc: the target sentence class
        :param fmt: 's'-sentence, 'a'-abstract, 'f'-full-text
        :return:
        """
        if tcfg.repl_spec_char:
            self.replace_bio_special_tokens(BIO_SPECIAL_CHARS)
        # split into sentences
        lines = split_bio_sentences(fmt, self.text)
        # make sentences
        for i, line in enumerate(lines):
            # tokenize the sentence
            if task == 'ner':  dline = tokenize_ner_bio_sentence(line)
            else:  dline = tokenize_bio_sentence(line)
            #
            if tcfg.sent_simplify:
                dline = simplify_bio_sentence(dline, 'PAREN')
            dline = tokenize_bio_eos(dline)
            # make the sentence, no tokens
            snt = Doc(id=self.id, no=i, title=self.title, text=dline)
            self.sntlist.append(snt)
        return

    def transfer_document_annotations(self, tcfg):
        """
        postprocess document sentences, usu. alignment between doc and sentences, transfer annotations
        :param tcfg:
        :return:
        """
        # build tokens for all sentences
        self.build_sntlist_tokens()
        # todo, add linguistic features here, ...
        # align the document with its sentences for NER
        self.align_document_with_sentences(verbose=tcfg.verbose)
        self.transfer_document_entity_mentions(verbose=tcfg.verbose)
        return

    def recover_entity_mentions(self, tcfg, snt):
        """
        recover entity mentions from a sentence in a document, refill entity mentions if necessary
        build coreference pair if corefID is set to True
        :param tcfg:
        :param snt:  the target sentence
        :return: set snt.emdict and snt.text
        """
        tokens = re.split('[ ]+', snt.text)
        # build entity coreference like ENT1(ENT2)
        if tcfg.ent_coref:
            tokens, coref_pair = bio_coref_build(tokens)
            for emid1, emid2 in coref_pair:
                self.crfdict[emid1].append(emid2)
                self.get_entity_mention(emid2).visible = False # disable visible if it is a coreferent
        # build entity mention list for the line
        # entity mention refilling for trigger word and entity in GE09 task
        ntokens = []    # new tokens refilled
        for j, token in enumerate(tokens):
            etype, emid = is_bio_entity(token)
            if emid:
                oem = self.get_entity_mention(emid) # original entity mention
                if oem and etype == oem.type:       # emid is found and the type matches
                    tlen = len(ntokens)
                    oem = self.get_entity_mention(emid)
                    #if task == 've' and oem.type in ('TRIG', 'ENTI'):     # trigger and entities replacement
                    # not in the blinded entity types, refill the placeholder with the original name
                    if oem.type not in tcfg.bld_ent_types:
                        names = oem.name.split('_')      # originally separated by space
                        ntokens.extend(names)
                        em = oem.__copy__(lineno=snt.no, hsno=tlen, heno=tlen+len(names))
                    else:
                        em = oem.__copy__(lineno=snt.no, hsno=tlen, heno=tlen+1)
                        ntokens.append(token)
                    snt.append_entity_mention(em)
                    for cid in self.crfdict[emid]:
                        em = self.get_entity_mention(cid).__copy__(lineno=snt.no, hsno=tlen, heno=tlen+1)
                        snt.append_entity_mention(em)
                    continue
                # elif verbose:   # a mismatched entity placeholder
                #     print('EntNotFound: {} in {}'.format(token, self.id))
            ntokens.append(token)
        # update the sentence
        snt.text = ' '.join(ntokens)
        return

    def match_gold_pred_instances(self):
        match_gold_pred_entity_mentions(self)

    def collect_ner_confusion_matrix(self, confusions, etypedict):
        collect_ner_confusion_matrix(self, confusions, etypedict)

    def generate_sentence_instances(self, tcfg):
        """
        generate SequenceLabel for the sentence
        :param tcfg:
        :return: list of SequenceLabel
        """
        sl = SequenceLabel(self.id, self.no, self.emlist, self.tokens)
        return [sl]

# Doc class for segmented sequence labeling
class sslDoc(nerDoc):
    def __init__(self,
                 id = None,
                 no = None,
                 title = '',
                 text = ''
                 ):
        super(sslDoc, self).__init__(id=id, no=no, title=title, text=text)
        self.smlist = []  # segment entity mention list

    def __str__(self):
        tdoc = super(sslDoc, self).__str__()
        slist = '\n'.join(['{}'.format(sm) for sm in self.smlist])
        sslist = '\nSEGMENTS:\n{}'.format(slist) if len(slist) > 0 else ''
        return '{}{}'.format(tdoc, sslist)

    # for NER, label schema is unknown at this time
    def generate_sentence_instances(self, tcfg):
        sl = segSequenceLabel(self.id, self.no, self.text, self.emlist, self.offsets, self.smlist)
        return [sl]
