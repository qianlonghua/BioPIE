# modules for BC utilities

# print a list on the screen
# def print_list(mlist):

# print a dict on the screen
# def print_dict(mdict):

# convert a number to a string with thousand separators
# def int2kstr(value): 

# remove a file safely, return True if successfully, False otherwise
# def safe_remove_file(sfilename):

# read a file into a string, fcoding: file encoding, lowerID: whether to convert to lower case
# def file_line2string(sfilename,fcoding,lowerID):  

# write a string into a file
# def file_string2line(line,dfilename,fcoding):

# read a file into a list of lines, stripID: whether to strip the surrounding white spaces
# def file_line2list(sfilename,fcoding,lowerID,stripID):    
    
# read a file into an array separated by sepch
# def file_line2array(sfilename,fcoding,lowerID,sepch):

# write a list of lines into a file
# def file_list2line(lines,dfilename,fcoding):

# load a mapping file with two columns to a dictionary without duplicate keys 
# msep: the separator for columns
# rvsID: whether to exchange two columns 
# initids: whether the conversion to integer is performed for column 0 and 1 respectively like [True,False]
# lowerids: whether the conversion to lower case is performed for colmun 0 and 1 respectively like [False,False]
# def mapping_dict_load(mfilename,fcoding,msep,rvsID,intids,lowids):

# save a dictionary to the mapping file, the key and value must be strings
# def mapping_dict_save(mdict,mfilename,fcoding):

# load a dictionary with duplicate keys, when duplicate keys happen the dict item becomes a list
# def multi_mapping_dict_load(mfilename,fcoding,msep,rvsID,intids,lowids):

# return a reverse dictionary for the input dictionary, i.e., exchange keys and values
# def reverse_dict_create(sdict):


import re
import os
import os.path
import sys
import shutil
import json
import numpy as np

# BE_DICT = {'GE':0, 'CH':1, 'DI':2, 'TR':3, 'EN':4, 'BP':5}
# BioENTs = ('GENE', 'CHEM', 'DISE', 'TRIG', 'ENTI', 'BPRO')
# phRE = re.compile('(CH|GE|DI|TR|EN|BP|ch|ge|di|tr|en|bp)([T|t]\d+)')

def clear_gpu_processes():
    os.system('/home/qlh/gpu.sh')

def npdiv0(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def percent(a, b):
    return a / b * 100 if b != 0 else 0

def div0(a, b):
    return a / b if b != 0 else 0

def calc_prf(tp, fp, fn):
    p = percent(tp, float(tp + fp))
    r = percent(tp, float(tp + fn))
    f1 = div0(2 * p * r, p + r)
    return p, r, f1

def is_upper(ch):
    return ch in ('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def is_lower(ch):
    return ch in ('abcdefghijklmnopqrstuvwxyz')

# pID: PRED_MODE
def mprint(pID, pmess):
    if not pID: print(pmess)
    return

# pID: PRED_MODE
def mwrite(pID, pmess):
    if not pID: write(pmess)
    return

biophRE = re.compile('(^[A-Z]{4})([T|t]\d+$)')
# determine if a token is a bio entity mention
# return:  token, etype, or False
def is_bio_entity(token):
    m = biophRE.match(token)
    if not m:  return None, None
    return m.group(1), m.group(2)   # etype, emid

# extend seq to wlen with pad
def seq_pad(slen, wlen, seq, pad):
    if wlen >= slen: return
    padding = [pad] * (slen - wlen)
    seq.extend(padding)
    return

# combination of m from n
def greedy_combination(n, m):
    if m == 0 or n < m:   return [[]]
    buf = [i for i in range(m)] # initial buffer
    cmb = [buf[:]]  # first element
    while True:
        idx = -1    # search the index of the next increase
        for i in range(-1, -1-m, -1):
            if buf[i] != n+i:
                idx = i+m
                break
        # reach the end
        if idx == -1:   break
        # move to the next combination
        ino = buf[idx]
        for i in range(idx, m):
            buf[i] = ino+i-idx+1
        #print(idx, ino, buf)
        cmb.append(buf[:])
    return cmb

# whether a string is a float
def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

def write(mess):
    sys.stdout.write(mess)
    return

def disp_total_files(file_no,total_no):
    print('Totally %s of %s files processed.' % (int2kstr(file_no),int2kstr(total_no)))
    return

# make gazeteer
def make_gazeteer_dict(filename, fcoding):
    temp_dict={}
    lines=file_line2list(filename, fcoding, True, True)
    for i in range(len(lines)):
        items=lines[i].split('\t')
        if items[0] not in temp_dict:
            temp_dict[items[0].lower()]=i
    return temp_dict

def print_list(mlist):
    for i in range(len(mlist)):  print(i, mlist[i])
    return

def print_dict(mdict):
    for key in sorted(mdict):  print(key, mdict[key])
    return

# convert number to string with thousand separators
def int2kstr(value):
    ret='{:,}'.format(value)
    return ret

# remove a file safely, True if successfully, False otherwise
def safe_remove_file(sfilename):
    if os.path.exists(sfilename):
        os.remove(sfilename)
        return True
    return False

# copy the sfilename if it exists, others remove tfilename if it exists
def safe_copy_file(sfilename, tfilename, verbose=0):
    if os.path.exists(sfilename):
        shutil.copyfile(sfilename, tfilename)
        if verbose:  print('Copy file {} to {}'.format(sfilename, tfilename))
        return True
    safe_remove_file(tfilename)
    return False

#read a file to a string
def file_line2string(sfilename, fcoding='utf8', lowerID=False, verbose=0):
    finput = open(sfilename, 'r', encoding=fcoding)
    text = finput.read()
    finput.close()
    if text.endswith('\n'): text = text[:-1]
    if lowerID: text = text.lower()
    return text

#write a string a file
def file_string2line(line, dfilename, fcoding='utf8'):
    foutput=open(dfilename, 'w', encoding=fcoding)
    print(line, file=foutput)
    foutput.close()
    return

# read a file into a list
# fcoding-encoding,lowerID: lowercase or not
def file_line2list(sfilename, fcoding='utf8', lowerID=False, stripID=True, verbose=0):
    finput = open(sfilename, 'r', encoding=fcoding)
    lines = finput.readlines()
    for i in range(len(lines)):
        if stripID: lines[i] = lines[i].strip()
        else: lines[i] = lines[i].strip('\r\n')
        if lowerID: lines[i] = lines[i].lower()
    finput.close()
    if verbose:  print('\nLoading {:,} lines from {} ...'.format(len(lines), sfilename))
    return lines

# read a file into an array separated by sepch
def file_line2array(sfilename, fcoding='utf8', lowerID=False, stripID=True, sepch='\t', verbose=0):
    lines = file_line2list(sfilename, fcoding, lowerID, stripID, verbose)
    arrays = []
    for line in lines:
        arrays.append(line.split(sepch))
    return arrays

# write a list of lines into a file
def file_list2line(lines, dfilename, fcoding='utf8', verbose=0, mode='w'):
    if verbose:  print('Saving {:,} lines to {} ...'.format(len(lines), dfilename))
    foutput = open(dfilename, mode, encoding=fcoding)
    print('\n'.join(lines), file=foutput)
    foutput.close()
    return

# convert list of list to list of string
def list_list2string(lines):
    dlines=[]
    for line in lines:
        dlines.append('\t'.join(line))
    return dlines

def fixedfile_list2line(lines, dfilename, fcoding, fsize):
    # get filename and extension
    m=re.match(r'(.*?)\.([^\.]*)$',dfilename)
    if m:   dfile, dext = m.group(1), m.group(2)
    else:   dfile, dext = dfilename, ''
    #
    file_index = 0
    file_size = 0
    foutput = open('%s_%02d.%s' % (dfile, file_index, dext), 'w', encoding=fcoding)
    for line in lines:
        file_size += len(line)
        if file_size > fsize:
            foutput.close()
            file_index += 1
            file_size = len(line)
            foutput = open('%s_%02d.%s' % (dfile, file_index, dext), 'w', encoding=fcoding)
        print(line, file=foutput)
    foutput.close()
    return

# load the dictionary file for translation pair and its probability
# return the dict data structure
def lexicon_load(dict_file):
    print('Loading the dictionary file %s (5000/.)' % dict_file)
    i=0
    ce_dict={}
    dfile=open(dict_file, 'r', encoding='utf8')
    for line in dfile.readlines():
        line = line.strip()
        i += 1
        if i % 5000 == 0:   write('.')
        mlist = line.split(' ')
        ce_dict[mlist[0]+mlist[1].lower()] = float(mlist[2])
    print('OK')
    return ce_dict

# Load a mapping dictionary like sstring-->dstring
# msep: separator
# rvsID: True-reverse,False-sequential,
# initid: whether conversion to integer is performed for item 0 and 1 respectively
# lowerID: whether the key is lowercased
def mapping_dict_load(pID, mfilename, fcoding, msep, rvsID, intids, lowids):
    mprint(pID, 'Loading the mapping file: %s (10,000/.)' % mfilename)
    map_dict={}
    lines=file_line2list(mfilename,fcoding,False,True)
    dupcnt=0
    for i in range(len(lines)):
        mlist=lines[i].split(msep)
        if lowids:     # convert into lowercase if necessary
            for j in range(2):
                if lowids[j]:  mlist[j] = mlist[j].lower()
        if intids:     # convert into integers if necessary
            for j in range(2):
                if intids[j]:  mlist[j] = int(mlist[j])
        if rvsID:   mlist = [mlist[1], mlist[0]]
        if mlist[0] not in map_dict:
            map_dict[mlist[0]] = mlist[1]
        else:
            print(mlist, map_dict[mlist[0]])
            dupcnt += 1
        if i % 10000 == 0:   mwrite(pID, '.')
    mprint(pID, 'OK')
    if dupcnt != 0:    print('%d duplicate items' % dupcnt)
    return map_dict

# save mapping dictionary file
def mapping_dict_save(mdict, mfilename, fcoding):
    mlines = []
    for key in sorted(mdict.keys()):
        mlines.append('%s\t%s' % (key, mdict[key]))
    file_list2line(mlines, mfilename, fcoding)
    return

# load a dictionary with multiple key values
def multi_mapping_dict_load(mfilename, fcoding, msep, rvsID, intids, lowids):
    print('Loading the multi-mapping file: %s (10,000/.)' % mfilename)
    map_dict = {}
    lines = file_line2list(mfilename, fcoding, False, True)
    for i in range(len(lines)):
        mlist = lines[i].split(msep)
        if lowids:     # convert into lowercase if necessary
            for j in range(2):
                if lowids[j]:  mlist[j] = mlist[j].lower()
        if intids:     # convert into integers if necessary
            for j in range(2):
                if intids[j]:  mlist[j] = int(mlist[j])
        if rvsID:  mlist = [mlist[1],mlist[0]]
        if mlist[0] not in map_dict:
            map_dict[mlist[0]] = [mlist[1]]
        else:
            map_dict[mlist[0]].append(mlist[1])
        if i % 10000 == 0:   write('.')
    print('OK')
    return map_dict

# return a reverse dictionary for the input dictionary
def reverse_dict_create(pID, sdict):
    mprint(pID, 'Creating the reverse dictionary (10,000/.)')
    ddict = {}
    i = 0
    for key in sdict.keys():
        value = sdict[key]
        if value not in ddict.keys():   # the value has existed
            ddict[value] = key
        i += 1
        if i % 10000 == 0:  mwrite(pID, '.')
    mprint(pID, 'OK')
    return ddict

# list1, element2, list2, element2, ...
def batch_list_append(*args):
    if len(args) % 2 != 0:
        print('Parameters {} should be even!'.format(len(args)))
        return
    for i in range(0, len(args), 2):
        args[i].append(args[i+1])
    return

# return a dictionary from a json file, None if the file doesn't exist.
def load_json_file(filename=None):
    jdict = None
    if os.path.exists(filename):
        with open(filename, 'r') as fjson:
            jdict = json.load(fjson)
    return jdict

def load_word_voc_file(filename=None, verbose=0):
    if not os.path.exists(filename): return None
    words = file_line2list(filename, verbose=verbose)
    word_dict = {word:i for i, word in enumerate(words)}
    return word_dict

def save_word_voc_file(word_dict, filename, verbose=0):
    words = [word for word, idx in sorted(word_dict.items(), key=lambda x:x[1])]
    file_list2line(words, filename, verbose=verbose)
    return word_dict

def combine_word_voc_files(wdir, sfiles, dfile, verbose=0):
    word_dict ={}
    for sfile in sfiles:
        dict = load_word_voc_file(os.path.join(wdir, sfile), verbose=verbose)
        if dict:  word_dict.update(dict)
    dfilename = os.path.join(wdir, dfile)
    save_word_voc_file(word_dict, dfilename, verbose=verbose)
    return

