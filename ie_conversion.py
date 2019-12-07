"""
Corpus conversion:
convert the original corpus files to the standard corpus files as follows:
i-instance, s-sentence, a-abstract, f-full-text
"""

from ie_docset import *

diseRE = re.compile(r'<category="([^"]*)">([^<]*)</category>')
#diseRE = re.compile(r'</category>')
def convert_ncbi_corpus(wdir, cps_file, verbose=0):
    ofilename ='{}/{}.org'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    #
    slines = file_line2list(ofilename, verbose=verbose)
    tlines, elines, ecnt = [], [], 0
    pmid, title, abs, eno, text = None, None, None, 1, ''
    for sline in slines:
        if sline == '':  pmid, title, abs, eno, text = None, None, None, 1, ''
        elif not pmid:  pmid, _, title = sline.split('|')
        elif not abs:
            _, _, abs = sline.split('|')
            if abs.endswith('..'):  abs = abs[:-1]
            elif not abs.endswith('.'):  abs = abs + '.'
            tlines.append('\t'.join([pmid, title, abs]))
            text = ' '.join([title, abs])
        else:
            _, spos, epos, name, stype, link = sline.split('\t')
            eid = 'T{}'.format(eno)
            eno += 1
            type = 'DISE|{}'.format(stype)
            # decompose links like D01234+D23456
            links = link.split('+')
            ecnt += len(links) - 1
            links = links[0].split(':')
            if len(links) == 1:  link = 'MESHD:{}'.format(link)
            # check entity name
            if verbose and name != text[int(spos):int(epos)]:
                print('PMID: {}'.format(pmid))
                print('Annotation: {}\tText: {}'.format(name, text[int(spos):int(epos)]))
                name = text[int(spos):int(epos)]
            eline = '\t'.join([pmid, eid, type, spos, epos, name, link])
            elines.append(eline)
    if verbose and ecnt > 0:  print('{} link entities discarded'.format(ecnt))
    file_list2line(tlines, tfilename, verbose=verbose)
    file_list2line(elines, efilename, verbose=verbose)
    return

# 3115150	1584	1626	cardiovascular, and respiratory depression	Disease	D002318|D012131	cardiovascular depression|respiratory depression
# 15579441	1134	1159	tubulointerstitial injury	Disease	-1
# 6794356	CID	D016651	D003490
def convert_cdr_corpus(wdir, cps_file, verbose=0):
    ofilename ='{}/{}.org'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    rfilename = '{}/{}.rel'.format(wdir, cps_file)
    ecfilename = '{}/{}-c.ent'.format(wdir, cps_file)
    edfilename = '{}/{}-d.ent'.format(wdir, cps_file)
    #
    slines = file_line2list(ofilename, verbose=verbose)
    tlines, elines, rlines, ecnt = [], [], [], 0
    eclines, edlines = [], []
    pmid, title, abs, eno, text = None, None, None, 1, ''
    for sline in slines:
        if sline == '':  pmid, title, abs, eno, text = None, None, None, 1, ''
        elif not pmid:  pmid, _, title = sline.split('|')
        elif not abs:
            _, _, abs = sline.split('|')
            if abs.endswith('..'):  abs = abs[:-1]
            elif not abs.endswith('.'):  abs = abs + '.'
            tlines.append('\t'.join([pmid, title, abs]))
            text = ' '.join([title, abs])
        else:
            tokens = sline.split('\t')
            if len(tokens) == 4:    # relation
                _, type, eid1, eid2 = tokens
                rid = '{}-{}'.format(eid1, eid2)
                rlines.append('\t'.join([pmid, rid, eid1, eid2, type, type]))
            else:
                _, spos, epos, name, type, link = tokens[:6]
                type = type[:4].upper()
                eid = 'T{}'.format(eno)
                eno += 1
                # decompose links like D01234|D23456
                links = link.split('|')
                ecnt += len(links) - 1
                if len(links) == 1:  link = links[0]
                link = 'None' if link == '-1' else 'MESHD:{}'.format(link)
                # check entity name
                if verbose and name != text[int(spos):int(epos)]:
                    print('PMID: {}'.format(pmid))
                    print('Annotation: {}\tText: {}'.format(name, text[int(spos):int(epos)]))
                    name = text[int(spos):int(epos)]
                eline = '\t'.join([pmid, eid, type, spos, epos, name, link])
                elines.append(eline)
                if type == 'CHEM':  eclines.append(eline)
                elif type == 'DISE':  edlines.append(eline)
    if verbose and ecnt > 0:  print('{} link entities discarded'.format(ecnt))
    file_list2line(eclines, ecfilename, verbose=verbose)
    file_list2line(edlines, edfilename, verbose=verbose)
    file_list2line(tlines, tfilename, verbose=verbose)
    file_list2line(elines, efilename, verbose=verbose)
    file_list2line(rlines, rfilename, verbose=verbose)
    return


# prepare NER docset for CHEMD
def convert_chemd_cemp_corpus(wdir, cps_file, verbose=0):
    otfilename = '{}/{}.txt.org'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    oefilename = '{}/{}.ent.org'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)

    # load abstract
    docdict = {}
    tlines = file_line2array(otfilename, verbose=verbose)
    dlines = []
    for tline in tlines:
        if len(tline) < 3:  print(tline)
        pmid, title, tline = tline
        if wdir in ('CEMP', 'GPRO'):
            if not title.endswith('.'):  title += '.'
            dlines.append('\t'.join([pmid, title, tline]))
        docdict[pmid] = title
    if wdir in ('CEMP', 'GPRO'):  file_list2line(dlines, tfilename, verbose=verbose)

    # read entities
    olines = file_line2array(oefilename, verbose=verbose)
    elines, opmid, eno = [], '', 1
    for tokens in olines:
        pmid, ttype, spos, epos, name, stype = tokens[:6]
        link = tokens[-1] if wdir =='GPRO' else 'None'  # there are 7 columns in GPRO
        if pmid != opmid:   # refresh entity mention no
            opmid, eno = pmid, 1
        # prepare entity parameters
        if pmid not in docdict:
            print(tokens)
            continue
        title = docdict[pmid]
        offset = len(title)+1 if ttype=='A' else 0   # 'T'-title, 'A'-abstract
        eid = 'T{}'.format(eno)
        eno += 1
        type = '{}|{}'.format('GENE' if wdir =='GPRO' else 'CHEM', stype)
        spos = '{}'.format(offset+int(spos))
        epos = '{}'.format(offset+int(epos))
        #
        elines.append('\t'.join([pmid, eid, type, spos, epos, name, link]))
    file_list2line(elines, efilename, verbose=verbose)

    return

# prepare NER docset for BC2GM
def convert_bc2gm_corpus(wdir, cps_file, verbose=0):
    otfilename = '{}/{}.txt.org'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    oefilename = '{}/{}.ent.org'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    # read text
    docset= ieDocSet()
    slines = file_line2list(otfilename, verbose=verbose)
    tlines = []
    for sline in slines:
        words = sline.split()
        text = sline[len(words[0]) + 1:]
        doc = DocumentIE(id=words[0], text=text)
        docset.append_doc(did=words[0], doc=doc)
        tlines.append('\t'.join([words[0], text]))
    file_list2line(tlines, tfilename, verbose=verbose)
    # read entities
    olines = file_line2array(oefilename, sepch='|', verbose=verbose)
    elines, opmid, eno = [], '', 1
    for pmid, pos, name in olines:
        ospos, oepos = pos.split()
        ospos, oepos = int(ospos), int(oepos)
        if pmid != opmid:   # refresh entity mention no
            opmid, eno = pmid, 1
        # prepare entity parameters
        eid = 'T{}'.format(eno)
        eno += 1
        doc = docset.contains_doc(pmid)
        if doc is None:
            print('Error in {}'.format(pmid))
        text = doc.text
        tpos, spos, epos = 0, -1, -1
        for i, ch in enumerate(text):
            if ch.isspace():  continue
            if tpos == ospos and spos < 0:  spos = i
            if tpos == oepos and epos < 0:  epos = i+1
            if spos >= 0 and epos >= 0:  break
            tpos += 1
        if name != doc.text[spos:epos]:
            print('{}[{}:{}], [{}:{}]'.format(pmid, ospos, oepos, spos, epos), '"{}"'.format(name), '"{}"'.format(doc.text[spos:epos]))
        elines.append('\t'.join([pmid, eid, 'GENE', str(spos), str(epos), name, 'None']))
    file_list2line(elines, efilename, verbose=verbose)
    return

# prepare NER docset for GAD
def convert_gad_corpus(wdir, cps_file, rvsID=True, blindID=False, verbose=0):
    otfilename = '{}/{}.txt.org'.format(wdir, cps_file)
    tfilename = '{}/{}.wrd'.format(wdir, cps_file)
    # read entities and text
    olines = file_line2array(otfilename, verbose=verbose)
    tlines, opmid, eno = [], '', 1
    for tokens in olines:
        for i, token in enumerate(tokens):
            if token.startswith('"') or token.endswith('"'):
                tokens[i] = tokens[i][1:-1]
        sid, ename1, pos1, ename2, pos2, text = tokens[0], tokens[5], tokens[6], tokens[-3], tokens[-2], tokens[-1]  # sentence id
        rtype = 'GAD' if tokens[1] == 'Y' else 'None'
        spos1, epos1 = pos1.split('#')
        spos1, epos1 = int(spos1), int(epos1)
        spos2, epos2 = pos2.split('#')
        spos2, epos2 = int(spos2), int(epos2)
        text = text.replace('""', '"')
        isrvs, sep1, sep2, type1, type2 = False, '#', '@', 'GENE', 'DISE'
        if rvsID and spos2 < spos1:
            spos1, epos1, ename1, type1, spos2, epos2, ename2, type2 = spos2, epos2, ename2, type2, spos1, epos1, ename1, type1
            isrvs = True
            #print(tokens)
        # check entity mention
        if ename1 != text[spos1:epos1]:
            print('{}-E1({}): "{}"\t"{}"'.format(sid, pos1, ename1, text[spos1:epos1]))
        if ename2 != text[spos2:epos2]:
            print('{}-E2({})："{}"\t"{}"'.format(sid, pos2, ename2, text[spos2:epos2]))
        # tokenize
        if blindID:
            ename1, ename2 = type1, type2
        #line = '{} {} {} {} {} {} {} {} {}'.format(text[:spos1], sep1, ename1, sep1, text[epos1:spos2], sep2, ename2, sep2, text[epos2:])
        line = ' '.join(['{}']*9).format(text[:spos1], sep1, ename1, sep1, text[epos1:spos2], sep2, ename2, sep2, text[epos2:])
        line = tokenize_bio_sentence(line)
        words = line.split()
        # get the new positions
        pos = [-1] * 4  # spos1, epos1, spos2, epos2
        for i, word in enumerate(words):
            if word == sep1:
                if pos[0] < 0:  pos[0] = i+2
                elif pos[1] < 0: pos[1] = i
            elif word == sep2:
                if pos[2] < 0:  pos[2] = i+2
                elif pos[3] < 0: pos[3] = i
        tlines.append('\t'.join(['ID', sid, '0', 'E1', 'E2', '4', str(len(words))]))
        if isrvs and rtype != 'None':
            tlines.append('{}\tR\t{}'.format('TYPE', rtype))
        else:
            tlines.append('{}\t{}'.format('TYPE', rtype))
        tlines.append('\t'.join(['E1', type1, ename1, str(pos[0]), str(pos[1])]))
        tlines.append('\t'.join(['E2', type2, ename2, str(pos[2]), str(pos[3])]))
        tlines.extend(['{}\t{}'.format(i+1, word) for i, word in enumerate(words)])
        tlines.append('')
    file_list2line(tlines, tfilename, verbose=verbose)
    return

# prepare NER docset for GAD
def convert_euadr_corpus(wdir, cps_file, rvsID=True, blindID=False, verbose=0):
    otfilename = '{}/{}.txt.org'.format(wdir, cps_file)
    tfilename = '{}/{}.wrd'.format(wdir, cps_file)
    # read entities and text
    olines = file_line2array(otfilename, verbose=verbose)
    tlines, opmid, eno = [], '', 1
    for tokens in olines:
        for i, token in enumerate(tokens):
            if token.startswith('"') or token.endswith('"'):
                tokens[i] = tokens[i][1:-1]
                tokens[i] = tokens[i].replace('""', '"')
        rtype, pmid, sno, ename1, spos1, epos1, etype1, ename2, spos2, epos2, etype2, text = tokens
        if rtype == 'NA':  rtype = 'None'
        spos1, epos1 = int(spos1), int(epos1)
        spos2, epos2 = int(spos2), int(epos2)
        etype1, etype2 = etype1[:4].upper().strip(), etype2[:4].upper().strip()
        #
        isrvs, sep1, sep2  = False, '#', '@'
        if rvsID and spos2 < spos1:
            spos1, epos1, ename1, etype1, spos2, epos2, ename2, etype2 = spos2, epos2, ename2, etype2, spos1, epos1, ename1, etype1
            isrvs = True
            #print(tokens)
        # check entity mention
        if ename1 != text[spos1:epos1]:
            print('{}-E1({}:{}): "{}"\t"{}"'.format(pmid, spos1, epos1, ename1, text[spos1:epos1]))
        if ename2 != text[spos2:epos2]:
            print('{}-E2({}:{})："{}"\t"{}"'.format(pmid, spos2, epos2, ename2, text[spos2:epos2]))
        # tokenize
        if blindID:
            ename1, ename2 = etype1, etype2
        #line = '{} {} {} {} {} {} {} {} {}'.format(text[:spos1], sep1, ename1, sep1, text[epos1:spos2], sep2, ename2, sep2, text[epos2:])
        line = ' '.join(['{}']*9).format(text[:spos1], sep1, ename1, sep1, text[epos1:spos2], sep2, ename2, sep2, text[epos2:])
        line = bio_sentence_tokenize(line)
        words = line.split()
        # get the new positions
        pos = [-1] * 4  # spos1, epos1, spos2, epos2
        for i, word in enumerate(words):
            if word == sep1:
                if pos[0] < 0:  pos[0] = i+2
                elif pos[1] < 0: pos[1] = i
            elif word == sep2:
                if pos[2] < 0:  pos[2] = i+2
                elif pos[3] < 0: pos[3] = i
        tlines.append('\t'.join(['ID', pmid, str(sno), 'E1', 'E2', '4', str(len(words))]))
        if isrvs and rtype != 'None':
            tlines.append('{}\tR\t{}'.format('TYPE', rtype))
        else:
            tlines.append('{}\t{}'.format('TYPE', rtype))
        tlines.append('\t'.join(['E1', etype1, ename1, str(pos[0]), str(pos[1])]))
        tlines.append('\t'.join(['E2', etype2, ename2, str(pos[2]), str(pos[3])]))
        tlines.extend(['{}\t{}'.format(i+1, word) for i, word in enumerate(words)])
        tlines.append('')
    file_list2line(tlines, tfilename, verbose=verbose)
    return


# prepare NER docset for LINN
def convert_linn_corpus(wdir, cps_file, verbose=0):
    oefilename = '{}/{}.ent.org'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    # read entities
    olines = file_line2array(oefilename, verbose=verbose)
    elines, opmid, eno = [], '', 1
    for tokens in olines[1:]:
        link, pmid, spos, epos, name = tokens[:5]
        links = link.split(':')
        link = ':'.join(links[1:])
        if pmid != opmid:   # refresh entity mention no
            opmid, eno = pmid, 1
        # prepare entity parameters
        eid = 'T{}'.format(eno)
        eno += 1
        type = 'SPECIES'
        #
        elines.append('\t'.join([pmid, eid, type, spos, epos, name, link]))
    file_list2line(elines, efilename, verbose=verbose)
    return


# prepare NER docset for LINN
def convert_s800_corpus(wdir, cps_file, verbose=0):
    oefilename = '{}/{}.ent.org'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    # read species001:pmid
    pfilename = '{}/pubmedid.txt'.format(wdir)
    plines = file_line2array(pfilename)
    spiddict = {pline[0]:pline[1][5:] for pline in plines}
    # read text
    docset = ieDocSet()
    mypath = os.path.join(wdir, cps_file)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.txt')]
    tlines =[]
    for file in files:
        spid = file[:-4]
        pmid = ''
        if spid in spiddict:
            pmid = spiddict[spid]
            #print(pmid)
        else:
            print('{} does not exist!'.format(spid))
            continue
        sfilename = os.path.join(mypath, file)
        text = file_line2list(sfilename, stripID=False)
        tlines.append('\t'.join([pmid]+[text[0].strip(), text[2].strip()]))
        docset.append_doc(pmid, DocumentIE(id=pmid, title=text[0], text=text[0].strip()+' '+text[2].strip()))
        #print(text)
    file_list2line(tlines, tfilename, verbose=verbose)
    # read entities
    olines = file_line2array(oefilename, verbose=verbose)
    elines, opmid, eno = [], '', 1
    for tokens in olines:
        link, pmid, spos, epos, name = tokens[:5]
        pmid = pmid.split(':')[1]
        link = 'NCBI:{}'.format(link)
        if pmid != opmid:   # refresh entity mention no
            opmid, eno = pmid, 1
        # prepare entity parameters
        eid = 'T{}'.format(eno)
        eno += 1
        type = 'SPECIES'
        doc = docset.contains_doc(pmid)
        spos, epos = int(spos), int(epos)
        offset = len(doc.title) - len(doc.title.strip())
        #print(offset)
        if spos >= len(doc.title):
            spos -= offset+1
            epos -= offset+1
        epos += 1
        if name != doc.text[spos:epos]:
            print(tokens)
            print(len(doc.title), spos, epos)
            print('Ent:'+name, 'Txt:'+doc.text[spos:epos])
        #
        elines.append('\t'.join([pmid, eid, type, str(spos), str(epos), name, link]))
    file_list2line(elines, efilename, verbose=verbose)
    #
    return


# preprocess for cpr-style annotation, *.txt, *.ent, *.rel
# sentence splitting & tokenization, generating ann & txt individual files
def convert_cpr_corpus(wdir, cps_file, verbose=0):
    docset = ieDocSet(id=wdir)
    # read abstracts
    tfilename = '%s/%s.txt' % (wdir, cps_file)
    docset.load_docset_bio_abstract(ieDoc, tfilename, verbose=verbose)
    # read entities
    efilename = '%s/%s.ent' % (wdir, cps_file)
    oefilename = '%s/%s.ent.org' % (wdir, cps_file)
    oelines = file_line2array(oefilename, verbose=verbose)
    elines, rlines = [], []
    for pmid, id, type, spos, epos, name in oelines:
        elines.append('\t'.join([pmid, id, type, spos, epos, name, 'None']))

    # read relations
    rfilename = '%s/%s.rel' % (wdir, cps_file)
    orfilename = '%s/%s.rel.org' % (wdir, cps_file)
    orlines = file_line2array(orfilename, verbose=verbose)
    for pmid, type, valstr, name, arg1, arg2 in orlines:
        # 10082498, CPR:4, Y, INDIRECT-DOWNREGULATOR, Arg1:T19, Arg2: T32
        types, evalid = type.split(':'), valstr.startswith('Y')
        if not evalid:  continue
        #
        args1, args2 = arg1.split(':'), arg2.split(':')
        eid1, eid2 = args1[1], args2[1]
        rid = '{}-{}'.format(eid1, eid2)
        rlines.append('\t'.join([pmid, rid, eid1, eid2, types[1], name]))
    # save
    file_list2line(elines, efilename, verbose=verbose)
    file_list2line(rlines, rfilename, verbose=verbose)
    return


# prepare BioASQ docset for training or prediction
def convert_asq_corpus(wdir, cps_file, verbose=0):
    # read abstracts
    jfilename = '{}/{}.json'.format(wdir, cps_file)
    tfilename = '{}/{}.txt'.format(wdir, cps_file)
    efilename = '{}/{}.ent'.format(wdir, cps_file)
    #
    tdict = load_json_file(jfilename)
    # read paragraphs
    paralist = tdict['data'][0]['paragraphs']
    qacnt = 0
    tlines, elines = [], []
    for i, para in enumerate(paralist):
        context = para['context']
        # read qa pairs
        qalist = para['qas']
        qacnt += len(qalist)
        for qad in qalist:
            # construct an abstract for every question-context pair
            qid, q = qad['id'], qad['question'].replace('\n', ' ')
            tlines.append('\t'.join([qid, q, context]))  # id, question, abstract
            # read multiple answers
            alist = qad['answers']
            if len(alist) != 1:  # zero or multiple answers
                print(alist)
            for j, a in enumerate(alist):
                text, spos = a['text'], a['answer_start']
                epos = spos + len(text)
                if text != context[spos:epos]:
                    print('Answer: {}\tText {}'.format(text, context[spos:epos]))
                elines.append('\t'.join([qid, 'T{}'.format(j+1), 'ANS', str(spos), str(epos), text, 'None']))
            #print(qa)
    file_list2line(tlines, tfilename, verbose=verbose)
    file_list2line(elines, efilename, verbose=verbose)
    return

def convert_jnlpba_corpus(wdir, cps_file, verbose=0):
    sfilename = '{}/{}.org'.format(wdir, cps_file)
    dfilename = '{}/{}.iob'.format(wdir, cps_file)
    #
    slines = file_line2array(sfilename, verbose=verbose)
    for sline in slines:
        if len(sline) >= 2 and len(sline[-1]) > 2:
            type = sline[-1][2:]
            if type == 'protein':  type = 'GENE'
            elif type == 'cell_line':  type = 'CLLN'
            elif type == 'cell_type':  type = 'CLTP'
            sline[-1] = sline[-1][:2]+type
    dlines = ['\t'.join(sline) for sline in slines]
    file_list2line(dlines, dfilename, verbose=verbose)
    return

def convert_bio_corpus(wdir, cpsfiles, verbose=0):
    if wdir == 'NCBI':
        for cf in cpsfiles:  convert_ncbi_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'CDR':
        for cf in cpsfiles:  convert_cdr_corpus(wdir, cf, verbose=verbose)
    elif wdir in ('CHEMD', 'CEMP', 'GPRO'):
        for cf in cpsfiles:  convert_chemd_cemp_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'BC2GM':
        #convert_bc2gm_corpus(wdir, 'test', verbose=verbose)
        for cf in cpsfiles:  convert_bc2gm_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'GAD':
        for cf in cpsfiles:  convert_gad_corpus(wdir, cf, rvsID=True, blindID=False, verbose=verbose)
    elif wdir == 'EUADR':
        for cf in cpsfiles:  convert_euadr_corpus(wdir, cf, rvsID=True, blindID=False, verbose=verbose)
    elif wdir == 'CPR':
        convert_cpr_corpus(wdir, 'sample', verbose=verbose)
        for cf in cpsfiles:  convert_cpr_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'LINN':
        for cf in cpsfiles:  convert_linn_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'S800':
        for cf in cpsfiles:  convert_s800_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'BIOASQ4B':
        for cf in cpsfiles:  convert_asq_corpus(wdir, cf, verbose=verbose)
    elif wdir == 'JNLPBA':
        for cf in cpsfiles:  convert_jnlpba_corpus(wdir, cf, verbose=verbose)
    return

