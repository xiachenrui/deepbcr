###########################################
# Functions for formatting TRUST BCR data #
###########################################

from __future__ import print_function

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from deep_bcr import *


###############################################################################
## Helper Functions for Basic Staffs

def read_fa_seq(fa_file):
    """ Read fastq file and yield sequences"""
    with open(fa_file) as tmp:
        s = ''
        for line in tmp:
            if line.startswith('>'):
                yield s
            else:
                s += line.strip()
        yield s

def read_meta(meta_file='../data/TCGA/tcga_clinical_CC.csv'):
    """ Return the clinical information indexed by patient ids"""
    meta = pd.read_csv(meta_file)
    print(list(meta))
    return meta.set_index('patient').T.to_dict()


def load_data(meta, case, ratio=0.5,
              min_samples_to_model=0,
              datapath='../data/',
              IGH_tab='../data/TCGA/tcga_bcrh_v20180405.txt.gz',
              IGL_tab='../data/TCGA/tcga_bcrl_v20190102.txt.gz'):

    info = os.path.join(datapath, case+'_info.csv')
    train = os.path.join(datapath, case+'_train.pkl.gz')
    test = os.path.join(datapath, case+'_test.pkl.gz')

    if os.path.exists(info) and os.path.exists(train) and os.path.exists(test):
        return pd.read_csv(info), pd.read_pickle(train), pd.read_pickle(test)

    ## Annotate light chain isotypes
    trust_tab = pd.read_csv(IGL_tab, sep='\t')
    for light in ['IGK','IGL']:
        for gene in ['Cgene','Jgene','Vgene']:
            trust_tab.loc[trust_tab[gene].str.contains('^'+light, na=False), 'Cgene'] = light

    trust_tab = trust_tab.append(pd.read_csv(IGH_tab, sep='\t'), ignore_index=True)
    data = extract_bcr(trust_tab, rep_col=['CDR3_aa', 'Cgene'])

    ## Select the right subset to work on
    if case == 'TCGA_BCR': ## default case
        pid_sel = [i for i in data.index if len(data[i]) >= 100]
        data = data[pid_sel] ## select by repertoire size
        tumors = get_label_map(meta, data.index.tolist(), min_count=0)
        pid_sel = [(pid2label(i, meta) in tumors) for i in data.index]
        min_samples_to_model = 300 ## compatable with TCGA_Seq30Min300

    elif case == 'TCGA_Seq30Min300':
        pid_sel = [i for i in data.index if len(data[i]) >= 30]
        data = data[pid_sel] ## select by repertoire size
        tumors = get_label_map(meta, data.index.tolist(), min_count=300)
        pid_sel = [(pid2label(i, meta) in tumors) for i in data.index]

    elif case == 'TCGA_BRCAsub':
        tumors = get_label_map(meta, data.index.tolist(), min_count=0)
        tumors = [t for t in tumors if t.startswith('BRCA')]
        pid_sel = [(pid2label(i, meta) in tumors) for i in data.index]

    elif case == 'TCGA_LungStomach':
        tumors = get_label_map(meta, data.index.tolist(), min_count=0)
        tumors = [t for t in tumors if t in ['LUAD','LUSC','STAD','Normal']]
        pid_sel = [(pid2label(i, meta) in tumors) for i in data.index]

    else:
        raise ValueError('Unknown case '+case)

    data = data[pid_sel]

    ## Randomly split the dataset by PID
    pids = data.index.str.replace('Normal_','')
    upids = np.array(pids.unique())
    random = np.random.RandomState(seed=0)
    random.shuffle(upids)

    idx = int(len(upids)*ratio)
    train_data = data[pids.isin(upids[:idx])]
    test_data  = data[pids.isin(upids[idx:])]
    print(len(train_data), len(test_data))

    train_tumors = [pid2label(i, meta) for i in train_data.index]
    test_tumors  = [pid2label(i, meta) for i in test_data.index]

    train_count = [train_tumors.count(i) for i in tumors]
    test_count  = [test_tumors.count(i) for i in tumors]

    data_info = pd.DataFrame(list(zip(tumors, train_count, test_count)), columns=['label', 'train', 'test'])
    data_info['total'] = data_info['train']+data_info['test']
    data_info.to_csv(info+'_complete.csv', index=False)
    data_info = data_info[data_info['total'] > min_samples_to_model]
    data_info.to_csv(info, index=False)
    train_data.to_pickle(train)
    test_data.to_pickle(test)
    return data_info, train_data, test_data


###############################################################################
## A Set of Functions for Encoding Amino Acid Sequences

AA_LIST = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
AA_CHEM = {'A':[-0.591, -1.302, -0.733, 1.57, -0.146],
           'C':[-1.343, 0.465, -0.862, -1.02, -0.255],
           'D':[1.05, 0.302, -3.656, -0.259, -3.242],
           'E':[1.357, -1.453, 1.477, 0.113, -0.837],
           'F':[-1.006, -0.59, 1.891, -0.397, 0.412],
           'G':[-0.384, 1.652, 1.33, 1.045, 2.064],
           'H':[0.336, -0.417, -1.673, -1.474, -0.078],
           'I':[-1.239, -0.547, 2.131, 0.393, 0.816],
           'K':[1.831, -0.561, 0.533, -0.277, 1.648],
           'L':[-1.019, -0.987, -1.505, 1.266, -0.912],
           'M':[-0.663, -1.524, 2.219, -1.005, 1.212],
           'N':[0.945, 0.828, 1.299, -0.169, 0.933],
           'P':[0.189, 2.081, -1.628, 0.421, -1.392],
           'Q':[0.931, -0.179, -3.005, -0.503, -1.853],
           'R':[1.538, -0.055, 1.502, 0.44, 2.897],
           'S':[-0.228, 1.399, -4.76, 0.67, -2.647],
           'T':[-0.032, 0.326, 2.213, 0.908, 1.313],
           'V':[-1.337, -0.279, -0.544, 1.242, -1.262],
           'W':[-0.595, 0.009, 0.672, -2.128, -0.184],
           'Y':[0.26, 0.83, 3.097, -0.838, 1.512]}

def encode_aa_seq_index(seq):
    """ Encode an amino acid sequence by its indexes """
    return [AA_LIST.index(i) if i in AA_LIST else 0 for i in seq]

def encode_aa_seq_binary(seq):
    """ Encode an amino acid sequence into a binary vector """
    length = len(AA_LIST)
    values = np.zeros(length*len(seq), dtype='float')
    for index, aa in enumerate(seq):
        if aa not in AA_LIST:
            continue ## all zeros
        values[length*index + AA_LIST.index(aa)] = 1.0
    return values

def encode_aa_seq_binary_ext_dim(seq):
    """ Encode the amino acid into an extrat dimension """
    v = encode_aa_seq_binary(seq)
    i = len(seq)
    j = len(v)//i
    return v.reshape((i,j))

def encode_aa_seq_atchley(seq):
    """ Encode an amino acid sequence into a vector of atchley factors """
    length = len(AA_CHEM)
    values = []
    for index, aa in enumerate(seq):
        values += AA_CHEM.get(aa, [0.0, 0.0, 0.0, 0.0, 0.0])
    return values


###############################################################################
## Helper Functions for Preparing the Input Data for Further Formatting

def to_list(df):
    if len(df.shape) == 1:
        return df.tolist()
    else:
        return [tuple(x) for x in df.values]

def extract_bcr(tab, rep_col='CDR3_aa'):
    """ Extract BCR repertorie for each patient 

        Args:
            tab: data table from TRUST BCR outputs
            rep_col: 'CDR3_aa' or 'complete_CDR3_sequences' or a list of keys

        Output: a Series vector containing lists of BCR CDR3 sequences
    """
    tab['patient'] = tab.TCGA_id.str.slice(0,12)
    tab['Sample_Type'] = tab.TCGA_id.str.slice(13,15)
## https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
## Code	Definition	Short Letter Code
## 01	Primary Solid Tumor	TP
## 02	Recurrent Solid Tumor	TR
## 03	Primary Blood Derived Cancer - Peripheral Blood	TB
## 04	Recurrent Blood Derived Cancer - Bone Marrow	TRBM
## 05	Additional - New Primary	TAP
## 06	Metastatic	TM
## 07	Additional Metastatic	TAM
## 08	Human Tumor Original Cells	THOC
## 09	Primary Blood Derived Cancer - Bone Marrow	TBM
## 10	Blood Derived Normal	NB
## 11	Solid Tissue Normal	NT
## 12	Buccal Cell Normal	NBC
## 40	Recurrent Blood Derived Cancer - Peripheral Blood	TRB
## 50	Cell Lines	CELL
## 60	Primary Xenograft Tissue	XP
## 61	Cell Line Derived Xenograft Tissue	XCL
    tumor = tab[tab.Sample_Type.isin(['01','02','06','07'])] ## select tumor samples
    normal = tab[tab.Sample_Type.isin(['10','11'])] ## select normal samples
    normal['patient'] = 'Normal_'+normal['patient'] ## rename Normal sample Ids

    print('Tumor data of shape', tumor.shape)
    print('Normal data of shape', normal.shape)
    out = [ tumor.groupby('patient')[rep_col].apply(to_list), 
           normal.groupby('patient')[rep_col].apply(to_list) ]
    return pd.concat(out)

def rep2kmer(rep, k=6):
    """ Encode a repertoire using kmers 

        Args:
            rep:    a list of BCR CDR3s
            k:      the size of the kmer

        Returns: 
            list: a list of kmers
    """
    germ = set()
    out = []
    info = {}
    for s in rep:
        if type(s) == float:
            continue
        elif type(s) == str:
            for i in range(len(s)-k+1):
                mer = s[i:(i+k)]
                if mer not in info and mer not in germ:
                    ## this simple trick will make the order in `out`
                    ## are consistant among different machines
                    info[mer] = []
                    out.append(mer)
        elif type(s) == tuple:
            seq, gene = s
            if type(seq) == float:
                continue
            for i in range(len(seq)-k+1):
                mer = seq[i:(i+k)]
                if mer not in germ:
                    if mer not in info:
                        out.append(mer)
                    cc = info.get(mer, {})
                    cc[gene] = cc.get(gene,0)+1
                    info[mer] = cc
    return out, info


###############################################################################
## A Set of Functions for Mapping Patient Id to Clinical Annotations

def pid2label(pid, meta, use_subtype=True):
    """ Map patient id to tumor type """
    prefix = ''
    if pid.startswith('Normal_TCGA'):
        prefix = 'Normal-'
        pid = pid.replace('Normal_TCGA','TCGA')
    if not pid.startswith('TCGA') and pid not in meta:
        if '_' not in pid: ## no idea
            return 'Unknown'
        return pid.split('_')[0] ## other datasets with the [Group]_[ID] format
    if pid not in meta: ## no entry
        return 'Unknown'
    cancer = meta[pid].get('cancer', np.nan)
    if pd.isnull(cancer): ## no record
        return 'Unknown'
    ## Reference: https://www.nature.com/articles/nature11252
    if cancer in ['READ','COAD']: ## merge the two closely-related tumor types
        cancer = 'COAD-READ'
    subtype = meta[pid].get('subtype', np.nan)
    if pd.isnull(subtype) or not use_subtype:
        return prefix+cancer
    if cancer == 'BRCA':
        if subtype in ['LumA','LumB','Basal']:
            if subtype.startswith('Lum'): ## combine two luminal types
                subtype = 'Lum'
            return prefix+cancer+'-'+subtype
        else:
            return prefix+cancer+'-TriN'
    return prefix+cancer

def pid2survival(pid, meta, neg_censor=True, use_log_ratio=True):
    """ Map patient id to overall survival """
    if pid not in meta:
        return None
    os = meta[pid].get('OS', np.nan)
    event = meta[pid].get('EVENT', np.nan)
    if pd.isnull(os) or pd.isnull(event):
        return None
    os = float(os)
    event = int(event) ## 1 means death and 0 means living
    if use_log_ratio:
        ## Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/
        os = np.log2(os+1)
    if neg_censor:
        return os if event == 1 else -os
    return os

def pid2response(pid, meta, keys=['CR','PR','SD','PD'], use_int=True):
    """ Map patient id to treatment response """ 
    if pid not in meta:
        return None
    response = meta[pid].get('drug_response', np.nan)
    if pd.isnull(response):
        return None
    if use_int:
        return int(response-1)
    else:
        return keys[int(response-1)]


###############################################################################
## A Set of Functions for Pareparing the Environment for Mapping Patient Ids

def setup_tumor_types(meta, tumors, verbose=False):
    """ Return a function for mapping pids to tumor types """
    def translate(pid):
        label = pid2label(pid, meta)
        if verbose:
            if label in tumors:
                return pid, label, tumors.index(label)
            return pid, label, -1 ## background
        else:
            if label in tumors:
                return tumors.index(label)
            return -1 ## background
    return translate

def setup_survival(meta, tumors, neg_censor=True, verbose=False):
    """ Return a function for mapping pids to survival time """
    def translate(pid):
        label = pid2label(pid, meta)
        time = pid2survival(pid, meta, neg_censor)
        if verbose:
            if label in tumors:
                return pid, label, time
            return pid, label, None
        else:
            if label in tumors:
                return time
            return None
    return translate

def setup_response(meta, tumors, neg_censor=True, verbose=False):
    """ Return a function for mapping pids to response types """
    def translate(pid):
        label = pid2label(pid, meta)
        response = pid2response(pid, meta)
        if verbose:
            if label in tumors:
                return pid, label, response
            return pid, label, None
        else:
            if label in tumors:
                return response
            return None
    return translate

###############################################################################
## A Set of Functions for Encoding the K-mer Count Table

def encode_count_binary(info, mer):
    if mer in info:
        return 1.0
    else:
        return 0.0

def get_isotype_map():
## The full list of Ig constant genes are
## ['IGHM','IGHD','IGHG3','IGHG1','IGHEP1','IGHA1','IGHGP','IGHG2','IGHG4','IGHE','IGHA2']

    gmap = {'IGHM' : 'IGHM|IGHD',
            'IGHD' : 'IGHM|IGHD',
            'IGHG1': 'IGHG1',
            'IGHG2': 'IGHG2/4',
            'IGHG3': 'IGHG3',
            'IGHG4': 'IGHG2/4',
            'IGHA1': 'IGHA1/2',
            'IGHA2': 'IGHA1/2',
            'IGK'  : 'IGK',
            'IGL'  : 'IGL'}
    new = sorted(set(gmap.values()))+['Others']

    return gmap, new

def encode_count_genes(info, mer):
    gmap, new = get_isotype_map()

    if mer in info:
        counts = info[mer]
        agg = {}
        for g in counts:
            if type(g)==str and g in gmap:
                iso = gmap[g]
            else:
                iso = 'Others'
            agg[iso] = agg.get(iso,0) + counts[g]
        out = [agg.get(g,0) for g in new]
        return np.array(out, dtype='int')
    return np.zeros(len(new), dtype='int')


###############################################################################
## Helper Funtion for Breaking the Whole Data into Data Batches

def get_next_batch(data, get_label, size=10, kmer=6, min_num=10, return_idx=False):
    """ Get the next batch of data in the same shape

        Args:
            data: raw data set
            get_label: function for getting the valid labels
            size: number of data points
            kmer: size of the kmers
            min_num: minimual number of kmers
            return_idx: also return the pid indexes

        Yields:
            xs: (size, size, kmer)
            cs: (size)
            ys: (size)

            idx: (size) when return_idex=True
    """
    xs = []; cs = []; ys = []; idx = []
    for pid, rep in data.iteritems():
        label = get_label(pid)
        if label is None: ## ignore missing labels
            continue

        mer, info = rep2kmer(rep, k=kmer)
        if len(mer) < min_num:
            continue
            
        xs.append(mer)
        cs.append(info)
        ys.append(label)
        idx.append(pid)

        if len(xs) == size:
            if return_idx:
                yield xs, cs, ys, idx
            else:
                yield xs, cs, ys
            xs = []; cs = []; ys = []; idx = []

    if len(xs) > 0:
        if return_idx:
            yield xs, cs, ys, idx
        else:
            yield xs, cs, ys


###############################################################################
## A Set of Master Functions for Packaging the Data Batch into Data Matrices

def __get_feature_size(xs):
    ## Try to get the third dimension size
    k = 0
    for mers in xs:
        for m in mers:
            k = len(m)
            break
        if k > 0:
            break
    if k == 0:
        raise ValueError('Please provide non-empty data')
    return k

def format_trim_dims(data, dim=None, seed=0, 
                     encode=encode_aa_seq_atchley, 
                     count=encode_count_binary):

    """ Fill in the zeros or trim extra dimensions 

        Args: (x, c, y) and the expected dimension

        Returns: (xs, cs, ys)
            
    """
    if dim is not None: ## for maximum dim, no need to shuffle
        random = np.random.RandomState(seed)
    else:
        random = None
    x, c, y = data[:3]
    if dim is None:
        dim = max([len(i) for i in x])
    null = ''.join(['*']*__get_feature_size(x))

    xs = []; cs = []
    for i in range(len(y)):
        mers = x[i]
        s1 = min(dim, len(mers)) ## size of kept values
        s2 = dim - s1            ## size for padding
        if random is not None:
            random.shuffle(mers)
        new = mers[:s1] + [null]*s2
        xs.append(np.array([encode(mer) for mer in new]))
        cs.append(np.array([count(c[i],mer) for mer in new]))
    return np.array(xs), np.array(cs), np.array(y)

def format_balance_labels(data, dim=None, seed=0, 
                          encode=encode_aa_seq_atchley,
                          count=encode_count_binary):

    """ Balance the number of data points for each label type

        Args: (x, c, y) and the expected dimension

        Returns: (xs, cs, ys)
            
    """
    random = np.random.RandomState(seed)
    x, c, y = data
    if dim is None:
        dim = max([len(i) for i in x])
    null = ''.join(['*']*__get_feature_size(x))

    ## sampling and get the indexes
    y = np.array(y, dtype='int')
    n = len(y)
    p = np.zeros(n, dtype='float')
    for i in np.unique(y):
        p[y == i] = n / float(sum(y == i)) ## inverse prob. to balance sizes
    p += 1.0 ## posudo-count
    p /= p.sum()
    idx = random.choice(n, size=n, replace=True, p=p)

    xs = []; cs = []
    for i in idx:
        mers = x[i]
        s1 = min(dim, len(mers)) ## size of kept values
        s2 = dim - s1            ## size for padding
        random.shuffle(mers)
        new = mers[:s1] + [null]*s2
        xs.append(np.array([encode(mer) for mer in new]))
        cs.append(np.array([count(c[i],mer) for mer in new]))
    return np.array(xs), np.array(cs), y[idx]


###############################################################################
## Hepler Function for Iterating Over the Whole Data and Yield Data Matrices

def sample_data_iter(data, data_size, kmer_size, max_num_kmer,
                     label_fun=setup_tumor_types,
                     format_fun=format_balance_labels, 
                     encode_fun=encode_aa_seq_binary_ext_dim,
                     count_fun=encode_count_binary):

    """ Iterations over all samples in the data

        The random seed is fixed in each run

        Args:
            data: raw input data
            data_size: batch size
            kmer_size: k of the mers
            max_num_kmer: number of kmers in each sampling

            label_fun: formating labels to predict for
            format_fun: formating function on repertoires
            encode_fun: encoding function for kmer sequences
            count_fun: encoding function for kmer counts

        Returns:
            (xs, cs, ys)
    """
    i = 0
    random = np.random.RandomState(0) ## fixed seed
    idx = np.arange(len(data))
    while True:
        random.shuffle(idx)
        batch = get_next_batch(data[idx], get_label=label_fun, size=data_size, kmer=kmer_size)
        xs, cs, ys = format_fun(next(batch), dim=max_num_kmer, seed=i, encode=encode_fun, count=count_fun)
        if i == 0:
            print('Data shapes are', xs.shape, cs.shape, ys.shape)
        yield xs, cs, ys
        i += 1


###############################################################################
## Test Functions for Different Deep Learning Models

def get_label_map(meta, data=None, min_count=1):
    """ Get cancer names which have enough patient samples """
    count = {}
    if data is None:
        data = meta
    for i in data:
        label = pid2label(i, meta)
        count[label] = count.get(label,0) + 1
    count = sorted([(count[i],i) for i in count])
    print('Sample counts are:')
    out = []
    total = 0
    for j,i in count:
        if j >= min_count:
            print(i, '\t', j)
            total += j
            out.append(i)
    print('Total\t', total)
    return out[::-1]

def get_test_environment(data_file = '../data/TCGA/tcga_bcrh_tophat.txt.gz',
                         meta_file = '../data/TCGA/tcga_clinical_CC.csv',
                         save_file = '../data/TCGA/test_set.txt.gz',
                         test_cancers = ['LUSC', 'LUAD'],
                         test_numbers = 200000,
                         model_path = '../work/test_tcga',
                         col='CDR3_aa'):

    """ Create a new test dataset or load the saved test data 
    
        Retures:
            meta_file:   File for the meta data 
            train_set:   Half for training
            test_set:    Another half for testing
            pred_labels: Labels for prediction
            model_path:  Path for saving the models
    """

    if not os.path.exists(save_file):
        if not os.path.exists(data_file):
            raise ValueError('Please provide the data file at '+data_file)
        outs = []
        total_size = 0
        for chunk in pd.read_csv(data_file, chunksize=10**6, sep='\t'):
            tmp = chunk[chunk['Disease'].isin(test_cancers)]
            total_size += tmp.shape[0]
            outs.append(tmp)
            if total_size > test_numbers:
                break
        pd.concat(outs, axis=1).to_csv(save_file, sep='\t', index=False, compression='gzip')

    trust_tab = pd.read_csv(save_file, sep='\t')
    data = extract_bcr(trust_tab, rep_col=col)
    meta = read_meta(meta_file)
    labels = get_label_map(meta, data.index.tolist(), 10)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    random = np.random.RandomState(seed=0)
    data = data[random.permutation(data.shape[0])]
    cut = int(data.shape[0]/2)
    return meta, data.iloc[:cut, ], data.iloc[cut:, ], labels, model_path

def test_labels_model(max_epoch=5):
    meta, data_train, data_test, tumors, model_path = get_test_environment()

    model = MultipleLabelModel(num_labels=len(tumors), save_path=model_path)

    labels = setup_tumor_types(meta, tumors)
    encode = encode_aa_seq_atchley
    data_iter = sample_data_iter(data_train, 50, 6, 100, labels, format_balance_labels, encode)

    batch = get_next_batch(data_test, labels, size=None, kmer=6)
    xs, cs, ys = format_trim_dims(next(batch), dim=None, encode=encode)

    for i in range(max_epoch):
        train_res = model.train_batch(data_iter, max_iterations=(i+1)*100, it_step_size=20)
        print('> Performance of the multiple label model')
        test_res = model.test(xs, cs, ys)

def test_encode_model(max_epoch=5):
    meta, data_train, data_test, tumors, model_path = get_test_environment()

    ei = np.array([AA_CHEM[i] for i in AA_LIST], dtype='float')
    model = EncodingLayerModel(num_labels=len(tumors), encode_init=ei, save_path=model_path)

    labels = setup_tumor_types(meta, tumors)
    encode = encode_aa_seq_index
    data_iter = sample_data_iter(data_train, 50, 6, 100, labels, format_balance_labels, encode)

    batch = get_next_batch(data_test, labels, size=None, kmer=6)
    xs, cs, ys = format_trim_dims(next(batch), dim=None, encode=encode)

    for i in range(max_epoch):
        train_res = model.train_batch(data_iter, max_iterations=(i+1)*100, it_step_size=20)
        print('> Performance of the encoding layer model')
        test_res = model.test(xs, cs, ys)

def test_switch_model(max_epoch=5):
    meta, data_train, data_test, tumors, model_path = get_test_environment(col=['CDR3_aa', 'Cgene'])

    ei = np.array([AA_CHEM[i] for i in AA_LIST], dtype='float')
    model = GeneSwitchModelFast(num_labels=len(tumors), encode_init=ei, save_path=model_path)

    labels = setup_tumor_types(meta, tumors)
    encode = encode_aa_seq_index
    count = encode_count_genes
    data_iter = sample_data_iter(data_train, 50, 6, 500, labels, format_balance_labels, encode, count)

    batch = get_next_batch(data_test, labels, size=None, kmer=6)
    xs, cs, ys = format_trim_dims(next(batch), dim=50000, encode=encode, count=count)

    for i in range(max_epoch):
        train_res = model.train_batch(data_iter, max_iterations=(i+1)*100, it_step_size=20)
        print('> Performance of the gene switch model')
        test_res = model.test(xs, cs, ys)

if __name__ == '__main__':
    from datetime import datetime
    START_TIME = datetime.now()

    test_labels_model()
    test_encode_model()
    test_switch_model()

    FINISH_TIME = datetime.now()
    print('Start  at', START_TIME)
    print('Finish at', FINISH_TIME)
    print("Time Cost", FINISH_TIME-START_TIME)
