# -*- coding: utf-8 -*-
"""
dautils Data analysis utility functions and datasets

@author: Salvatore Ruggieri
"""

import pandas as pd
import sys
import urllib
import gzip
import codecs
import matplotlib.pyplot as plt

def plt_setting():
    """ preferred settings for plots """
    plt.style.use('seaborn-whitegrid') # https://matplotlib.org/devdocs/gallery/style_sheets/style_sheets_reference.html
    plt.rc('font', size=11)
    plt.rc('legend', fontsize=11)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', linewidth=2)
    plt.rc('axes', edgecolor='k')
    plt.rc('xtick.major', width=2)
    plt.rc('xtick.major', size=6)
    plt.rc('ytick.major', width=2)
    plt.rc('ytick.major', size=6)
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

def flatten(l):
    return [v for subl in l for v in flatten(subl)] if type(l) is list else [l]


def argmax(values, f=lambda x: x):
    """ Argmax function 
    
    Parameters:
    values (iterable): collection of values
    f (value->number): functional 
    
    Returns:
    p: index of max value
    mv: max value, i.e., max{f(v) | v in values}
    """
    mv = None
    p = None
    for i, v in enumerate(values):
        fv = f(v)
        if mv is None or (fv is not None and mv < fv):
            mv, p = fv, i
    return p, mv

def getReader(filename, encoding='utf8'):
    """ Return a reader from a file, url, or gzipped file/url """
    if filename=='':
        return sys.stdin
    try:
        if filename.endswith('.gz'):
            file = gzip.open(filename)
        else:
            file = open(filename, encoding=encoding)
    except:
        file = urllib.request.urlopen(filename)
        if filename.endswith('.gz'):
            file = gzip.GzipFile(fileobj=file)
        reader = codecs.getreader(encoding)
        file = reader(file)
    return file  

def getContents(filename, n=None):
    """ get  lines from a text file or whole contents """
    with getReader(filename) as f:
        if n is None:
            return f.read()
        res = ''
        i = 0
        for line in f:
            if i>=n:
                break
            i += 1
            res += line

def getCSVattributes(freader, sep=','):
    """ Return the list of attributes in the header of a CSV or ARFF file reader """
    result = []
    line = freader.readline()
    while line=='':
        line = freader.readline()
    if line.startswith('@relation'):
        # read header from ARFF
        for line in freader:
            if line.startswith('@data'):
                break
            elif line.startswith('@attribute'):
                    result.append(line.split(' ')[1])
    else:
        # read header from CSV
        result = line.strip().split(sep)
    return result

def get_att(itemDesc, sep='=', ifNotItem=''):
    """ Extract attribute name from attribute=value string """
    pos = itemDesc.find(sep)
    return itemDesc[:pos] if pos>=0 else ifNotItem
 
def get_val(itemDesc, sep='=', ifNotItem=''):
    """ Extract attribute name from attribute=value string """
    pos = itemDesc.find(sep)
    return itemDesc[pos+1:] if pos>=0 else ifNotItem
 
class Encode:
    """ Encoding of discrete attributes in dataframes """
    def __init__(self, nominal=None, ordinal=[], decode=dict(), onehot=False, prefix_sep="="):
        """ Constructor 
        
        Parameters:
        nominal (iterable): nominal attributes to encode or None for all attributes
        ordinal (iterable): ordinal attributes to encode
        decode (dictionary): pre-set encodings to extend  
        onehot: use one-hot-encoding for nominal (if False, map values to integers)
        """
        self.nominal = set(nominal)
        self.ordinal = set(ordinal)
        self.decode = decode
        self.encode = { c:{i:v for v, i in self.decode[c].items()} for c in self.decode}
        self.onehot = onehot
        self.prefix_sep = prefix_sep

    def fit_transform(self, df):
        """ Encode a dataframe creating new encodings for not yet encoded columns
        
        Parameters:
        df (pd.DataFrame): dataframe
        
        Returns:
        pd.DataFrame: encoded dataframe
        """
        if self.nominal is None:
            self.nominal = set(df.columns)
        cols = set(df.columns)
        atts = self.nominal | self.ordinal
        atts &= cols # only atts in df
        encoded = self.encode.keys()
        atts -= encoded # not already encoded
        cols -= encoded # not already encoded
        for col in cols:
            if col in atts:
                uniq = sorted([v for v in df[col].unique() if pd.notna(v)])
                self.encode[col] = { v:i for i, v in enumerate(uniq) }
            else: # continuous
                self.encode[col] = (df[col].min(), df[col].max())
        self.decode = { c:{i:v for v, i in self.encode[c].items()} for c in self.encode if not isinstance(self.encode[c], tuple)}
        return self.transform(df)
    
    def transform(self, df):
        """ Encode a dataframe raising error if new encodings are needed
        
        Parameters:
        df (pd.DataFrame): dataframe
        
        Returns:
        pd.DataFrame: encoded dataframe
        """
        if self.nominal is None:
            raise "no nominal attributes provided"
        cols = set(df.columns)
        atts = self.ordinal if self.onehot else (self.nominal | self.ordinal)
        atts &= cols # only atts in df
        res = pd.DataFrame()
        for col in atts:
            if col not in self.encode:
                raise "no encoding for attribute "+col
            res[col] = df[col].map(self.encode[col])
            res[col] = res[col].astype('category')
        for col in self.ordinal:
            res[col] = res[col].astype(int)
        for col in cols-atts-self.nominal:
            res[col] = df[col]
        if self.onehot:
            res = res[[c for c in df.columns if c not in self.nominal]]
            for col in cols & self.nominal:
                dummies = pd.get_dummies(df[col], prefix=col, prefix_sep=self.prefix_sep)
                res = pd.concat([res, dummies], axis=1)
            res = res[self.encoded_atts(df.columns)]
        else:
            res = res[df.columns]
        return res
    
    def encoded_atts(self, cols, value=False):
        colnames = lambda c: [c+self.prefix_sep+v for v in self.decode[c].values()] if c in self.nominal else [c]
        colvalues = lambda c: [v for v in self.decode[c].values()] if c in self.nominal else [c]
        if value:
            return [ (c1, v1) for c in cols for c1, v1 in zip(colnames(c), colvalues(c))]
        return [ c1 for c in cols for c1 in colnames(c)]

    def inverse_transform(self, df):
        """ Decode an encoded dataframe
        
        Parameters:
        df (pd.DataFrame): dataframe
        
        Returns:
        pd.DataFrame: decoded dataframe
        """
        cols = set(df.columns)
        res = pd.DataFrame()
        for col in cols & self.ordinal:
            res[col] = df[col].map(self.decode[col])
            res[col] = res[col].astype('category')
        if self.onehot:
            col_lists = set()
            for col in self.nominal:
                col_list = [col+self.prefix_sep+v for v in self.decode[col].values()]
                col_set = set(col_list)
                if col_set.issubset(df.columns):
                    res[col] = df[col_list].idxmax(axis=1).apply(lambda c:get_val(c, self.prefix_sep))
                    res[col] = res[col].astype('category')
                    col_lists |= col_set
            remainder = cols-col_lists-self.ordinal 
        else:
            for col in cols & self.nominal:
                res[col] = df[col].map(self.decode[col])
                res[col] = res[col].astype('category')
            remainder = cols-self.nominal-self.ordinal 
        for col in remainder:
            res[col] = df[col]
        columns = [get_att(c, self.prefix_sep, c) for c in df.columns]
        columns = list(dict.fromkeys(columns))
        res = res[columns]
        return res
