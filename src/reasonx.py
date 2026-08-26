# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:06:44 2022

"""

# standard packages
import numpy as np
np.set_printoptions(legacy='1.25') # for output in raw format
import pandas as pd
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
import os
import sys
import io
import re
import subprocess

# pip install lark-parser
import lark
# pip install linear-tree
from lineartree import LinearTreeClassifier
# pip install janus-swi
import janus_swi as janus
# pip install hopsy
import hopsy

# local packages
import dautils

class ReasonX:
    # janus=False will go back to cmd line call of Python
    def __init__(self, pred_atts, target, df_code, verbose=2, janus=True):
        self.wdir = os.path.dirname(os.path.abspath(__file__))
        self.pl_post = os.path.join(self.wdir, "post.pl")
        self.pl_source = os.path.join(self.wdir, "newfile.pl")
        self.pred_atts = pred_atts
        self.target = target
        self.df_code = df_code
        self.verbose = verbose
        if "SWI_HOME_DIR" in os.environ:
            self.swipl = os.path.join(os.environ["SWI_HOME_DIR"], "bin", "swipl")
        else: # hope it is in the PATH
            self.swipl = "swipl"
        self.janus = janus
        self.janus_first = True
        # pos -> one-hot feature name
        self.feature_names = df_code.encoded_atts(pred_atts)
        # one-hot feature name -> pos
        self.feature_pos = {n:i for i, n in enumerate(self.feature_names)}
        # one-hot feature name -> original feature name
        self.feature_original = self.feature_names.copy()
        # pos -> one-hot value
        self.feature_value = [""]*len(self.feature_names)
        # pos -> is one-hot encoded feature
        self.feature_iscat = [False]*len(self.feature_names)
        for f in self.df_code.nominal:
            for a, v in self.df_code.encoded_atts([f], value=True):
                pos = self.feature_pos[a]
                self.feature_iscat[pos] = True
                self.feature_original[pos] = f
                self.feature_value[pos] = v
        # pos -> is ordinal feature
        self.feature_isord = [False]*len(self.feature_names)
        for f in self.df_code.ordinal: 
            if f != self.target:
                self.feature_isord[self.feature_pos[f]] = True
        self.transform = self.Parse(self)
        self.parse = lark.Lark(self.grammar_exp, parser='lalr', transformer=self.transform).parse
        self.reset()
       
    def reset(self, keep_model=False):
        self.constraints = []
        self.instances = dict()
        self.irules = dict()
        self.ans = []
        self.n_samples = 10000 # number of samples in MCMC
        self.minvals = []
        self.fidelitys = []
        # feature_bounds[pos] = (min, max)
        self.feature_bounds = dict()
        self.last_asserted = ""
        if not keep_model:
            self.models = []
            self.bb = []

    def verbosity(self, verbose):
        self.verbose = min(2, max(0, int(verbose)))

    # language grammar and parser
    grammar_exp = """
        _separated{x, sep}: x (sep x)*  // Define a sequence of 'x sep x sep x ...'
        
        ?start: seqc
            | exp
        ?seqc: cons
            | seqc "," cons       -> seq
        ?cons: exp "<" exp        -> lt
            | exp "<=" exp        -> le
            | exp "=" exp         -> eq
            | exp "!=" exp        -> neq
            | exp ">=" exp        -> ge
            | exp ">" exp         -> gt
            | cons "<=>" cons     -> iff
        ?exp: product
            | exp "+" product   -> add
            | exp "-" product   -> sub
        ?product: atom
            | product "*" atom  -> mul
            | product "/" atom  -> div
        ?atom: NUMBER_NAME        -> val
             | NUMBER           -> number
             | "-" atom         -> neg
             | NAME "." NAME    -> var
             | NAME             -> val
             | "l0norm" "(" NAME "," NAME ")" -> l0norm
             | "l1norm" "(" NAME "," NAME ")" -> l1norm
             | "l1norm" "(" NAME "," "[" _separated{NAME, ","} "]" ")" -> l1normd
             | "l1norml" "(" NAME ", [" NAME "," NAME "])" -> l1norml
             | "l1normll" "([" NAME "," NAME "], [" NAME "," NAME "])" -> l1normll
             | "linfnorm" "(" NAME "," NAME ")" -> linfnorm
             | "(" exp ")"
        NUMBER_NAME.2: /[0-9]+[A-Za-z_][A-Za-z0-9_]*/
        %import common.CNAME    -> NAME
        %import common.NUMBER
        %import common.WS_INLINE
        %ignore WS_INLINE
    """

    class Parse(lark.InlineTransformer):

        def __init__(self, m2clp):
            self.m2clp = m2clp

        def number(self, value):
            return ['number_const', str(float(value))]

        def val(self, value):
            for var in self.m2clp.df_code.nominal:
                if value in self.m2clp.df_code.encode[var]:
                    return ['val', value]
            for var in self.m2clp.df_code.ordinal:
                if value in self.m2clp.df_code.encode[var]:
                    return ['val', value]
            raise ValueError("unknown value "+value)

        def var(self, inst, var):
            if inst not in self.m2clp.instances:
                raise ValueError("unknown instance "+inst)
            if var not in self.m2clp.pred_atts:
                raise ValueError("unknown var "+var)
            return ['var', inst, var]
       
        def l0norm(self, linst, rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst)
            return ['l0norm', ['inst', linst], ['inst', rinst]]
        
        # l1 norm between two instances
        def l1norm(self, linst, rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst)
            return ['l1norm', ['inst', linst], ['inst', rinst]]
        
        # l1 norm between with diversity
        def l1normd(self, linst, *rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if len(rinst)==0:
                    raise ValueError("no instance provided")
            for inst in rinst:
                if inst not in self.m2clp.instances:
                    raise ValueError("unknown instance "+inst)
            return ['l1normd', ['inst', linst], 
                    [['inst', inst] for inst in rinst]]
        
        # diversity optimization
        def l1norml(self, linst, rinst1, rinst2):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst1 not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst1)
            if rinst2 not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst2)
            return ['l1norml', ['inst', linst], ['inst', rinst1], ['inst', rinst2]]
        
        # diversity optimization
        def l1normll(self, linst1, linst2, rinst1, rinst2):
            if linst1 not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst1)
            if linst2 not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst2)
            if rinst1 not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst1)
            if rinst2 not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst2)
            return ['l1normll', ['inst', linst1], ['inst', linst2], ['inst', rinst1], ['inst', rinst2]]
      
                        
        def linfnorm(self, linst, rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst)
            return ['linfnorm', ['inst', linst], ['inst', rinst]]
       
        def par(self, name, value):
            return ['par', name, value]

        def add(self, left, right):
            return ['+', left, right]

        def sub(self, left, right):
            return ['-', left, right]

        def mul(self, left, right):
            return ['*', left, right]

        def div(self, left, right):
            return ['/', left, right]

        def neg(self, left):
            return ['-', left]

        def lt(self, left, right):
            return ['<', left, right]

        def le(self, left, right):
            return ['=<', left, right]

        def eq(self, left, right):
            if left[0]=='var' and right[0]=='val':
                # inst.var = val
                var, val = left[2], right[1]
                if var in self.m2clp.df_code.nominal:
                    if val not in self.m2clp.df_code.encode[var]:
                        raise ValueError("value not in domain " + val)
                    left[2] += '_'+val
                    return ['=', left, ['number_const', '1']]
                if var in self.m2clp.df_code.ordinal:
                    if val not in self.m2clp.df_code.encode[var]:
                        raise ValueError("value not in domain " + val)
                    val_n = self.m2clp.df_code.encode[var][val]
                    return ['=', left, ['number_const', str(val_n)]]
                return ['=', left, ['number_const', val]]
            if left[0]=='var' and right[0]=='var':
                # inst1.var1 = inst2.var2
                inst1, inst2 = left[1], right[1]
                var1, var2 = left[2], right[2]
                var1n, var2n = var1 in self.m2clp.df_code.nominal, var2 in self.m2clp.df_code.nominal
                if var1n or var2n:
                    # at least one var1 or var2 nominal
                    if var1n and var2n:
                        # var1 and var2 nominal
                        d1 = set(self.m2clp.df_code.encode[var1].keys())
                        d2 = set(self.m2clp.df_code.encode[var2].keys())
                        if d1 != d2:
                            raise ValueError("equality between different domains "+var1+" "+var2)
                        res = []
                        for v in d1:
                            con = ["=", ['var', inst1, var1+'_'+v], ['var', inst2, var2+'_'+v]]
                            res = [',', con, res] if res != [] else con
                        return res
                    # one but not both var1 and var2 nominal
                    raise ValueError("equality between different types "+var1+" "+var2)
            return ['=', left, right]

        def neq(self, left, right):
            if left[0]=='var' and right[0]=='val':
                # inst.var != val
                var, val = left[2], right[1]
                left[2] += '_'+val
                if var not in self.m2clp.df_code.nominal:
                    raise ValueError("!= defined only for nominal variables")
                if val not in self.m2clp.df_code.encode[var]:
                    raise ValueError("value not in domain " + val)
                return ['=', left, ['number_const', '0']]
            raise ValueError("!= defined only form inst.var != value with var nominal")

        def ge(self, left, right):
            return ['>=', left, right]

        def gt(self, left, right):
            return ['>', left, right]

        def iff(self, left, right):
            # inst1.var1 = val1 <=> inst2.var2 = val2
            # arrives already transformed by eq() as
            # inst1.var1_val1 = 1 <=> inst2.var2_val2 = 1
            assert left[0]=='=' and right[0]=='='
            assert left[1][0]=='var' and right[1][0]=='var'
            assert left[2][0]=='number_const' and right[2][0]=='number_const'
            left1 = ['var', left[1][1], left[1][2]]
            right1 = ['var', right[1][1], right[1][2]]
            return ['=', left1, right1]

        def seq(self, left, right):
            return [',', left, right]

        def toCLP(self, tree):
            #print("tree", tree)
            op = tree[0]
            if op=='number_const':
                return tree[1]
            if op=='val':
                return tree[1]
            if op=='var':
                return 'var(i'+tree[1]+', v'+tree[2]+')'
            if op=='inst':
                return 'i'+tree[1]
            if op=='l1norm':
                return 'l1norm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            # diversity optimization
            if op=='l1normd':
                return 'l1normd(' + self.toCLP(tree[1]) + ', [' +\
                            ','.join([self.toCLP(el) for el in tree[2]]) + '])'
            # LAURA diversity optimization
            if op=='l1norml':
                return 'l1norml(' + self.toCLP(tree[1]) + ', [' + self.toCLP(tree[2]) + "," + self.toCLP(tree[3]) + '])'
            # LAURA diversity optimization
            if op=='l1normll':
                return 'l1normll([' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + '], [' + self.toCLP(tree[3]) + "," + self.toCLP(tree[4]) + '])'
            if op=='l0norm':
                return 'l0norm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            if op=='linfnorm':
                return 'linfnorm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            if op in {'+', '*', '/', '=<', '<', '=', ',', '>=', '>'}:
                return self.toCLP(tree[1]) + op + self.toCLP(tree[2])
            if op=='-' and len(tree)==3:
                return self.toCLP(tree[1]) + '-' + self.toCLP(tree[2])
            # add white space before negative number
            if op=='-' and len(tree)==2:
                return ' -' + self.toCLP(tree[1])
            raise ValueError("unknown operator"+op)

    # dynamic part of CLP code
    def toCLP(self, o=sys.stdout, project=None, norm=None, eps=0):
        # header
        if self.janus_first:
            # call or server mode first time: set predicate properties
            o.write(''':- use_module(library(lists)).
:- use_module(library(clpr)).
:- dynamic feature/2.
:- dynamic nfeatures/1.
:- dynamic cat_features/2.
:- dynamic ord_features_pos/2.
:- dynamic l1_weights/1.
:- dynamic linf_weights/1.
:- dynamic data_instance/6.
:- dynamic ninstances/1.
:- dynamic data_instance_proj/2.
:- dynamic path/5.
:- dynamic user_constraints/2.
:- dynamic min_norm/1.
:- dynamic relax_eps/1.''')
        # features
        o.write("\n% feature(pos, name) :- name for the i-th feature")
        for i, f in enumerate(self.feature_names):
            o.write('\nfeature({}, v{}).'.format(i, f))
        nf = len(self.feature_names)
        o.write('\nnfeatures({}).'.format(nf))

        # categorical fatures
        o.write("\n% cat_features(name, positions) :- positions of categorical features")
        for f in self.df_code.nominal:
            pos = [self.feature_pos[v] for v in self.df_code.encoded_atts([f])]
            o.write("\ncat_features(v{}, {}).".format(f, pos))
        if len(self.df_code.nominal)==0:
            o.write("\ncat_features(v_dummy, []).")

        # ordinal fatures
        o.write("\n% ord_features_pos(positions) :- positions of ordinal features")
        pos = [self.feature_pos[f] for f in self.df_code.ordinal if f != self.target]
        bounds = [(min(self.df_code.decode[f]), max(self.df_code.decode[f])) 
                      for f in self.df_code.ordinal if f != self.target]
        o.write("\nord_features_pos({}, {}).".format(pos, bounds))

        # l1 norm weights
        o.write("\n% norm_weights(weights) :- features weights")
        weights = [0.5 if self.feature_iscat[i] # nominal
                   else 
                   (1/(max(self.df_code.decode[f]) - min(self.df_code.decode[f])) if self.feature_isord[i] # ordinal
                   else 
                   1/(self.df_code.encode[f][1] - self.df_code.encode[f][0]) ) # continuous
                       for i, f in enumerate(self.feature_names)]
        o.write(f"\nl1_weights({weights}).")
        # linf norm weights
        for i, f in enumerate(self.feature_names):
            if self.feature_iscat[i]:
                weights[i] = 1
        o.write("\nlinf_weights({}).".format(weights))

        # instances
        fcon = []
        o.write("\n% data_instance(id, name, class, minconf, mincov, model) :- instance id, name, class label, minumal confidence, minimal coverage, model")
        for name, (n, label, minconf, mincov, model, con) in self.instances.items():
            o.write("\ndata_instance({}, i{}, {}, {}, {}, {}).".format(n, name, label, minconf, mincov, model))
            fcon.extend(con)
            for att in self.feature_bounds:
                minv, maxv = self.feature_bounds[att]
                rangecon = '{} >= {}.{}, {}.{} >= {}'.format(maxv, name, att, name, att, minv)
                fcon.extend(self.constraint(rangecon, only_ret=True))
        ni = len(self.instances.items())
        o.write('\nninstances({}).'.format(ni))
        
        # projections
        project = self.instances.keys() if project is None else project # whole instance
        # TBD: check v in project is either an existing instance or a feature (I.f)
        project = [ [v+'.'+f for f in self.pred_atts] if v in self.instances else v for v in project] # attributes in each instance
        project = set(dautils.flatten(project)) 
        o.write("\n% data_instance_proj(id, pos) :- positions to project instance id")
        for name, (n, _, _, _, _, _) in self.instances.items():
            pos = list()
            for f in self.pred_atts:
                if (name+'.'+f) not in project:
                    continue
                if f in self.df_code.nominal:
                    pos.extend([self.feature_pos[v] for v in self.df_code.encoded_atts([f])])
                else:
                    pos.append(self.feature_pos[f])
            o.write("\ndata_instance_proj({}, {}).".format(n, pos))
     
        # models
        for m in self.models:
            o.write(m)

        # instance + user constraints
        o.write("\n% user_constraints(+Vars, -Cs)")
        fcon.extend(self.constraints)
        if len(fcon)>0:
            Cs = ", ".join(c for c in fcon)
            o.write("""\nuser_constraints(Vars, Cs) :-
            Constraints_list = [{}], 
            exp2cons(Constraints_list, Vars, Cs).
            """.format(Cs))
        else:
            o.write("\nuser_constraints(_, []).")
 
        # norm
        o.write("\nmin_norm({}).".format(0 if norm is None else self.transform.toCLP(self.parse(norm))))
        
        # eps
        o.write("\nrelax_eps({}).".format(eps))
        
        if not self.janus:
            # include post.pl
            o.write("\n:- ['post.pl'].")

    # get answer constraints of last query
    def answers(self, distinct=False):
        return list(set(self.ans)) if distinct else self.ans
    
    # get minimized value of last  query
    def minvalues(self):
        return self.minvals
    
    # get fidelity estimates of last query
    def fidelities(self):
        return self.fidelitys
    
    # get factual rules (premize, confidence, coverage) of last query
    def rules(self, instance, distinct=False):
        return list(set(self.irules[instance])) if distinct else self.irules[instance]
    
    # main function
    def solveopt(self, minimize=None, project=None, eps=sys.float_info.epsilon, n_samples=None):
        if self.recompile or project is not None or minimize is not None:
            o = open(self.pl_source, "w") if not self.janus else io.StringIO()
            self.toCLP(project=project, norm=minimize, eps=eps, o=o)
            if self.janus:
                source = o.getvalue()
            else:
                o.close()
            self.recompile = project is not None
        goal = 'q_nominimize' if minimize is None else 'q_minimize'
        if self.janus:
            # run Prolog from server janus
            janus.consult("newfile", source) # this reconsult if not janus_first
            if self.janus_first:
                self.janus_first = False
                #with open(self.pl_post, "r", encoding="utf-8") as f:
                #    source = f.read()
                #janus.consult("post", source)
                janus.consult(self.pl_post)
            #print(source)
            #print('CURRENT PROLOG')
            #result = janus.query_once("with_output_to(string(S), listing)")
            #print(result["S"])
            janus_res = janus.query_once(f"with_output_to(string(Output), findall(_, {goal}, _))")
            res = janus_res["Output"]
        else:
            # run Prolog from cmdline
            cmd = [self.swipl, "-q", "-t", "halt", "-g", goal, "-l", self.pl_source]
            cmdres = subprocess.run(cmd, capture_output=True, text=True)
            res = cmdres.stdout
        res = res.replace('=<', '<=', )
        res = res.replace('=>', '>=', )
        res = res.split('\n')
        #print(res)
        # parse results
        ninstances = len(self.instances)
        pos2inst = {val[0]:name for name, val in self.instances.items()}
        self.ans = []
        self.minvals = []
        self.fidelitys = []
        self.irules = {name:[] for name in self.instances.keys()}
        if minimize is None:
            # q_minimize
            step = 7
        else:
            # q1_3
            step = 8
        # compute number of results
        number_of_results = (len(res) - 1) / step
        for i in range(0, len(res)-1, step):
            inst = res[i]
            inst = eval(re.sub(r"(\_\d+)", r"'\g<0>'", inst))
            con = r","+res[i+1][1:-1]+r","
            paths = res[i+2]      
            confs = eval(res[i+3])     
            coverages = eval(res[i+step-1])
            minvalue = None if minimize is None else float(res[i+4])
            used = [ [] for i in range(ninstances)]
            pvar2ivar = { vnum:(ipos, fpos)\
                            for ipos, ilist in enumerate(inst)\
                                for fpos, vnum in enumerate(ilist) }   
            for k in sorted(pvar2ivar.keys(), key=len, reverse=True):
                ipos, fpos = pvar2ivar[k]
                r = pos2inst[ipos]+'.'+self.feature_names[fpos]
                con2 = con.replace(k, r)
                if con != con2:
                    con = con2
                    used[ipos].append(fpos)
                    paths = paths.replace(k, r)
            # first pass
            for ipos, a in enumerate(used):
                # detect =1.0
                for fpos in a:
                    if self.feature_iscat[fpos]:
                        r = r","+pos2inst[ipos]+'.'+self.feature_names[fpos]
                        if con.find(r+"=1.0,")>=0:
                            # remove all other "=0.0"
                            f = self.feature_original[fpos]
                            for v in self.df_code.encoded_atts([f]):
                                if v != self.feature_names[fpos]:
                                    r = r","+pos2inst[ipos]+'.'+v
                                    con = re.sub(r+r"=0.0,", ",", con)
            # second pass
            for ipos, a in enumerate(used):
                for fpos in a:
                    if self.feature_iscat[fpos]:
                        r = pos2inst[ipos]+'.'+self.feature_names[fpos]
                        val = self.feature_value[fpos]
                        s = pos2inst[ipos]+'.'+self.feature_original[fpos]
                        con = re.sub(r+r"=1.0", s+"="+val, con) 
                        con = re.sub(r+r"=0.0", s+"!="+val, con) 
            # SR: ordinal features are not decoded back! Better readability
            # output constraints
            con = con[1:-1]
            self.ans.append(con)
            # NEW calculate CFruleConf
            fidelity = CFruleConf = CFruleClass = None
            is_project = project is not None and len(project)==1 and project[0] in self.instances # projecting over one instance
            #print('is_project', is_project)
            if is_project:
                pjvar = project[0] # projected instance
                #print(pjvar)
                modelID = self.instances[pjvar][3]
                modelClf = self.bb[modelID]
                CFruleClass = self.instances[pjvar][1]
                is_bb = modelClf is not None
                #print('is_bb', is_bb)
                if is_bb:
                    # remove constraints on constant variables
                    #print(con)
                    try:
                        # simple parsing of Var = const or Var <= const or Var >= const - more refined parsing still possibile but computationally expensive
                        text, var_to_pos, var_to_const = self.hopsy_parse(con)
                        # check if var_to_pos contains only continuous and bounded features
                        # only continuous ensured by inf operator
                        boundedness_condition = True
                        for f in self.pred_atts:
                            if f in self.df_code.nominal or f in self.df_code.ordinal:
                                if f not in var_to_const:
                                    boundedness_condition = False
                                    #print('boundedness_condition', boundedness_condition, f)
                                    break
                            else:
                                if f not in self.feature_bounds: # weaker than c implies f is bounded, but faster to check
                                    boundedness_condition = False
                                    #print('boundedness_condition', boundedness_condition, f)
                                    break
                    except ValueError:
                        boundedness_condition = False
 
                    #print('boundedness_condition', boundedness_condition)
                    if boundedness_condition:
                        # adds missing continuous vars
                        for var in self.pred_atts:
                            if var not in var_to_const and var not in var_to_pos:
                                pos = len(var_to_pos)
                                var_to_pos[var] = pos
                        if len(var_to_pos)==0: # all features are constant
                            polySamples = pd.DataFrame([var_to_const])
                        else:
                            polySamples = self.hopsy_random(text, var_to_pos, n_variables=len(var_to_pos), 
                                                       n_samples=self.n_samples if n_samples is None else n_samples)
                            # re-adds constant features
                            for var, value in var_to_const.items():
                                polySamples[var] = value
                        polySamples = polySamples[self.pred_atts]
                        ordinal = [f for f in self.df_code.ordinal if f != self.target]
                        polySamples[ordinal] = polySamples[ordinal].astype(float).astype(int)
                        polySamples = self.df_code.transform(polySamples, unprocess=self.df_code.ordinal)
                        modelPred = modelClf.predict(polySamples)
                        CFruleConf = np.mean(modelPred==CFruleClass) # class predicted as expected?
                        # TBD: coverage for contrastive rules can also be estimated
                        # generate random instance in the factual rule of CE, and check how many of them satisfy the contrastive constraint
                fidelity = CFruleConf if is_bb else 1
            self.fidelitys.append(fidelity)
            # END calculate CFruleConf
            if minvalue is not None:
                self.minvals.append(minvalue)
            if self.verbose >= 1:
                #print("--\nAnswer constraint for %s: %s" % (n, con_sub) )
                # dimentionality (new), for the CE
                print('---\nAnswer constraint:', con)
                if is_project:
                    #print(CFruleConf)
                    #print(CFruleClass)
                    l = self.df_code.decode[self.target][CFruleClass]
                    #print(l)
                    pstr = "NA" if fidelity is None else f"{fidelity:.4f}"
                    print(f"Contrastive rule satisfied by {pjvar}: IF Answer constraint THEN {l} [{pstr}]")
                if minimize is not None:
                    print('Min value:{:.4f}'.format(minvalue))
            paths = paths.split("],[")
            paths[0] = paths[0][2:] # remove trailing []
            paths[-1] = paths[-1][:-2] 
            for i, (p, c, cov) in enumerate(zip(paths, confs, coverages)):
                name = pos2inst[i]
                #print(name, p, c)
                label = self.instances[name][1]
                l = self.df_code.decode[self.target][label]
                if self.verbose >= 2 and (project is None or name in project):
                    print('Rule satisfied by {}: IF {} THEN {} [{:.4f}, {:.4f}]'.format(name, p, l, c, cov))
                self.irules[name].append((p, c, cov))
        if self.verbose>= 1 and len(res)<2:
            print('No answer.')

    # retract an instance
    def iretract(self, name):
        self.recompile = True
        if name in self.instances:
            del self.instances[name]
            # reorder all other instances
            for i, (k, v) in enumerate(self.instances.items()):
                _, a, b, c, d, e = v
                self.instances[k] = (i, a, b, c, d, e)
            # retract all constraints involving name
            prologVar = 'var(i'+name+','
            self.constraints = [cs for cs in self.constraints if prologVar not in cs]
        else:
            raise Exception("instance "+name+" does not exist")
        
    # declare an instance  
    def instance(self, name, label, features=None, minconf=0, mincov=0, overwrite=True, model=None):
        self.recompile = True
        if name in self.instances:
            if not overwrite:
                raise Exception("instance "+name+" exists already")
            n, _, _, _, _, _  = self.instances[name]
        else:
            n = len(self.instances)
        model = len(self.models)-1 if model is None else model # else: assume model is the ID of an already asserted model
        if features is None:
            self.instances[name] = (n, label, minconf, mincov, model, [])
            return
        if isinstance(features, list):
            con = ", ".join([name+"." + f + " = " + str(features[i]) for i, f in enumerate(self.pred_atts)])
        elif isinstance(features, dict):
            con = ", ".join([name+"." + f + " = " + str(v) for f, v in features.items()])
        else:
            if len(features) != 1:
                raise Exception("only one row in the data frame, please!")
            features = self.df_code.inverse_transform(features).reset_index()
            con = ", ".join([name+"." + f + " = " + str(features.loc[0, f]) for f in self.pred_atts])
        if self.verbose >= 3:
            print('Generated constraint:'+con)
        self.instances[name] = 'dummy' # need to have the instance to call self.constraint()
        self.instances[name] = (n, label, minconf, mincov, model, self.constraint(con, only_ret=True))

    # declare a DT and possibly its black-box
    def model(self, clf, bb=None, returnID=False):
        self.recompile = True
        nf = len(self.feature_names)
        nm = len(self.models)
        res = "\n% path(ModelID, Vars, Constraint, Pred, Conf, Cov) :- Constraint in a path of a decision tree over Vars with prediction Pred, confidence Conf, and coverage Cov"
        if isinstance(clf, DecisionTreeClassifier):
            tree_ = clf.tree_
            classes_ = clf.classes_
            feature_pos = {f:i for i, f in enumerate(self.feature_names)}
            feature_name = [
                feature_pos[self.feature_names[i]] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            def recurse(node, body="", varset=set()):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    var = feature_name[node]
                    name = 'X' + str(var)
                    threshold = tree_.threshold[node]        
                    if body != '':
                        body = body + ','
                    body_left = body + "{} =< {}".format(name, threshold)
                    varset = varset | set([var])
                    res_left = recurse(tree_.children_left[node], body_left, varset)
                    body_right = body + "{} > {}".format(name, threshold)
                    res_right = recurse(tree_.children_right[node], body_right, varset)
                    return res_left + "\n" + res_right
                else:
                    freqs = tree_.value[node][0]
                    cov = clf.tree_.weighted_n_node_samples [node] / clf.tree_.weighted_n_node_samples [0]
                    pred, maxfreq = dautils.argmax(freqs)
                    maxfreq /= sum(freqs)
                    allf = ','.join( ('X'+str(i) if i in varset else '_') for i in range(nf) )
                    return "path({}, [{}], [{}], {}, {}, {}).".format(nm, allf, body, classes_[pred], maxfreq, cov)
            modelID = len(self.models)
            self.models.append(res + "\n" + recurse(0))
            #print(self.models[modelID])
            self.bb.append(bb)
        elif isinstance(clf, LinearTreeClassifier):
            tree_ = clf.summary()
            if len(clf.classes_) != 2:
                raise ValueError("only binary model trees are admissible so far")
            def recurse(n, body="", varset=set()):
                node = tree_[n]
                if 'col' in node:
                    var = node['col']
                    name = 'X' + str(var)
                    threshold = node['th']
                    if body != '':
                        body = body + ','
                    body_left = body + "{} =< {}".format(name, threshold)
                    varset = varset | set([var])
                    res_left = recurse(node['children'][0], body_left, varset)
                    body_right = body + "{} > {}".format(name, threshold)
                    res_right = recurse(node['children'][1], body_right, varset)
                    return res_left + "\n" + res_right
                else:
                    coef = node['models'].coef_[0]        
                    threshold = float(node['models'].intercept_[0])
                    varset = varset | set([i for i, v in enumerate(coef) if v != 0])
                    allf = ','.join( ('X'+str(i) if i in varset else '_') for i in range(nf) )
                    maxfreq = 1 # TBD confidence to be calculated
                    # left
                    name = '+'.join(str(v)+'*X'+str(i) for i, v in enumerate(coef) if v != 0)
                    if body != '':
                        body = body + ','
                    body_left = body + "{} =< {}".format(name, threshold)
                    body_right = body + "{} > {}".format(name, threshold)
                    body_left = body_left.replace("+-", "+ -")
                    body_right = body_right.replace("+-", "+ -")
                    cov = 0 # TBD - add rule coverage calculation
                    left = "path({},[{}],[{}],{},{},{}).".format(nm, allf, body_left, node['classes'][0], maxfreq, cov)
                    right = "path({},[{}],[{}],{},{},{}).".format(nm, allf, body_right, node['classes'][1], maxfreq, cov)
                    return left + "\n" + right
            modelID = len(self.models)
            self.models.append(res + "\n" + recurse(0))
            self.bb.append(bb)
        else:
            raise ValueError("unknown model " + str(clf))
        if returnID:
            return modelID

    # assert a constraint
    def constraint(self, con, only_ret=False):
        self.recompile = True
        cons = [self.transform.toCLP(self.parse(c)) for c in re.split(r',\s*(?![^()]*\))', con)]
        if only_ret:
            return cons
        self.last_asserted = con
        self.constraints.extend(cons)
        
    # assert bounds on a feature wrt to all instances
    def bounds(self, att, minv, maxv):
        pos = self.feature_pos[att]
        if self.feature_iscat[pos] or self.feature_isord[pos]:
            raise Exception('bounds for continuous attributes only')
        self.feature_bounds[att] = (minv, maxv)
                    
    # retract a constraint
    def retract(self, con="", last=False):
        nr = 0
        if last:
            con = con+","+self.last_asserted if con!="" else self.last_asserted
        for c in re.split(r',\s*(?![^()]*\))', con):
            ret = self.transform.toCLP(self.parse(c))
            # retract from self.constraints
            nc = len(self.constraints)
            self.constraints = [cs for cs in self.constraints if cs != ret]
            nc -= len(self.constraints)
            nr += nc  # update number retracted
            for name in self.instances:
                (n, label, minconf, mincov, model, con) = self.instances[name]
                nc = len(con)
                con = [cs for cs in con if cs != ret]
                nc -= len(con)
                if nc > 0: # retract from instances                    
                    self.instances[name] = (n, label, minconf, mincov, model, con) 
                    nr += nc # update number retracted
        self.recompile = nr > 0
        if self.verbose >= 2:
            print(nr, 'constraint(s) retracted')

    # utility function for parsing constraints to the input to hopsy
    def hopsy_parse(self, con):
        text = ""
        var_to_pos = dict()
        var_to_const = dict()
        for constraint in con.split(","):
            constraint = constraint.strip()
            #print(constraint)
            match = re.fullmatch(
                r"(\w+)\.(\w+)\s*(<=|>=|=|<|>)\s*(.*)",
                constraint,
            )
            if match is None:
                raise ValueError(f"Error parsing '{constraint}' in reasonx.hopsy_parse!")
            inst = match.group(1)
            var = match.group(2)
            operator = match.group(3)
            value = match.group(4)
            if operator=="=":
                var_to_const[var] = value if var in self.df_code.nominal or var in self.df_code.ordinal else float(value)
                continue
            if var in var_to_pos:
                pos = var_to_pos[var]
            else:
                pos = len(var_to_pos)
                var_to_pos[var] = pos
            constraint_mapped = f"{inst}.{pos}{operator}{value}"
            text += constraint_mapped if text=="" else ","+constraint_mapped
        return text, var_to_pos, var_to_const

    # utility function to generate random instances using hopsy MCMC
    def hopsy_random(self, text, var_to_pos, n_variables, n_samples):
        polyA, polyb = self.hopsy_matrix(text, n_variables)
        #print(polyA, polyb)
        problem = hopsy.Problem(polyA, polyb)
        chain = hopsy.MarkovChain(problem)
        rng = hopsy.RandomNumberGenerator(seed=42)
        acceptance_rate, polySamples = hopsy.sample(
            chain,
            rng,
            n_samples=self.n_samples if n_samples is None else n_samples
        )
        polySamples = np.asarray(polySamples).squeeze()
        vars_by_pos = sorted(var_to_pos, key=var_to_pos.get)
        polySamples = pd.DataFrame(polySamples, columns=vars_by_pos)
        return polySamples

    # utility function to transform textual constraint to matricial form required by hopsy
    def hopsy_matrix(self, text, n_variables, epsilon=1e-9):
        rows = []
        bounds = []
    
        for constraint in text.split(","):
            constraint = constraint.strip()
    
            match = re.fullmatch(
                r"\w+\.(\d+)\s*(<=|>=|=|<|>)\s*"
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                constraint,
            )
    
            if not match:
                raise ValueError(f"Invalid constraint: {constraint}")
    
            position = int(match.group(1))
            operator = match.group(2)
            value = float(match.group(3))
    
            if not 0 <= position < n_variables:
                raise ValueError(
                    f"CF.{position} is outside the valid range "
                    f"0 to {n_variables - 1}."
                )
    
            if operator == "=":
                # x[position] <= value
                row_upper = np.zeros(n_variables)
                row_upper[position] = 1
                rows.append(row_upper)
                bounds.append(value)
    
                # -x[position] <= -value
                row_lower = np.zeros(n_variables)
                row_lower[position] = -1
                rows.append(row_lower)
                bounds.append(-value)
    
                continue
    
            row = np.zeros(n_variables)
    
            if operator == "<=":
                row[position] = 1
                bound = value
    
            elif operator == "<":
                row[position] = 1
                bound = value - epsilon
    
            elif operator == ">=":
                row[position] = -1
                bound = -value
    
            elif operator == ">":
                row[position] = -1
                bound = -(value + epsilon)
    
            rows.append(row)
            bounds.append(bound)
    
        return np.asarray(rows), np.asarray(bounds)

# global utility function for generating random instances
def hopsy_random(con, atts, n_samples=100):
    df_code = dautils.Encode(nominal=[], ordinal=[], decode=dict())
    r = ReasonX(atts, None, df_code)
    hcon, var_to_pos, _ = r.hopsy_parse(con)
    return r.hopsy_random(hcon, var_to_pos, len(atts), n_samples=n_samples)
