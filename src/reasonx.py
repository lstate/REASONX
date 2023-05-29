# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:06:44 2022

"""

# standard packages
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
import sys
import os
import re
#import numpy as np

# pip install lark-parser
import lark
# pip install linear-tree
from lineartree import LinearTreeClassifier

# local packages
import dautils

class ReasonX:
    def __init__(self, pred_atts, target, df_code):
        self.pred_atts = pred_atts
        self.target = target
        self.df_code = df_code
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
        # feature_bounds[pos] = (min, max)
        self.feature_bounds = dict()
        self.transform = self.Parse(self)
        self.parse = lark.Lark(self.grammar_exp, parser='lalr', transformer=self.transform).parse
        self.reset()

    def reset(self, keep_model=False):
        self.constraints = []
        self.instances = dict()
        self.irules = dict()
        self.ans = []
        self.feature_bounds = dict()
        if not keep_model:
            self.models = []

    grammar_exp = """
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
        ?atom: NUMBER           -> number
             | "-" atom         -> neg
             | NAME "." NAME    -> var
             | NAME             -> val
             | "l1norm" "(" NAME "," NAME ")" -> l1norm
             | "l0norm" "(" NAME "," NAME ")" -> l0norm
             | "linfnorm" "(" NAME "," NAME ")" -> linfnorm
             | "(" exp ")"
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
            return ['val', value]

        def var(self, inst, var):
            if inst not in self.m2clp.instances:
                raise ValueError("unknown instance "+inst)
            if var not in self.m2clp.pred_atts:
                raise ValueError("unknown var "+var)
            return ['var', inst, var]

        def norm(self, *pars):
            if len(pars) != 3 or pars[0][1] != 'Alpha':
                raise Exception("one par Alpha expected"+pars)
            elif pars[1][1] != 'Beta':
                raise Exception("one par Beta expected"+pars)
            elif pars[2][1] != 'Gamma':
                raise Exception("one par Gamma expected"+pars)
            alpha = str(float(pars[0][2]))
            beta = str(float(pars[1][2]))
            gamma = str(float(pars[2][2]))
            return ['norm', alpha, beta, gamma] 
       
        def l1norm(self, linst, rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst)
            return ['l1norm', ['inst', linst], ['inst', rinst]]
        
        def l0norm(self, linst, rinst):
            if linst not in self.m2clp.instances:
                raise ValueError("unknown instance "+linst)
            if rinst not in self.m2clp.instances:
                raise ValueError("unknown instance "+rinst)
            return ['l0norm', ['inst', linst], ['inst', rinst]]
        
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
            if op=='norm':
                return tree[1]
            if op=='var':
                return 'var(i'+tree[1]+', v'+tree[2]+')'
            if op=='inst':
                return 'i'+tree[1]
            if op=='l1norm':
                return 'l1norm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            if op=='l0norm':
                return 'l0norm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            if op=='linfnorm':
                return 'linfnorm(' + self.toCLP(tree[1]) + ',' + self.toCLP(tree[2]) + ')'
            if op in {'+', '*', '/', '=<', '<', '=', ',', '>=', '>'}:
                return self.toCLP(tree[1]) + op + self.toCLP(tree[2])
            if op=='-' and len(tree)==3:
                return self.toCLP(tree[1]) + '-' + self.toCLP(tree[2])
            if op=='-' and len(tree)==2:
                return -self.toCLP(tree[1])
            raise ValueError("unknown operator"+op)

    def toCLP(self, o=sys.stdout, project=None, norm=None):

        o = open("newfile.pl", "w")

        # header
        o.write(":- use_module(library(clpr)).")
        o.write("\n:- use_module(library(lists)).")

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

        # norm weghts
        o.write("\n% norm_weights(weights) :- features weights")
        weights = [0.5 if self.feature_iscat[i] # nominal
                   else 
                   (1/(max(self.df_code.decode[f]) - min(self.df_code.decode[f])) if self.feature_isord[i] # ordinal
                   else 
                   1/(self.df_code.encode[f][1] - self.df_code.encode[f][0]) ) # continuous
                       for i, f in enumerate(self.feature_names)]
        o.write("\nnorm_weights({}).".format(weights))

        # instances
        fcon = []
        o.write("\n% data_instance(id, name, class, minconf, model) :- instance id, name, class label, minumal confidence, model")
        for name, (n, label, minconf, model, con) in self.instances.items():
            o.write("\ndata_instance({}, i{}, {}, {}, {}).".format(n, name, label, minconf, model))
            fcon.extend(con)
            for att in self.feature_bounds:
                minv, maxv = self.feature_bounds[att]
                rangecon = '{} >= {}.{}, {}.{} >= {}'.format(maxv, name, att, name, att, minv)
                fcon.extend(self.constraint(rangecon, only_ret=True))
        ni = len(self.instances.items())
        o.write('\nninstances({}).'.format(ni))
        
        # projections
        project = self.instances.keys() if project is None else project # whole instance
        project = [ [v+'.'+f for f in self.pred_atts] if v in self.instances else v for v in project] # attributes in each instance
        project = set(dautils.flatten(project)) 
        o.write("\n% data_instance_proj(id, pos) :- positions to project instance id")
        for name, (n, _, _, _, _) in self.instances.items():
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
        
        o.write("\n:- ['post.pl'].")
        o.close()

    def answers(self, distinct=False):
        return list(set(self.ans)) if distinct else self.ans
    
    def rules(self, inst, distinct=False):
        return list(set(self.irules[inst])) if distinct else self.irules[inst]
    
    # new function that encapsulates the optimization
    # uses specific predicates from the pl script
    def solveopt(self, minimize=None, project=None, verbose=1):
        if self.recompile or project is not None or minimize is not None:
            self.toCLP(project=project, norm=minimize)
            self.recompile = project is not None
        # run Prolog from cmdline
        goal = 'q0' if minimize is None else 'q1'
        cmd = "swipl -q -t halt -g " + goal + " -l newfile.pl"
        res = os.popen(cmd).read()
        res = res.replace('=<', '<=', )
        res = res.replace('=>', '>=', )
        res = res.split('\n')
        # parse results
        ninstances = len(self.instances)
        pos2inst = {val[0]:name for name, val in self.instances.items()}
        self.ans = []
        self.irules = {name:[] for name in self.instances.keys()}
        for i in range(0, len(res)-1, 4):
            inst = res[i]
            inst = eval(re.sub("(\_\d+)", "'\g<0>'", inst))
            #print(dec_var)
            con = res[i+1]               
            paths = res[i+2]      
            confs = eval(res[i+3])
            used = [ [] for i in range(ninstances)]
            for ipos, ilist in enumerate(inst):
                for fpos, f in enumerate(ilist):
                    r = pos2inst[ipos]+'.'+self.feature_names[fpos]
                    con2 = con.replace(f, r)
                    if con2 != con:
                        used[ipos].append(fpos)
                    con = con2
                    paths = paths.replace(f, r)
            for ipos, a in enumerate(used):
                # detect =1.0
                for fpos in a:
                    if self.feature_iscat[fpos]:
                        r = pos2inst[ipos]+'.'+self.feature_names[fpos]
                        if con.find(r+"=1.0")>=0:
                            # remove all other "=0.0
                            f = self.feature_original[fpos]
                            for v in self.df_code.encoded_atts([f]):
                                r = pos2inst[ipos]+'.'+v
                                con = re.sub(r+r"=0.0(\,)", "", con) 
                # second pass
                for fpos in a:
                    if self.feature_iscat[fpos]:
                        r = pos2inst[ipos]+'.'+self.feature_names[fpos]
                        val = self.feature_value[fpos]
                        s = pos2inst[ipos]+'.'+self.feature_original[fpos]
                        con = re.sub(r+r"=1.0", s+"="+val, con) 
                        con = re.sub(r+r"=0.0", s+"!="+val, con) 
            # output constraints
            con = con[1:-1]
            self.ans.append(con)
            if verbose >= 1:
                print('---\nAnswer constraint:', con)
            paths = paths.split("],[")
            paths[0] = paths[0][2:] # remove trailing []
            paths[-1] = paths[-1][:-2] 
            for i, (p, c) in enumerate(zip(paths, confs)):
                name = pos2inst[i]
                label = self.instances[name][1]
                l = self.df_code.decode[self.target][label]
                if verbose >= 2:
                    print('Rule satisfied by {}: IF {} THEN {} [{:.4f}]'.format(name, p, l, c))
                self.irules[name].append((p, c))
        if verbose>= 1 and len(res)<2:
            print('No answer')
      
    def instance(self, name, label, features=None, minconf=0, overwrite=True, model=None):
        self.recompile = True
        if name in self.instances:
            if not overwrite:
                raise "instance "+name+" exists already"
            n, _, _, _, _  = self.instances[name]
        else:
            n = len(self.instances)
        model = len(self.models)-1 if model is None else model
        if features is None:
            self.instances[name] = (n, label, minconf, model, [])
            return
        if len(features) != 1:
            raise "only one row in the data frame, please!"
        features = self.df_code.inverse_transform(features).reset_index()
        con = ",".join([name+"." + f + " = " + str(features.loc[0, f]) for f in self.pred_atts])
        self.instances[name] = 'dummy' # need to have the instance to call self.constraint()
        self.instances[name] = (n, label, minconf, model, self.constraint(con, only_ret=True))

    def model(self, clf, round_coef=2):
        self.recompile = True
        nf = len(self.feature_names)
        nm = len(self.models)
        res = "\n% path(Vars, Constraint, Pred, Conf) :- Constraint in a path of a decision tree over Vars with prediction Pred and confidence Conf"
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
                    pred, maxfreq = dautils.argmax(freqs)
                    maxfreq /= sum(freqs)
                    allf = ','.join( ('X'+str(i) if i in varset else '_') for i in range(nf) )
                    return "path({}, [{}], [{}], {}, {}).".format(nm, allf, body, classes_[pred], maxfreq)
            self.models.append(res + "\n" + recurse(0))
            return
        if isinstance(clf, LinearTreeClassifier):
            tree_ = clf.summary()
            if len(clf.classes_) != 2:
                raise ValueError("only binary model trees are admissible so far")
            def recurse(n, body="", varset=set(), round_coef=round_coef):
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
                    coef = [(round(v, round_coef) if abs(v)>=0.01 else 0) for v in coef]
                    threshold = round(float(node['models'].intercept_[0]), round_coef)
                    varset = varset | set([i for i, v in enumerate(coef) if v != 0])
                    allf = ','.join( ('X'+str(i) if i in varset else '_') for i in range(nf) )
                    maxfreq = 1 # TBD confidence to be calculated
                    # left
                    name = '+'.join(str(v)+'*X'+str(i) for i, v in enumerate(coef) if v != 0)
                    if body != '':
                        body = body + ','
                    body_left = body + "{} =< {}".format(name, threshold)
                    body_right = body + "{} > {}".format(name, threshold)
                    left = "path({},[{}],[{}],{},{}).".format(nm, allf, body_left, node['classes'][0], maxfreq)
                    right = "path({},[{}],[{}],{},{}).".format(nm, allf, body_right, node['classes'][1], maxfreq)
                    return left + "\n" + right
            self.models.append(res + "\n" + recurse(0))
            return
        raise ValueError("unknown model " + str(clf))

    def constraint(self, con, only_ret=False):
        self.recompile = True
        cons = [self.transform.toCLP(self.parse(c)) for c in re.split(r',\s*(?![^()]*\))', con)]
        if only_ret:
            return cons
        self.constraints.extend(cons)
        
    def bounds(self, att, minv, maxv):
        pos = self.feature_pos[att]
        if self.feature_iscat[pos] or self.feature_isord[pos]:
            raise Exception('bounds for continuous attributes only')
        self.feature_bounds[att] = (minv, maxv)

    def retract(self, con):
        self.recompile = True
        for c in re.split(r',\s*(?![^()]*\))', con):
            ret = self.transform.toCLP(self.parse(c))
            # retract from self.constraints
            self.constraints = [cs for cs in self.constraints if cs != ret]
            for name in self.instances:
                (n, label, minconf, model, con) = self.instances[name]
                con2 = [cs for cs in con if cs != ret]
                if len(con) != len(con2):
                    # retract from instances
                    self.instances[name] = (n, label, minconf, model, con2) 
