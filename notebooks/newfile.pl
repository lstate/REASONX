:- use_module(library(clpr)).
:- use_module(library(lists)).
% feature(pos, name) :- name for the i-th feature
feature(0, vrace_AmerIndianEskimo).
feature(1, vrace_AsianPacIslander).
feature(2, vrace_Black).
feature(3, vrace_Other).
feature(4, vrace_White).
feature(5, vsex_Female).
feature(6, vsex_Male).
feature(7, vworkclass_Federalgov).
feature(8, vworkclass_Localgov).
feature(9, vworkclass_Neverworked).
feature(10, vworkclass_Private).
feature(11, vworkclass_Selfempinc).
feature(12, vworkclass_Selfempnotinc).
feature(13, vworkclass_Stategov).
feature(14, vworkclass_Withoutpay).
feature(15, veducation).
feature(16, vage).
feature(17, vcapitalgain).
feature(18, vcapitalloss).
feature(19, vhoursperweek).
nfeatures(20).
% cat_features(name, positions) :- positions of categorical features
cat_features(vworkclass, [7, 8, 9, 10, 11, 12, 13, 14]).
cat_features(vrace, [0, 1, 2, 3, 4]).
cat_features(vsex, [5, 6]).
% ord_features_pos(positions) :- positions of ordinal features
ord_features_pos([15], [(1, 16)]).
% norm_weights(weights) :- features weights
norm_weights([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.06666666666666667, 0.0136986301369863, 1.000010000100001e-05, 0.0002295684113865932, 0.01020408163265306]).
% data_instance(id, name, class, minconf, model) :- instance id, name, class label, minumal confidence, model
data_instance(0, iF, 0, 0, 0).
data_instance(1, iCE, 1, 0.6, 0).
ninstances(2).
% data_instance_proj(id, pos) :- positions to project instance id
data_instance_proj(0, [16]).
data_instance_proj(1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).
% path(Vars, Constraint, Pred, Conf) :- Constraint in a path of a decision tree over Vars with prediction Pred and confidence Conf
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X15,_,X17,X18,_], [X17 =< 5119.0,X15 =< 12.5,X18 =< 1820.5], 0, 0.9754769507901416).
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X15,_,X17,X18,_], [X17 =< 5119.0,X15 =< 12.5,X18 > 1820.5], 1, 0.5886654478976234).
path(0, [_,_,_,_,_,X5,_,_,_,_,_,_,_,_,_,X15,_,X17,_,_], [X17 =< 5119.0,X15 > 12.5,X5 =< 0.5], 1, 0.5452975047984645).
path(0, [_,_,_,_,_,X5,_,_,_,_,_,_,_,_,_,X15,_,X17,_,_], [X17 =< 5119.0,X15 > 12.5,X5 > 0.5], 0, 0.9363327674023769).
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X17,_,_], [X17 > 5119.0,X17 =< 7055.5,X17 =< 5316.5], 1, 1.0).
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X17,_,_], [X17 > 5119.0,X17 =< 7055.5,X17 > 5316.5], 0, 0.74).
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X16,X17,_,_], [X17 > 5119.0,X17 > 7055.5,X16 =< 20.0], 0, 0.8).
path(0, [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X16,X17,_,_], [X17 > 5119.0,X17 > 7055.5,X16 > 20.0], 1, 0.9902642559109874).
% user_constraints(+Vars, -Cs)
user_constraints(Vars, Cs) :-
            Constraints_list = [var(iF, vrace_Black)=1, var(iF, vsex_Male)=1, var(iF, vworkclass_Private)=1, var(iF, veducation)=10, var(iF, vcapitalgain)=0.0, var(iF, vcapitalloss)=0.0, var(iF, vhoursperweek)=40.0, var(iCE, vage)=var(iF, vage), var(iF, vage)=<19.0], 
            exp2cons(Constraints_list, Vars, Cs).
            
min_norm(l1norm(iF,iCE)).
:- ['post.pl'].