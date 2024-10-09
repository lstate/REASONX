%% no optimization -------------------------------------------------------------------------------------
q_nominimize :- 
	query_nominimize(Vars, ResCs, ResPs, Probs),
	length(ResCs, LResCs),
	lengthsublist(ResPs, LResPs),
	print(Vars), nl,
	print(ResCs), nl,
	print(ResPs), nl,
	print(Probs), nl,
	print(LResPs), nl,
	print(LResCs), nl,
	fail.

%% optimization ----------------------------------------------------------------------------------------
q_minimize :- 
	query_minimize(Vars, ResCs, ResPs, Probs, Min),
	length(ResCs, LResCs),
	lengthsublist(ResPs, LResPs),
	print(Vars), nl,
	print(ResCs), nl,
	print(ResPs), nl,
	print(Probs), nl,
	print(Min), nl,
	print(LResPs), nl,
	print(LResCs), nl,
	fail.	
	
%% meta-reasoning over constraints

%% no optimization -------------------------------------------------------------------------------------
query_nominimize(Vars, ResCs, ResPs, Probs) :-
	instvar(Vars),
	proj_vars(Vars, LVars),
	findall(inst(I), data_instance(_, I, _, _, _), Is), !,
	solve(project(satisfiable(cross([unp(typec), userc | Is])), LVars), Vars, ResCs, _, Extras),
	extract(Extras, LVars, ResPs, Probs).

%% optimization ----------------------------------------------------------------------------------------
% added Min (minimization value)
query_minimize(Vars, ResCs, ResPs, Probs, Min) :-
	instvar(Vars),
	proj_vars(Vars, LVars),
	findall(inst(I), data_instance(_, I, _, _, _), Is), !,
	solve(project(minimize(relax(cross([unp(typec), userc | Is]))), LVars), Vars, ResCs, _, [Min|Extras]),
	extract(Extras, LVars, ResPs, Probs).

%% -----------------------------------------------------------------------------------------------------
%% -----------------------------------------------------------------------------------------------------

solve(empty, _, [], [], []).

solve(userc, Vars, Cons, [], []) :-
	user_constraints(Vars, Cons).

solve(typec, Vars, Cons, [], []) :-
	cat_constraints(Vars, CCat),
	ord_constraints(Vars, COrd),
	append(CCat, COrd, Cons).

solve(inst(I), Vars, R, [], [(R, L, P)]) :-
	data_instance(N, I, L, Pr, M),
	nth0(N, Vars, V),
	path(M, V, R, L, P),
	P >= Pr.

solve(cross([]), _, [], [], []).
solve(cross([T|Ts]), Vars, Cons, OCons, Extras) :-
	solve(T, Vars, Cs0, Os0, Extras0),
	solve(cross(Ts), Vars, Cs1, Os1, Extras1),
	append(Cs0, Cs1, Cons),
	append(Os0, Os1, OCons), 
	append(Extras0, Extras1, Extras).

solve(satisfiable(T), Vars, Cons, OCons, Extras) :-
	solve(T, Vars, Cons, OCons, Extras),
	int_vars(Vars, OrdVars),
	append(Cons, OCons, ACons),
	satisfiable(ACons, OrdVars).

solve(minimize(T), Vars, Cons, OCons, [Min|Extras0]) :-
	solve(T, Vars, Cs, OCons, Extras0),
	int_vars(Vars, OrdVars),
	min_norm(MN),
	exp_eval(MN, Vars, Norm, CNorm),
	append(CNorm, Cs, CTmp),
	append(CTmp, OCons, CMin),
	satisfiable(CMin, OrdVars, Norm, Min, Vertex),
	eq_con(OrdVars, Vertex, CEq),
	append([Norm=Min|CNorm], Cs, Cs1),
	append(CEq, Cs1, Cons).

solve(relax(T), Vars, Cons, OCons, Extras) :-
	solve(T, Vars, Cs, Os, Extras),
	relax(Cs, Cons),
	relax(Os, OCons).
	
solve(unp(T), Vars, [], OCons, Extras) :-
	solve(T, Vars, Cons, Os, Extras),
	append(Cons, Os, OCons).

solve(project(T, PVars), Vars, Cons, OCons, Extras) :-
	solve(T, Vars, Cs, Os, Extras),
	project(PVars, Cs, Cons),
	project(PVars, Os, OCons).

eq_con([], [], []).
eq_con([X|Xs], [Y|Ys], [X=Y|Cs]) :-
	eq_con(Xs, Ys, Cs).

extract([], _, [], []).
extract([(A, _, C)|Es], LVars, [Con|As], [C|Cs]) :-
	project(LVars, A, Con),
	extract(Es, LVars, As, Cs).

int_vars(Vars, OrdVars) :-
	findall(Ps, cat_features(_, Ps), PPs),
	ord_features_pos(Ord, _),
	flatten([Ord|PPs], Pos),
	varpos(Pos, Vars, Ords),
	flatten(Ords, OrdVars).

proj_vars(Vars, ProjVars) :-
	proj_vars(0, Vars, PVars),
	flatten(PVars, ProjVars).

proj_vars(_, [], []).

proj_vars(N, [V|Vars], [Pv|ProjVars]) :-
	data_instance_proj(N, Pos),
	varpos_i(Pos, V, Pv),
	N1 is N+1,
	proj_vars(N1, Vars, ProjVars).

%%%%%%% returns constraints on one-hot-encoded categorical variables
%  cat_constraints
%
% :- instvar(Vars), cat_constraints(Vars, Cs).
%%%%%%%%%%%%%%

ord_constraints(Vars, Res) :-
	ord_features_pos(Ord, Bounds),
	varpos(Ord, Vars, VXs),
	ord_constraints_var(VXs, Bounds, Res).

ord_constraints_var([], _, []).
ord_constraints_var([V|Vs], Bounds, Res) :-
	ord_constraints_pos(V, Bounds, C),
	ord_constraints_var(Vs, Bounds, Cs),
	append(C, Cs, Res).

ord_constraints_pos([], [], []).
ord_constraints_pos([V|Vs], [(L,U)|B], [U >= V, V >= L|Cs]) :-
	ord_constraints_pos(Vs, B, Cs).

cat_constraints(Vars, Res) :-
	findall(Ps, cat_features(_, Ps), PPs),
	cat_constraints_pos(Vars, PPs, Res).
	
cat_constraints_pos(_, [[]], []) :- !.
cat_constraints_pos(_, [], []).
cat_constraints_pos(Vars, [Ps|PPs], Res) :-
	cat_constraints_var(Vars, Ps, Res1),
	cat_constraints_pos(Vars, PPs, Res2),
	append(Res1, Res2, Res).

cat_constraints_var([], _, []).
cat_constraints_var([V|Vars], Ps, [Ss=1|Res]) :-
	varpos_i(Ps, V, VXs),
	range(VXs, 0, 1, Cs),
	list2sum(VXs, Ss), 
	cat_constraints_var(Vars, Ps, Res1),
	append(Cs, Res1, Res).
	
range([], _, _, []).
range([X|Xs], L, H, [H>=X,X>=L|Cs]) :-
	range(Xs, L, H, Cs).

list2sum([X], X).
list2sum([X, Y|Xs], X+S) :-
	list2sum([Y|Xs], S).

varpos(_, [], []).
varpos(Ps, [INST|Vars], [V|Vs]) :-
	varpos_i(Ps, INST, V),
	varpos(Ps, Vars, Vs).

varpos_i([], _, []).
varpos_i([P|Ps], INST, [VX|VXs]) :-
	nth0(P, INST, VX),
	varpos_i(Ps, INST, VXs).

%%%%%%% list of expressions on variables to constraints
%  instvar, exp2cons, exp2con
%
% :- instvar(Vars), exp2cons([var(iF, vage) =< var(iCF, vage)], Vars, Cons).
%%%%%%%%%%%%%%

instvar(Vars) :-
	ninstances(Ni),
	nfeatures(Nf),
	length(Vars, Ni),
	lengths(Vars, Nf).

exp2cons([], _, []).
exp2cons([E|Es], Vars, Cs) :-
	exp2con(E, Vars, C1s),
	exp2cons(Es, Vars, C2s),
	append(C1s, C2s, Cs).

exp2con(X=Y, Vars, [VX=VY|Cs]) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp2con(X=<Y, Vars, [VX=<VY|Cs]) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp2con(X<Y, Vars, [VX<VY|Cs]) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp2con(X>=Y, Vars, [VY=<VX|Cs]) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).
	
exp2con(X>Y, Vars, [VY<VX|Cs]) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp_eval(var(I, V), Vars, VX, []) :- !,
	data_instance(N, I, _, _, _),
	nth0(N, Vars, INST),
	feature(P, V),
	nth0(P, INST, VX).

exp_eval(N*X, Vars, N*VX, Cs) :- !,
	number(N), exp_eval(X, Vars, VX, Cs).

exp_eval(X*N, Vars, N*VX, Cs) :- !,
	number(N), exp_eval(X, Vars, VX, Cs).

exp_eval(X+Y, Vars, VX+VY, Cs) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp_eval(X-Y, Vars, VX-VY, Cs) :- !,
	exp_eval(X, Vars, VX, CXs), exp_eval(Y, Vars, VY, CYs), append(CXs, CYs, Cs).

exp_eval(l1norm(I1, I2), Vars, S, Cs) :- !,
	data_instance(N1, I1, _, _, _),
	data_instance(N2, I2, _, _, _),
	nth0(N1, Vars, INST1),
	nth0(N2, Vars, INST2),
	l1_con(INST1, INST2, Cs, S).

% - TBD: l1 norm with diversity
% l1normd(I1, I2s, Vars, S, Cs)

exp_eval(linfnorm(I1, I2), Vars, S, Cs) :- !,
	data_instance(N1, I1, _, _, _),
	data_instance(N2, I2, _, _, _),
	nth0(N1, Vars, INST1),
	nth0(N2, Vars, INST2),
	linf_con(INST1, INST2, Cs, S).

% LAURA diversity optimization
% distance between F and one/several CE (as list)
exp_eval(l1normd(_, []), _, 0, []).
exp_eval(l1normd(I1, [I2Head|I2Tail]), Vars, S + STail, NewCs) :- !,
	data_instance(N1, I1, _, _, _),
	% head
	data_instance(N2List, I2Head, _, _, _),
	nth0(N1, Vars, INST1),
	nth0(N2List, Vars, INST2),
	l1_con(INST1, INST2, Cs, S),
	% tail
	exp_eval(l1normd(I1, I2Tail), Vars, STail, CsTail),
	append(Cs, CsTail, NewCs).

% distance between F and one/several CE (as list)
exp_eval(l1norml(_, []), _, 0, []).
exp_eval(l1norml(I1, [I2Head|I2Tail]), Vars, S + STail, NewCs) :- !,
	data_instance(N1, I1, _, _, _),
	% head
	data_instance(N2List, I2Head, _, _, _),
	nth0(N1, Vars, INST1),
	nth0(N2List, Vars, INST2),
	l1_con(INST1, INST2, Cs, S),
	% tail
	exp_eval(l1norml(I1, I2Tail), Vars, STail, CsTail),
	append(Cs, CsTail, NewCs).
	
% distance between one/several CE (as list)
exp_eval(l1normll([], _), _, 0, []).
exp_eval(l1normll([I1Head|I1Tail], [I2Head|I2Tail]), Vars, S + STail, NewCs) :-
	% head
	exp_eval(l1norml(I1Head, [I2Head|I2Tail]), Vars, S, Cs),
	% tail
	exp_eval(l1normll(I1Tail, [I2Head|I2Tail]), Vars, STail, CsTail),
	append(Cs, CsTail, NewCs).

exp_eval(X, _, X, []) :-
	number(X), !.

l1_con(Inst1, Inst2, Cs, Norm) :-
	l1_weights(W),
	l1_con(W, Inst1, Inst2, Cs, Norm).	

l1_con([], [], [], [], 0).
l1_con([W|Ws], [X|Xs], [Y|Ys], [S >= X-Y, S >= Y-X|Cs], W*S+Sum) :-
	l1_con(Ws, Xs, Ys, Cs, Sum).

linf_con(Inst1, Inst2, Cs, Norm) :-
	linf_weights(W),
	linf_con(W, Inst1, Inst2, Cs, Norm).	

linf_con([], [], [], [], _).
linf_con([W|Ws], [X|Xs], [Y|Ys], [S >= W*(X-Y), S >= W*(Y-X)|Cs], S) :-
		linf_con(Ws, Xs, Ys, Cs, S).


%%%%%%% basic operations on linear constraints

nentails(_, [], 0).
nentails(S1, [C|S2], N) :-
	nentails(S1, S2, N1),
	(entails(S1, C) -> N is N1+1; N is N1).

nconsistent(_, [], 0).
nconsistent(S1, [C|S2], N) :-
	nconsistent(S1, S2, N1),
	(satisfiable([C|S1]) -> N is N1+1; N is N1).

satisfiable(P) :-
	copy_term(P, CopyP),
	tell_cs(CopyP).	

satisfiable(P, Ints) :-
	copy_term(P-Ints, CopyP-CopyInts),
	tell_cs(CopyP),
	bb_inf(CopyInts, 0, _).

satisfiable(P, Ints, Norm, Min) :-
	copy_term(P-Ints-Norm, CopyP-CopyInts-CopyNorm),
	tell_cs(CopyP),
	bb_inf(CopyInts, CopyNorm, Min).

satisfiable(P, Ints, Norm, Min, Vertex) :-
	copy_term(P-Ints-Norm, CopyP-CopyInts-CopyNorm),
	tell_cs(CopyP),
	relax_eps(E),
	bb_inf(CopyInts, CopyNorm, Min, Vertex, E).

projects([], [], []).
projects([X|Xs], [C|Cxs], [P|Ps]) :-
	project(X, C, P),
	projects(Xs, Cxs, Ps).

equivalent(S, C) :-
	entails(S, C),
	entails(C, S).

entails(S, C) :-
	copy_term(S-C, S1-C1),
	tell_cs(S1),
	is_entailed(C1).
	
is_entailed([]).
is_entailed([C|Cs]) :- 
	entailed(C),
	is_entailed(Cs).

is_constant(V, P, C) :-
	copy_term(V-P, CopyV-CopyP),
	tell_cs(CopyP),
	sup(CopyV, C), 
	inf(CopyV, C).
	
tells_cs([]).
tells_cs([C|Cs]) :-  
	tell_cs(C),
	tells_cs(Cs).
	
relax([], []).
relax([X=Y|Cs], [X=Y|Rs]) :-
	relax(Cs, Rs).
relax([X=<Y|Cs], [X=<Y|Rs]) :-
	relax(Cs, Rs).
relax([X>=Y|Cs], [X>=Y|Rs]) :-
	relax(Cs, Rs).
relax([X<Y|Cs], [X+E =<Y|Rs]) :-
	relax_eps(E),
	relax(Cs, Rs).
relax([X>Y|Cs], [X-E >=Y|Rs]) :-
	relax_eps(E),
	relax(Cs, Rs).

%%%%%%% project and prepare_dump from 
%% Florence Benoy, Andy King, Frédéric Mesnard: Computing convex hulls with a linear solver. Theory Pract. Log. Program. 5(1-2): 259-271 (2005)

project(Xs, Cxs, ProjectCxs) :-
	call_residue_vars( copy_term(Xs-Cxs, CpyXs-CpyCxs), _),
	tell_cs(CpyCxs),
	prepare_dump(CpyXs, Xs, Zs, DumpCxs, ProjectCxs),
	dump(Zs, Vs, DumpCxs), Xs = Vs.

prepare_dump([], [], [], Cs, Cs).
prepare_dump([X|Xs], YsIn, ZsOut, CsIn, CsOut) :-
	(ground(X) ->
	YsIn = [Y|Ys],
	ZsOut = [_|Zs],
	CsOut = [Y=X|Cs]
	;
	YsIn = [_|Ys],
	ZsOut = [X|Zs],
	CsOut = Cs
	),
	prepare_dump(Xs, Ys, Zs, CsIn, Cs).

tell_cs([]).
tell_cs([C|Cs]) :-  
	{C}, 
	tell_cs(Cs).
	
%%%%%%% utilities

%% list of lists, each of length N

lengths([], _).
lengths([V|Vs], N) :-
	length(V, N),
	lengths(Vs, N).

%% length of sublists (list inside of lists)
lengthsublist([], []).
lengthsublist([V|Vs], [L|Ls]) :-
	length(V, L),
	lengthsublist(Vs, Ls).

cls :- write('\33\[2J').

:- set_prolog_flag(toplevel_print_options, [quoted(true), portray(true)]). % to print full list
:- set_prolog_flag(answer_write_options,[max_depth(0)]). % to print full list