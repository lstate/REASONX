import pandas as pd
import numpy as np

# local imports
import sys
sys.path.append('../src/') # local path
import reasonx
import dautils

# adult dataset preparation

def read_adult(continuous_only=False, simplified=False, return_feature_type = False):
    # read dataset
    df = pd.read_csv('../data/adult_continuous.csv', na_values='?')
    # remove special characters in column names and values
    df.columns = df.columns.str.replace("[-&()]", "", regex=True)
    df = df.replace('[-&()]', '', regex=True)
    # missing values imputation with mode (needed for Decision Trees)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # nominal-ordinal-continuous partition of predictive attributes
    nominal_atts = [] if continuous_only else (['sex'] if simplified else ['race', 'sex', 'workclass'])
    ordinal_atts = [] if continuous_only or simplified else ['education']
    continuous_atts = ['age', 'capitalgain', 'hoursperweek'] if continuous_only or simplified else ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    # class attribute
    target = 'class'
    # predictive atts
    pred_atts = nominal_atts + ordinal_atts + continuous_atts
    # forcing encoding of class attribute (0=negatives, 1=positives)
    decode = {
        'class': {
            0: '<=50K', 1: '>50K'
        }
    }
    if not (simplified or continuous_only):
        decode['education'] = {
            1:'Preschool', 2:'1st4th', 3:'5th6th', 4:'7th8th', 5:'9th', 6:'10th', 7:'11th',
            8:'12th', 9:'HSgrad', 10:'Somecollege', 11:'Assocvoc', 12:'Assocacdm', 13:'Bachelors', 
            14:'Masters', 15:'Profschool', 16:'Doctorate' 
        }
    df = df[pred_atts+[target]]
    # encode nominal (as categories), ordinal+target (as int), passing the encoding of ordinal+target
    df_code = dautils.Encode(nominal=nominal_atts, ordinal=ordinal_atts+[target], decode=decode, onehot=True, prefix_sep="_")
    if return_feature_type == False:
        return df, pred_atts, target, df_code 
    else:
        return df, pred_atts, target, df_code, nominal_atts, ordinal_atts, continuous_atts

def df_numeric_stats(df, pred_atts, df_code):

    def true_precision(x, max_check=5, tol=1e-9):
        for d in range(max_check + 1):
            if abs(x - round(x, d)) < tol:
                return d
        return max_check
    
    continuous_atts = list(set(pred_atts)-set(df_code.nominal)-set(df_code.ordinal))
    df = df[continuous_atts]
    std_series = df.std()
    dict_stats = {
        col: (
            float(std_series[col]),  # std
            df[col].min(), # min
            df[col].max(), # max
            df[col].apply(true_precision).max(), # precision
            df[col].dtype # type
        )
        for col in continuous_atts
    }
    return dict_stats

def local_neigh(x_features, atts, numeric_stats, N=10000):
    large = 10e12
    cons, continuous_atts = [], sorted(numeric_stats)
    for f in continuous_atts:
        fval = x_features[f].iloc[0]
        std, minv, maxv, _, _ = numeric_stats[f]
        fvall, fvalu = max(fval-std, minv), min(fval+std, maxv)  
        cons.append(f"F.{f}>={fvall},F.{f}<={fvalu}")
    other_atts = [f for f in atts if f not in continuous_atts]
    con = ",".join(cons)
    # generate neighbor instances
    df_neigh = reasonx.hopsy_random(con, continuous_atts, n_samples=N)
    for f in continuous_atts:
        _, _, _, digits, typ = numeric_stats[f]
        if digits is not None:
            df_neigh[f] = df_neigh[f].round(digits)
        if typ is not None:
            df_neigh[f] = df_neigh[f].astype(typ)
    for f in other_atts:
        df_neigh[f] = x_features.iloc[0][f]
    df_neigh = df_neigh[atts]
    return df_neigh

# give me some credit dataset preparation

def read_give_me_some_credit(continuous_only=False, simplified=False):
    # read dataset
    df = pd.read_csv('../data/give_me_some_credit/cs-training.csv', na_values='?')
    # remove special characters in column names and values
    df.columns = df.columns.str.replace("[-&()]", "", regex=True)
    df = df.replace('[-&()]', '', regex=True)
    # missing values imputation with mode (needed for Decision Trees)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # nominal-ordinal-continuous partition of predictive attributes
    nominal_atts = []
    ordinal_atts = []
    continuous_atts = ['age', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime3059DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime6089DaysPastDueNotWorse', 'NumberOfDependents']
    # if continuous_only or simplified else [...]
    # class attribute
    target = 'SeriousDlqin2yrs'
    # predictive atts
    pred_atts = nominal_atts + ordinal_atts + continuous_atts
    # forcing encoding of class attribute (0=negatives, 1=positives)
    decode = {
        'SeriousDlqin2yrs': {
            0: 0, 1: 1
        }
    }
    df = df[pred_atts+[target]]
    # encode nominal (as categories), ordinal+target (as int), passing the encoding of ordinal+target
    df_code = dautils.Encode(nominal=nominal_atts, ordinal=ordinal_atts+[target], decode=decode, onehot=True, prefix_sep="_")
    return df, pred_atts, target, df_code 

# south german credit dataset preparation

def read_south_german_credit(continuous_only=False, simplified=False):
    # read dataset
    dff = np.loadtxt("../data/south_german_credit/SouthGermanCredit.asc", skiprows=1)
    df = pd.DataFrame(dff, columns=['Laufkont', 'Laufzeit', 'Moral', 'Verw', 'Hoehe', 'Sparkont', 'Beszeit', 'Rate', 'Famges', 'Buerge', 'Wohnzeit', 'Verm', 'Alter', 'Weitkred', 'Wohn', 'Bishkred', 'Beruf', 'Pers', 'Telef', 'Gastarb', 'Kredit'])
    # remove special characters in column names and values
    df.columns = df.columns.str.replace("[-&()]", "", regex=True)
    df = df.replace('[-&()]', '', regex=True)
    # missing values imputation with mode (needed for Decision Trees)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # nominal-ordinal-continuous partition of predictive attributes
    nominal_atts = []
    ordinal_atts = ['Laufkont', 'Moral', 'Verw', 'Sparkont', 'Beszeit', 'Rate', 'Famges', 'Buerge', 'Wohnzeit', 'Verm', 'Weitkred', 'Wohn', 'Bishkred', 'Beruf', 'Pers', 'Telef', 'Gastarb']
    continuous_atts = ['Laufzeit', 'Hoehe', 'Alter']
    # ordinal variabes that do not start at level zero
    df['Laufkont'] = df['Laufkont'] - 1
    df['Sparkont'] = df['Sparkont'] - 1
    df['Beszeit'] = df['Beszeit'] - 1
    df['Rate'] = df['Rate'] - 1
    df['Famges'] = df['Famges'] - 1
    df['Buerge'] = df['Buerge'] - 1
    df['Wohnzeit'] = df['Wohnzeit'] - 1
    df['Verm'] = df['Verm'] - 1
    df['Weitkred'] = df['Weitkred'] - 1
    df['Wohn'] = df['Wohn'] - 1
    df['Bishkred'] = df['Bishkred'] - 1
    df['Beruf'] = df['Beruf'] - 1
    df['Pers'] = df['Pers'] - 1
    df['Telef'] = df['Telef'] - 1
    df['Gastarb'] = df['Gastarb'] - 1
    # class attribute
    target = 'Kredit'
    # predictive atts
    pred_atts = nominal_atts + ordinal_atts + continuous_atts
    # forcing encoding of class attribute (0=negatives, 1=positives)
    decode = {
        'Kredit': {
            0: 0, 1: 1
        }
    }
    df = df[pred_atts+[target]]
    # encode nominal (as categories), ordinal+target (as int), passing the encoding of ordinal+target
    df_code = dautils.Encode(nominal=nominal_atts, ordinal=ordinal_atts+[target], decode=decode, onehot=True, prefix_sep="_")
    return df, pred_atts, target, df_code 

def read_credit_card_default(continuous_only=False, simplified=False):
    # read dataset
    df = pd.read_excel("../data/credit_card_default/credit_card_default.xls", skiprows=1)
    # remove special characters in column names and values
    df.columns = df.columns.str.replace("[-&()]", "", regex=True)
    df = df.replace('[-&()]', '', regex=True)
    # missing values imputation with mode (needed for Decision Trees)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # nominal-ordinal-continuous partition of predictive attributes
    df = df.rename(columns={
        'PAY_0':'PAY0', 'PAY_2':'PAY2', 'PAY_3':'PAY3', 'PAY_4':'PAY4', 'PAY_5':'PAY5', 'PAY_6':'PAY6', 
        'LIMIT_BAL':'LIMITBAL', 
        'BILL_AMT1':'BILLAMT1', 'BILL_AMT2':'BILLAMT2', 'BILL_AMT3':'BILLAMT3', 'BILL_AMT4':'BILLAMT4', 'BILL_AMT5':'BILLAMT5', 'BILL_AMT6':'BILLAMT6',
        'PAY_AMT1': 'PAYAMT1', 'PAY_AMT2': 'PAYAMT2', 'PAY_AMT3': 'PAYAMT3', 'PAY_AMT4': 'PAYAMT4', 'PAY_AMT5': 'PAYAMT5', 'PAY_AMT6': 'PAYAMT6'
    })
    nominal_atts = ['SEX', 'MARRIAGE', 'EDUCATION']
    ordinal_atts = ['PAY0', 'PAY2', 'PAY3', 'PAY4', 'PAY5', 'PAY6']
    continuous_atts = ['LIMITBAL', 'AGE', 'BILLAMT1', 'BILLAMT2', 'BILLAMT3', 'BILLAMT4', 'BILLAMT5', 'BILLAMT6','PAYAMT1', 'PAYAMT2', 'PAYAMT3', 'PAYAMT4', 'PAYAMT5', 'PAYAMT6']
    # nominal variables as strings (prepare one-hot-encoding) and replace by string values (default is label encoder)
    df['SEX'] = df['SEX'].astype('string')
    df['SEX'] = df['SEX'].replace({
        '1': 'male',
        '2': 'female'
    })
    df['MARRIAGE'] = df['MARRIAGE'].astype('string')
    # no info, should not occur according to documentation
    df['MARRIAGE'] = df['MARRIAGE'].replace({
        '0': 'unknown',
        '1': 'married',
        '2': 'single',
        '3': 'others'
    })
    # no info, should not occur according to documentation
    df['EDUCATION'] = df['EDUCATION'].astype('string')
    df['EDUCATION'] = df['EDUCATION'].replace({
        '0': 'unknown',
        '1': 'graduate_school',
        '2': 'university',
        '3': 'high_school',
        '4': 'others',
        '5': 'unknown',
        '6': 'unknown'
    })
    # delete negative features
    df = df.drop(['PAY0', 'PAY2', 'PAY3', 'PAY4', 'PAY5', 'PAY6', 'BILLAMT1', 'BILLAMT2', 'BILLAMT3', 'BILLAMT4', 'BILLAMT5', 'BILLAMT6'], axis=1)
    nominal_atts = ['SEX', 'MARRIAGE', 'EDUCATION']
    ordinal_atts = []
    continuous_atts = ['LIMITBAL', 'AGE','PAYAMT1', 'PAYAMT2', 'PAYAMT3', 'PAYAMT4', 'PAYAMT5', 'PAYAMT6']
    # class attribute
    target = 'target'
    # predictive atts
    pred_atts = nominal_atts + ordinal_atts + continuous_atts
    # forcing encoding of class attribute (0=negatives, 1=positives)
    decode = {
        'target': {
            0: 0, 1: 1
        }
    }
    df = df[pred_atts+[target]]
    # encode nominal (as categories), ordinal+target (as int), passing the encoding of ordinal+target
    df_code = dautils.Encode(nominal=nominal_atts, ordinal=ordinal_atts+[target], decode=decode, onehot=True, prefix_sep="_")
    return df, pred_atts, target, df_code 

def read_australian_credit(continuous_only=False, simplified=False):
    # read dataset
    df = pd.read_csv("../data/australian_credit_approval/australian.dat", sep=  " ",  header=None)
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6','A7','A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    df.columns = column_names
    #ddf = np.fromfile("../data/australian_credit_approval/australian.dat")
    #df = pd.DataFrame(data=ddf)
    # remove special characters in column names and values
    #df.columns = df.columns.str.replace("[-&()]", "", regex=True)
    #df = df.replace('[-&()]', '', regex=True)
    # missing values imputation with mode (needed for Decision Trees)
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # nominal-ordinal-continuous partition of predictive attributes
    nominal_atts = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    ordinal_atts = []
    continuous_atts = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    # nominal variables as strings (prepare one-hot-encoding)
    df['A1'] = df['A1'].astype('string')
    df['A1'] = df['A1'].replace({
        '0': 'a',
        '1': 'b'
    })
    
    df['A4'] = df['A4'].astype('string')
    df['A4'] = df['A4'].replace({
        '1': 'p',
        '2': 'g',
        '3': 'gg'
    })

    df['A5'] = df['A5'].astype('string')
    df['A5'] = df['A5'].replace({
        '1': 'ff',  '2': 'd',   '3': 'i',   '4': 'k',
        '5': 'j',   '6': 'aa',  '7': 'm',   '8': 'c',
        '9': 'w',   '10': 'e',  '11': 'q',  '12': 'r',
        '13': 'cc', '14': 'x'
    })   
    
    df['A6'] = df['A6'].astype('string')
    df['A6'] = df['A6'].replace({
        '1': 'ff', '2': 'dd', '3': 'j', 
        '4': 'bb', '5': 'v',  '6': 'n', 
        '7': 'o',  '8': 'h',  '9': 'z'
    }) 
    
    df['A8'] = df['A8'].astype('string')
    df['A8'] = df['A8'].replace({
        '1': 't',
        '0': 'f'
    })    
    
    df['A9'] = df['A9'].astype('string')
    df['A9'] = df['A9'].replace({
        '1': 't',
        '0': 'f'
    })    

    df['A11'] = df['A11'].astype('string')
    df['A11'] = df['A11'].replace({
        '1': 't',
        '0': 'f'
    })    
    
    df['A12'] = df['A12'].astype('string')
    df['A12'] = df['A12'].replace({
        '1': 's',
        '2': 'g',
        '3': 'p'
    })   

    # delete negative features
    # class attribute
    target = 'A15'
    # predictive atts
    pred_atts = nominal_atts + ordinal_atts + continuous_atts
    # forcing encoding of class attribute (0=negatives, 1=positives)
    decode = {
        'target': {
            0: 0, 1: 1
        }
    }
    df = df[pred_atts+[target]]
    # encode nominal (as categories), ordinal+target (as int), passing the encoding of ordinal+target
    df_code = dautils.Encode(nominal=nominal_atts, ordinal=ordinal_atts+[target], decode=decode, onehot=True, prefix_sep="_")
    return df, pred_atts, target, df_code 

def linf_norm_df(df_row_1, df_row_2, weights):
    # linf for all types of variables
    # weights as list with same length as columns dataframe
    delta = weights * (df_row_1.astype(float) - df_row_2.astype(float)).abs()
    return np.max(delta)

def l1_norm_df(df_row_1, df_row_2, weights):
    # l1 for all types of variables
    # weights as list with same length as columns dataframe
    delta = weights * (df_row_1.astype(float) - df_row_2.astype(float)).abs()
    return np.sum(delta)

def get_linf_weights(encoded_pred_atts, pred_atts, df_code):
    # default is one hot encoded nominal feature
    continuous_atts = list(set(pred_atts)-set(df_code.nominal)-set(df_code.ordinal))
    weights = [1] * len(encoded_pred_atts)
    for i, f in enumerate(encoded_pred_atts):
        # continous features
        if f in continuous_atts:
            weights[i] = (1/(df_code.encode[f][1] - df_code.encode[f][0]))
        # label encoded ordinal features (decoder)
        elif f in df_code.ordinal:
            weights[i] = (1/(max(df_code.decode[f]) - min(df_code.decode[f])))
    return weights
    
def get_l1_weights(encoded_pred_atts, pred_atts, df_code):
    # default is one hot encoded nominal feature
    continuous_atts = list(set(pred_atts)-set(df_code.nominal)-set(df_code.ordinal))
    weights = [0.5] * len(encoded_pred_atts)
    for i, f in enumerate(encoded_pred_atts):
        # continous features
        if f in continuous_atts:
            weights[i] = (1/(df_code.encode[f][1] - df_code.encode[f][0]))
        # label encoded ordinal features (decoder)
        elif f in df_code.ordinal:
            weights[i] = (1/(max(df_code.decode[f]) - min(df_code.decode[f])))
    return weights

def get_distances(x_features, df, l1weights, linfweights):
    l1 = [l1_norm_df(x_features, df.iloc[i,:], l1weights) for i in range(len(df))]
    linf = [linf_norm_df(x_features, df.iloc[i,:], linfweights) for i in range(len(df))]
    return (l1, linf)

# evaluation for one data instance

def evaluation_return_array(r, X, y, n_instances, minconf_f = 0, minconf_ce = 0, constraints_fce = ""):
    
    evaluation_list_ = np.empty((10,n_instances))
    evaluation_list_[:] = np.nan
    
    for i in range(n_instances):
        # return only one instance (case c)
        if n_instances == 1:
            r.instance('F', features=X, label=y, minconf = minconf_f)
        else:
            r.instance('F', features=X.iloc[i:i+1], label=y.iloc[i], minconf = minconf_f)

        a, b, c, d, e = r.solveopt(evaluation = 1)

        # a - number of solutions
        # b - distance (only for opt)
        # c - number of premises in rule
        # c[0] refers to first solution and all rules > here: factual rule
        # d - number of constraints in "answer constraints"
        # e - dimensionality check (only for opt)

        evaluation_list_[0, i] = a

        # no F solution (disagreement label/minconf)
        if a == 0:
            r.reset(keep_model=True)
            continue

        else:
            evaluation_list_[1, i] = c[0][0]
            # return only one instance (case c)
            if n_instances == 1:
                r.instance('CF', label=1-y, minconf = minconf_ce)
            else:
                r.instance('CF', label=1-y.iloc[i], minconf = minconf_ce)

            if len(constraints_fce) > 0:
                r.constraint(constraints_fce)
                
            a, b, c, d, e = r.solveopt(evaluation= 1)

            # a - number of solutions > here: admissible CF pathes
            # b - distance (only for opt)
            # c - number of premises in rule
            # c[0] refers to first solution and all rules > here: factual and CE rule
            # d - number of constraints in "answer constraints"
            # e - dimensionality check

            evaluation_list_[3, i] = a

            # no CE solution (minconf)
            if a == 0:        
                # set also number of solutions for l1/linf to zero
                evaluation_list_[4, i] = 0
                evaluation_list_[7, i] = 0
                r.reset(keep_model=True)
                continue

            else:
                evaluation_list_[2, i] = np.mean(c, axis = 0)[1]
                
                print("L1 NORM \n")
                a, b, c, d, e = r.solveopt(minimize='l1norm(F, CF)', evaluation=1, eps = 0.01)

                # a - number of solutions
                # b - distance (only for opt)
                # c - number of premises in rule
                # c[0] refers to first solution and all rules > here: factual and CE rule
                # d - number of constraints in "answer constraints"
                # e - dimensionality check

                evaluation_list_[4, i] = a
                evaluation_list_[5, i] = np.mean(b)
                evaluation_list_[6, i] = np.mean(e, axis = 0)

                print("Linf NORM \n")
                a, b, c, d, e = r.solveopt(minimize='linfnorm(F, CF)', evaluation=1, eps = 0.01)

                # a - number of solutions
                # b - distance (only for opt)
                # c - number of premises in rule
                # c[0] refers to first solution and all rules > here: factual and CE rule
                # d - number of constraints in "answer constraints"
                # e - dimensionality check

                evaluation_list_[7, i] = a
                evaluation_list_[8, i] = np.mean(b)
                evaluation_list_[9, i] = np.mean(e, axis = 0)

                r.reset(keep_model=True)

    return(evaluation_list_)