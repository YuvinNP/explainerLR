import math
import pandas as pd
import pickle
import numpy as np
from collections import OrderedDict
from six import string_types
from chefboost.training import Training

rules_list = []

def decode_data(encoder, data, columns):
    
    decoded_data = encoder.inverse_transform(data)
    decoded_data_df = pd.DataFrame(decoded_data, columns=columns)
    
    return decoded_data, decoded_data_df

def calculate_distance(training_set, predict):
    
    total = 0
    output = []
    for index, outer_row in predict.iterrows():
        for index, inner_row in training_set.iterrows():
            row_total = 0
            for e in range(len(predict.columns)):
               
                row_total = row_total + (int(outer_row[e]) - int(inner_row[e]))**2

            output.append(math.sqrt(row_total))
    
    return output

def get_neighbhours(n, df):
    
    neighbout_count = int(len(df) * n)

    neighbours = df[0:neighbout_count]

    return neighbours


def dump_neighbhours(neighbours):
    neighbour_dic = {}
    
    neighbours_updated = neighbours.drop(columns = 'distance')

    for key in neighbours_updated.columns:
        neighbour_dic[key] = neighbours_updated[key].tolist()

    with open('neighbour.json', 'wb') as fp:
        pickle.dump(neighbour_dic, fp)
        
        
def deldup(li):
    """ Deletes duplicates from list _li_
        and return new list with unique values.
    """
    return list(OrderedDict.fromkeys(li))


def is_mono(t):
    """ Returns True if all values of _t_ are equal
        and False otherwise.
    """
    for i in t:
        if i != t[0]:
            return False
    return True


def get_indexes(table, col, v):
    """ Returns indexes of values _v_ in column _col_
        of _table_.
    """
    li = []
    start = 0
    for row in table[col]:
        if row == v:
            index = table[col].index(row, start)
            li.append(index)
            start = index + 1
    return li


def get_values(t, col, indexes):
    """ Returns values of _indexes_ in column _col_
        of the table _t_.
    """
    return [t[col][i] for i in range(len(t[col])) if i in indexes]


def del_values(t, ind):
    """ Creates the new table with values of _ind_.
    """
    return {k: [v[i] for i in range(len(v)) if i in ind] for k, v in t.items()}


def add_to_dic_list(rule):
    
    dict = {}
    
    for i in rule:
        kv = i.split(':')
        dict[kv[0]] = kv[1]
#     print(dict)
    rules_list.append(dict)
    
def formalize_rules(list_rules):
    """ Gives an list of rules where
        facts are separeted by coma.
        Returns string with rules in
        convinient form (such as
        'If' and 'Then' words, etc.).
    """
    text = ''
    for r in list_rules:
        t = [i for i in r.split(',') if i]
        # print('rules', t)
        add_to_dic_list(t)
        text += 'If %s:' % t[0]
        for i in t[1:-1]:
            text += ' %s:' % i
            # print('TEXT1: ', text)
        text += ' Then => %s.\n' % t[-1]
    # print('TEXT: ', text)
    return text


def get_subtables(t, col):
    """ Returns subtables of the table _t_
        divided by values of the column _col_.
    """
    return [del_values(t, get_indexes(t, col, v)) for v in deldup(t[col])]

def freq(table, col, v):
    """ Returns counts of variant _v_
        in column _col_ of table _table_.
    """
    return table[col].count(v)


def info(table, res_col):
    """ Calculates the entropy of the table _table_
        where res_col column = _res_col_.
    """
    s = 0 # sum
    for v in deldup(table[res_col]):
        p = freq(table, res_col, v) / float(len(table[res_col]))
        s += p * math.log(p, 2)
    return -s


def infox(table, col, res_col):
    """ Calculates the entropy of the table _table_
        after dividing it on the subtables by column _col_.
    """
    s = 0 # sum
    for subt in get_subtables(table, col):
        s += (float(len(subt[col])) / len(table[col])) * info(subt, res_col)
    return s
    


def gain(table, x, res_col):
    """ The criterion for selecting attributes for splitting.
    """
    # print(info(table, res_col) - infox(table, x, res_col))
    return info(table, res_col) - infox(table, x, res_col)



def get_rules_list():
    return rules_list

def findGain(df, value, name):
    config = {'algorithm': 'ID3'}
    
    idx = df[df[name] <= value].index
    
    tmp_df = df.copy()
    tmp_df[name] = '>' + str(value)
    tmp_df.loc[idx, name] = '<=' + str(value)
    
    gain = Training.findGains(tmp_df, config)['gains'][name]
    # print(value, ": ", gain)
    
    return value, gain, tmp_df

def get_neighbours_con(df, columns, target):
    
    # config = {'algorithm': 'ID3'}
            
    output = df.copy()
    
    max_val = {}
    
    for name in columns:
        gain_list = {}

        df1 = output.copy()
        
        output.drop([name], axis=1, inplace=True)
        
        df1 = df1[[name, target]]
        df1.columns = [name, 'Decision']            
        
        uniques = sorted(df1[name].unique())
            
        for i in uniques:
            gain_list[findGain(df1, i, name)[0]] = findGain(df1, i, name)[1]
        
        max_key = max(gain_list, key=gain_list.get)

        df1 = findGain(df1, max_key, name)[2]
        max_val[name] = max_key
        
        output[name] = df1[name].tolist()
        
    return output, max_val
            
            

def mine_c45(table, result):
    error =  None
    try:
        col = max([(k, gain(table, k, result)) for k in table.keys() if k != result],
                key=lambda x: x[1])[0]
        tree = []
        for subt in get_subtables(table, col):
            v = subt[col][0]
            if is_mono(subt[result]):
                tree.append(['%s:%s' % (col, v),
                            '%s:%s' % (result, subt[result][0])])
            else:
                del subt[col]
                tree.append(['%s:%s' % (col, v)] + mine_c45(subt, result))
    except Exception as e:
        error = e
        
    return tree


def tree_to_rules(tree):
    # print(tree)
    return formalize_rules(__tree_to_rules(tree))

def __tree_to_rules(tree, rule=''):
    rules = []
    for node in tree:
        if isinstance(node, string_types):
            rule += node + ','

        else:
            rules += __tree_to_rules(node, rule)

    if rules:
        return rules
    return [rule]

def validate_table(table):
    assert isinstance(table, dict)
    for k, v in table.items():
        assert k
        assert isinstance(k, string_types)
        assert len(v) == len(table.values()[0])
        for i in v: assert i
        
def load_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        
    return data

def build_decision_rules(data, target):
    
    trees = mine_c45(data, result = target)
    return tree_to_rules(trees)

def model_predict(columns, class_name):
    dict = columns
    output = {}
    drop_key = "1"
    
    for node in rules_list:
        check = True
        for k1,k2 in zip(node.keys(), columns.keys()):
            if k1 == class_name:
                output[k1] = node[k1]
                return output
            else:

                if dict[k1] == node[k1]:
                    output[k1] = node[k1]
                else:
                    check = False
                    break
            if check == False:
                break

    return output

def counterfactual_rules(current, expected_change, class_name, target):
    
    output = []    
    for node in get_rules_list():
        for k1, k2 in zip(node.keys(), current.keys()):
            if k1 == class_name:
                if node[k1] == target:
                    try:
                        if current[expected_change] != node[expected_change]:   
                            output.append(node) 
                        
                    except KeyError:
                        continue
                  
    return output

def fit_data(n, X_test, y_test, X_train, y_train, columns, class_v, ohe, le):
    print("Loading........................................")
    explainer_rule = {}
    all_rules = []
    df = pd.DataFrame(columns=['explainer_val'])
    
    for index, row in X_test.iterrows():
        
        global rules_list
        
        rules_list = []
        
        current_row = row.to_frame().T
        
        distance_output = calculate_distance(X_train,current_row)
    
        staging_df = decode_data(ohe, X_train, columns)[1]
        
        staging_df[class_v] = decode_data(le, y_train, [class_v])[1]

        staging_df['distance'] = distance_output
        
        staging_df_final = staging_df.sort_values(by=['distance'], inplace=None)
        
        
        neighbhours = get_neighbhours(n, staging_df_final)
        
        # if num_cols is not None:
        #     neighbhours = get_neighbours_con(neighbhours, num_cols, class_v)

        # fname = file_name + '.json'
        
        dump_neighbhours(neighbhours)
        
        data = load_pickle('neighbour.json')
        
        build_decision_rules(data, class_v)
        
        for key in decode_data(ohe, current_row, columns)[1].columns:
            explainer_rule[key] = decode_data(ohe, current_row, columns)[1][key].values[0]
    
        
        explainer_rule['class'] = 'None'
                
        try:
            df = df.append({'explainer_val': model_predict(explainer_rule, class_v)['class']}, ignore_index = True)
            all_rules.append(model_predict(explainer_rule, class_v))

        except KeyError:
            df = df.append({'explainer_val': "NONE"}, ignore_index = True)
            all_rules.append('NONE')
    print("Execution Completed.")        
    return df, all_rules


def fit_data_for_num(n, X_test, y_test, X_train, y_train, cat_enc, num_val, class_v, ohe, le, cat_cols, num_cols):
    print("Loading........................................")
    explainer_rule = {}
    all_rules = []
    df = pd.DataFrame(columns=['explainer_val'])
    
    for index, row in X_test.iterrows():

        
        global rules_list
        
        rules_list = []
        
        current_row = row.to_frame().T
        
        distance_output = calculate_distance(X_train,current_row)
    
        staging_df = decode_data(ohe, cat_enc, cat_cols)[1]
        
        staging_df[class_v] = decode_data(le, y_train, [class_v])[1]
        
        staging_df = pd.concat([staging_df, num_val], axis = 1, join = 'inner')

        
        staging_df['distance'] = distance_output
        
        staging_df_final = staging_df.sort_values(by=['distance'], inplace=None)
        
        
        neighbhours = get_neighbhours(n, staging_df_final)
        
        # if num_cols is not None:
        neighbhours_out = get_neighbours_con(neighbhours, num_cols, class_v)
        neighbhours = neighbhours_out[0]

        # fname = file_name + '.json'                
        dump_neighbhours(neighbhours)
        
        data = load_pickle('neighbour.json')
                
        build_decision_rules(data, class_v)
        
        num_col_values = current_row[num_cols]
                
        cat_col_values = current_row.drop(num_cols, axis = 1)
        
        for key in decode_data(ohe, cat_col_values, cat_cols)[1].columns:
            explainer_rule[key] = decode_data(ohe, cat_col_values, cat_cols)[1][key].values[0]
    
        for column in num_cols:
            if int(num_col_values[column]) <= int(neighbhours_out[1][column]):
                explainer_rule[column] = '<=' + str(neighbhours_out[1][column])
            else:
                explainer_rule[column] = '>' + str(neighbhours_out[1][column])
        
        explainer_rule['class'] = 'None'
                
        try:
            df = df.append({'explainer_val': model_predict(explainer_rule, class_v)['class']}, ignore_index = True)
            all_rules.append(model_predict(explainer_rule, class_v))
            

        except KeyError:
            df = df.append({'explainer_val': "NONE"}, ignore_index = True)
            all_rules.append('NONE')
    print("Execution Completed.")        
    return df, all_rules