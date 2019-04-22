import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def seperate_split_data(df, label_col=1, test_size=0.4):
    # seperate data base on given label index

    Y = df.iloc[:, label_col]
    # X = df.drop(df.columns[1,], axis=1)
    X = df.iloc[:,4:]
    # Spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


class Node:
    def __init__(self, feature = None, prediction=None,parent = None):
        self.parent = parent
        self.children = []
        self.feature = feature
        self.prediction = prediction

    def add_child(self, node):
        self.children.append(node)


class NumberNode(Node):
    def __init__(self,parent = None, bound=0):
        super().__init__(parent)
        self.bound = bound


class StrNode(Node):
    def __init__(self,feature = None,parent = None):
        super().__init__(feature,parent)


def get_majority(df, col_index=0):
    x = df.columns[col_index]
    df['count'] = 1
    df_counts = df.groupby(df.columns[col_index]).count()
    df_v = df_counts.idxmax(axis=0, skipna=True)
    v = df_v.values[0]
    return v


def calc_entropy(occurance_list):
    # formula from: https://www.saedsayad.com/decision_tree.htm
    #     df = pd.DataFrame({1: occurance_list})
    #     df2 =df.groupby(df.columns[0]).count()
    #     count_arr = df2.values
    summ = np.sum(occurance_list)
    T = [x / summ for x in occurance_list]
    s = [-x * math.log2(x) if x != 0 else 0 for x in T]
    return np.sum(s)


def calc_entropy_given_arrays(T, X=None, ):
    # formula from: https://www.saedsayad.com/decision_tree.htm
    # T = label, always discrete
    # X = feature, might be continous
    # input are arrays
    if X is None:
        df = pd.DataFrame({'label': T})
        df['count'] = 1
        df2 = df.groupby(df.columns[0]).count()
        count_arr = df2.values
        summ = np.sum(count_arr)
        T = [x / summ for x in count_arr]
        s = [-x * math.log2(x) if x != 0 else 0 for x in T]
        return np.sum(s)
    else:
        label_name = 'label'
        feature_name = 'feature'
        df = pd.DataFrame({label_name: T, feature_name: X})

        if isinstance(X[0], (float,int)) and len(set(X))<10:
            dic = get_countinous_split_threshold(df, feature_name, label_name)
            thresholds = list(dic.keys())

            def indexing(x):
                for i, v in enumerate(thresholds):
                    if x <= v:
                        return i

            df['feature'] = df[feature_name].apply(indexing)
        #             print(df)
        # build occurance table

        df['count'] = 1
        table = pd.DataFrame.pivot_table(df, values='count', index=[feature_name], columns=[label_name],
                                         aggfunc=np.sum).fillna(0)
        table = pd.DataFrame(table.to_records())
        summ = np.sum(np.sum(table.iloc[:,1:]))
        dfc = table

        def fx(x, y):
            return (x + y) / summ * calc_entropy([x, y])

        dfc['entropy'] = np.vectorize(fx)(dfc[dfc.columns[1]], dfc[dfc.columns[2]])
        return np.sum(dfc['entropy'])


def calc_infomation_gain(T, X):
    #     print(T)
    #     return  calc_entropy_given_arrays(T,X)
    return calc_entropy_given_arrays(T) - calc_entropy_given_arrays(T, X)


def split_by_entropy(df):
    # find attribute with highest information gain
    ouput = df.iloc[:, 0].values
    ig_max = 0
    ig_max_i = 1
    # for col in df.columns[1:]:
    for i in range(1, len(df.columns)):
        inp = df.iloc[:, i].values
        ig = calc_infomation_gain(ouput,inp)
        if ig>ig_max:
            ig_max = ig
            ig_max_i = i
    col_name = df.columns[ig_max_i]
    return col_name,df[col_name],df.drop(col_name,axis=1)


def get_countinous_split_threshold(df, feature_name, label_name):
    def fx(c1, c2, c1_label, c2_label):
        if c1 > c2:
            return c1_label
        else:
            return c2_label

    df_count = df.iloc[:, :]
    df_count['count'] = 1
    table = pd.DataFrame.pivot_table(df_count, values='count', index=feature_name, columns=label_name,
                                     aggfunc=np.sum).fillna(0)

    table[label_name] = np.vectorize(fx)(table[table.columns[0]], table[table.columns[1]], table.columns[0],
                                         table.columns[1])

    flattened = pd.DataFrame(table.to_records())

    df_values = flattened.sort_values(by=feature_name)
    pred_dict = {}
    prev_value,prev_label = None, None
    print(df_values)
    for index, row in df_values.iterrows():

        if index == 0:
            prev_value = row[feature_name]
            prev_label = row[label_name]
        else:
            curr_label = row[label_name]
            curr_value = row[feature_name]
            if curr_label != prev_label:
                # print(curr_value,prev_value)
                ubound = (curr_value + prev_value) / 2
                pred_dict[ubound] = prev_label
            prev_value = curr_value
            prev_label = curr_label
    pred_dict[float("inf")] = prev_label
    return pred_dict


class DecisionTree:
    def __init__(self, X, y, max_depth=10):
        self.root = None
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.D = pd.concat([y, X], axis=1)

    def fit(self):
        self.root = Node()
        self.build_tree(self.D, self.root, 0)

        pass

    def build_tree(self, D, current_node, depth):
        output = D.iloc[:, 0]
        label_name = D.columns[0]
        output_set = set(list(output))

        if len(output_set) < 1 or len(D.columns) < 1:
            raise ValueError("build tree error")

        # if all output from dataset D has same value v
        if len(output_set) == 1:
            # return a signal Node predict v

            v = output_set.pop()
            current_node.prediction = v
            # node = Node(prediction=v)
            # parent_node.add_child(node)
            return

        # if all inputs are the same (only 0 feature column left)
        if len(D.columns) == 1 or depth == self.max_depth:
            # return a Node predict majority output
            v = get_majority(D)
            current_node.prediction = v
            # node = Node(prediction=v)
            # parent_node.add_child(node)
            return

        # best_split_feature = attribute with highest information gain
        feature_name, D_split, D_remain = split_by_entropy(D)
        current_node.feature = feature_name

        # get a set [x1,x2...xn] of possible value, size N
        # if instance of int or float:
        # sort the instance
        # loop though the ouput, when it changes label, add medium to the set
        values = set(D_split)

        # create a Node with N children
        if isinstance(D_split.iloc[0], (int, float)):

            # a dict of {ubound, label},sorted
            pred_dict = get_countinous_split_threshold(D, feature_name, label_name)

            Di = D.iloc[:, :]
            for bound, label in pred_dict.items():
                # Di = rows of D where D[x] = x[i]
                node = NumberNode(parent=current_node, bound=bound)
                current_node.add_child(node)

                # FIXME: <= or < for bound
                print(feature_name)
                print(Di.columns[2]==feature_name)
                Di = Di[Di[feature_name] <= bound]

                # build tree for each child
                self.build_tree(Di.drop([feature_name], axis=1), node, depth+1)
            # TODO:
            pass
        else:
            node = StrNode(parent=current_node)
            # if parent_node is not None:
            #     parent_node.add_child(node)
            for v in values:
                # Di = rows of D where D[x] = x[i]
                Di = D[D[feature_name] == v].drop(feature_name,axis=1)
                # build tree for each child
                self.build_tree(Di, node,depth+1)


def main():
    datafile_dir = '../rsrc/titanic.csv'
    data = pd.read_csv(datafile_dir, sep=',', header=0)
    data = data.fillna(data.mean()).fillna("")
    print(data.mean())

    # X and Y contain unsplitted data
    X, Y, X_train, X_test, y_train, y_test = seperate_split_data(data, label_col=1, test_size=.4)

    clf = DecisionTree(X_train, y_train, max_depth=10)
    clf.fit()

main()

if __name__ == "__main__":
    main()
