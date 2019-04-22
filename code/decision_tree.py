import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def seperate_split_data(df, label_col=1, test_size=0.4):
    # seperate data base on given label index
    Y = df.iloc[:, label_col]

    col_drop = [1,0,3,8,10,9,5]
    col_names = []
    for i in col_drop:
        col_names.append(df.columns[i])

    X = df.drop(col_names, axis=1)
    # X = df.iloc[:, 4:5]
    # X = df.iloc[:, 4:]
    # Spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


class Node:
    def __init__(self, feature=None, prediction=None, parent=None, type=None, value=None):
        self.parent = parent
        self.children = []
        self.feature = feature
        self.prediction = prediction
        self.type = type
        self.value = value

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def has_child(self):
        return len(self.children) != 0

    def set_type(self, t):
        self.type = t

    def get_prediction(self, feature_dict = None):

        if self.prediction is not None:  # leaf node
            return self.prediction
        if feature_dict is None:
            return get_majority([child.get_prediction() for child in self.children])
        # FIXME: assumption: feature exists in dict
        feature_value = feature_dict[self.feature]
        feature_dict.pop(self.feature, None)

        if self.type == 1:
            # print(self.feature,len(self.children))
            bounds = [child.value for child in self.children]
            for i, b in enumerate(bounds):
                if feature_value < b:
                    child = self.children[i]
                    return self.children[i].get_prediction(feature_dict)

            # fixme: handle this case
            raise Exception(" value not found")

            # a = [child.prediction() for child in self.children]
            # (values, counts) = np.unique(a, return_counts=True)
            # ind = np.argmax(counts)
            # return values[ind]  # the most frequent element

        else:
            values = [child.value for child in self.children]
            for i, v in enumerate(values):
                if feature_value == v:
                    return self.children[i].get_prediction(feature_dict)
            # fixme: handle this case
            # value not found in tree, aka not available when trained
            # solution 1(EZ): return random child's prediction
            # solution 2: return majority prediction of childrens
                # make dict not required, if not provided, call self.predict and return majority of children
            return self.get_prediction()
            # raise Exception(" value not found")

        # def get_child(self, v=None):
        #     if type == 'continuous':
        #         for child in self.children:
        #     #             if (v < child.bound):

        # return None


#
# class NumberNode(Node):
#     def __init__(self, parent=None, bound=0):
#         super().__init__(parent)
#         self.bound = bound
#     def get_child(self, v=None):
#         for child in self.children:
#             if (v < child.bound):
#                 pass
# class StrNode(Node):
#     def __init__(self, feature=None, parent=None):
#         super().__init__(feature, parent)


def get_majority(df2, col_index=0):
    if not isinstance(df2, pd.DataFrame):
        df = pd.DataFrame(df2)
    else:
        df = df2.copy(deep=True)

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

        if isinstance(X[0], (float, int)) and len(set(X)) < 10:
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
        summ = np.sum(np.sum(table.iloc[:, 1:]))
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
        ig = calc_infomation_gain(ouput, inp)
        if ig > ig_max:
            ig_max = ig
            ig_max_i = i
    col_name = df.columns[ig_max_i]
    return col_name, df[col_name], df.drop(col_name, axis=1)


def get_countinous_split_threshold(df, feature_name, label_name):
    def fx(c1, c2, c1_label, c2_label):
        if c1 > c2:
            return c1_label
        else:
            return c2_label

    df_count = df.copy(deep=True)
    df_count['count'] = 1
    table = pd.DataFrame.pivot_table(df_count, values='count', index=feature_name, columns=label_name,
                                     aggfunc=np.sum).fillna(0)

    table[label_name] = np.vectorize(fx)(table[table.columns[0]], table[table.columns[1]], table.columns[0],
                                         table.columns[1])

    flattened = pd.DataFrame(table.to_records())

    df_values = flattened.sort_values(by=feature_name)
    pred_dict = {}
    prev_value, prev_label = None, None
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

        # TODO: put index in param
        label = D.iloc[:, 0]
        label_name = D.columns[0]
        label_set = set(list(label))

        if len(label_set) < 1 or len(D.columns) < 1:
            raise ValueError("build tree error")

        ### if all output from dataset D has same value v
        if len(label_set) == 1:
            # return a signal Node predict v
            v = label_set.pop()
            current_node.prediction = v
            # print("***parant:{},predict:{}***".format(current_node.parent, v))
            # node = Node(prediction=v)
            # parent_node.add_child(node)
            # fixme: debug only
            current_node.feature = v
            return

        ### if all inputs are the same (only 0 feature column left)
        if len(D.columns) == 1 or depth == self.max_depth:
            # return a Node predict majority output
            v = get_majority(D)
            current_node.prediction = v

            # fixme: debug only
            current_node.feature = v
            # node = Node(prediction=v)
            # parent_node.add_child(node)
            # print("***leaf***parant:{},predict:{},value:{}***".format(current_node.parent, v, current_node.value))

            return

        # best_split_feature = attribute with highest information gain
        feature_name, D_split, D_remain = split_by_entropy(D)

        current_node.feature = feature_name

        # get a set [x1,x2...xn] of possible value, size N
        # if instance of int or float:
        # sort the instance
        # loop though the ouput, when it changes label, add medium to the set
        values = set(D_split)

        # print("***parant:{},feature:{},value:{}***".format(current_node.parent, feature_name, current_node.value))
        ### create a Node with N children
        if isinstance(D_split.iloc[0], (int, float)):
            print(feature_name)

            current_node.set_type(1)

            # a dict of {ubound, label}, sorted
            pred_dict = get_countinous_split_threshold(D, feature_name, label_name)
            # Di = D.iloc[:, :]
            Di = D.copy(deep=True)

            for bound, label in pred_dict.items():
                # Di = rows of D where D[x] = x[i]
                node = Node(value=bound)
                current_node.add_child(node)

                # FIXME: <= or < for bound
                D_rest = Di[Di[feature_name] > bound]
                Di = Di[Di[feature_name] <= bound]
                # build tree for each child
                self.build_tree(Di.drop([feature_name], axis=1), node, depth + 1)
                Di = D_rest
            pass
        else:
            current_node.set_type(0)

            # if parent_node is not None:
            #     parent_node.add_child(node)
            for v in values:
                node = Node(value=v)
                current_node.add_child(node)
                # Di = rows of D where D[x] = x[i]
                Di = D[D[feature_name] == v].drop(feature_name, axis=1)
                # build tree for each child
                self.build_tree(Di, node, depth + 1)

    def predict_one(self, x):
        return self.root.get_prediction(x)

        # while True:
        #     # reach leaf
        #     if not node.has_child():
        #         # return prediction
        #         return node.get_prediction()
        #     # has children
        #
        pass

    def predict(self, X):
        names = X.columns

        predictions = []
        for index, row in X.iterrows():
            dictionary = dict(zip(names, row))
            prediction = self.predict_one(dictionary)
            predictions.append(prediction)
        return predictions

    def score(self, X, expectation):
        '''
        Calculate the accuracy of predictor by matching prediction with expectation labels
        :param X: 2d features
        :param expectation: 1d true labels
        :return: accuracy
        '''
        prediction = self.predict(X)
        return len([x for x, y in zip(prediction, expectation) if x == y]) / len(prediction)


def main():
    ### load data ###
    datafile_dir = '../rsrc/titanic.csv'
    data = pd.read_csv(datafile_dir, sep=',', header=0)
    data = data.fillna(data.mean()).fillna("")

    ### train model ###
    # X and Y contain unsplitted data
    X, Y, X_train, X_test, y_train, y_test = seperate_split_data(data, label_col=1, test_size=.4)

    clf = DecisionTree(X_train, y_train, max_depth=10)
    clf.fit()

    # fixme: debug
    # https://github.com/clemtoy/pptree
    import pptree as ppt
    ppt.print_tree(clf.root, childattr='children', nameattr='feature')

    ### test model ###
    clf.predict(X_train)
    print(clf.score(X_train, y_train.values))
    print(clf.score(X_test, y_test.values))


if __name__ == "__main__":
    main()
