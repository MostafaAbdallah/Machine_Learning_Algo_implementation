# - ID3 Algorithm is an algorithm to build an efficient decision tree based on the maximum Gain information
#  for the descriptive features.
# - The Entropy for the data set:
#               H(t, D) = -1 * SUM[ p(t == l) * log2(p(t == l)) ].
# where:   t --->> the target feature
#          D --->> The data set
#          l --->> The values of the target feature
# - To calculate the information gain for each descriptive feature in our dataset we need to calculate the entropy
# remaining after partition the data based on particular descriptive feature d
# rem(d, D) = SUM [   (|D(d == v)| / |D|) *  H(t,  D(d == v)  ]
# where  d --->>  particular descriptive feature
#        v --->> the values of the d
#  after that we can calculate the gain information using this equation
#   IG(d, D) = H(t, D) - rem(d, D);
# after calculate the gain info for all descriptive features, the maximum gain info induct that this descriptive feature
# is the most discriminatory feature and it should be my current node in the decision tree, nad its branches
# is the feature value.
# then we repeat the same process again on the sub dataset
# more information : https://www.python-course.eu/Decision_Trees.php
# and read book Fundamentals of Machine Learning for Predictive Data Analytics, chapter 4, section 2 and 3.


# ========================== implementation ===================================
#   Create class Node which contain :
#       node_value  : Hold the descriptive feature name
#       node_childs : Hold the  branches Name
#       node_next   : To refer to next node

import re
import numpy as np
from typing import Dict
from collections import deque
from graphviz import Digraph
from operator import itemgetter
import ntpath
import enum


class Mearsure(enum.Enum):
    Entropy_Gain_Info = 0
    Entropy_Gain_Info_Ratio = 1
    Gini_Gain_Info = 2

class Node:
    def __init__(self):
        self.node_value = None
        self.node_childs = None
        self.node_next = None


class DataSet:
    def __init__(self, file_name):
        file = open(file_name)
        head, self.data_set_name = ntpath.split(file_name)
        self.data_set_name = re.sub('\.(.*)', '', self.data_set_name)
        print(self.data_set_name)
        self.attributes_names = file.readline().split(',')
        self.attributes_names = self.attributes_names[0: len(self.attributes_names) - 1]
        # print(self.attributes_names)

        self.samples_values = file.readlines()
        self.target_feature = []

        for i in range(len(self.samples_values)):
            self.samples_values[i] = re.sub('\+,', '', self.samples_values[i])
            self.samples_values[i] = self.samples_values[i].strip().split(',')
            for j in range(len(self.samples_values[i])):
                if self.samples_values[i][j].lstrip('-').replace('.','',1).isdigit():
                    self.samples_values[i][j] = float(self.samples_values[i][j])
            self.target_feature.append(self.samples_values[i].pop())

        target_freq = dict()
        for item in self.target_feature:
            if item in target_freq:
                target_freq[item] += 1
            else:
                target_freq[item] = 1
        v = list(target_freq.values())
        k = list(target_freq.keys())
        self.max_target_freq = k[v.index(max(v))]
        self.attributes_ids = [x for x in range(len(self.attributes_names))]
        self.samples_values_ids = [x for x in range(len(self.samples_values))]

        self.attributes_values = dict()
        # print(self.attributes_ids)
        # print(self.samples_values)
        for att_id in range(len(self.attributes_ids)):
            if not self.isContinuousAttribute(self.attributes_names[att_id]):
                for sam_id in range(len(self.samples_values_ids)):
                    if self.attributes_names[att_id] in self.attributes_values:
                        if self.samples_values[sam_id][att_id] not in (
                                self.attributes_values[self.attributes_names[att_id]]):
                            self.attributes_values[self.attributes_names[att_id]].append(
                                self.samples_values[sam_id][att_id])
                    else:
                        self.attributes_values[self.attributes_names[att_id]] = [self.samples_values[sam_id][att_id]]

        # print(self.attributes_values)

    def issamplescontainonetargetval(self, samples_Ids):
        target_val = self.target_feature[samples_Ids[0]]
        for i in range(len(samples_Ids)):
            if target_val != self.target_feature[samples_Ids[i]]:
                return False

        return True

    def isContinuousAttribute(self, attribute_name):
        if attribute_name[len(attribute_name) - 1] == 'D':
            return False

        return True


class DecisionTree:
    def __init__(self, mearsure_type=Mearsure.Entropy_Gain_Info):
        self.root = Node()
        self.impurity_measure = mearsure_type

    def getGini(self, dataset, samples_values_ids, attribute_id=None, attribute_nam=None):
        gini = 0
        sam_count = 0
        target_val = dict()
        for i in samples_values_ids:
            if attribute_id is None or dataset.samples_values[i][attribute_id] == attribute_nam:
                if dataset.target_feature[i] in target_val:
                    target_val[dataset.target_feature[i]] += 1
                else:
                    target_val[dataset.target_feature[i]] = 1
                sam_count += 1

        for t_val in target_val:
            p = target_val[t_val] / sam_count
            gini += (p * p)

        gini = 1 - gini
        return gini

    def getAttributeEntropy(self, dataset, samples_values_ids, attribute_id):
        entropy = 0
        attrbute_name = dataset.attributes_names[attribute_id]
        att_vals_freq = dict()

        for v in dataset.attributes_values[attrbute_name]:
            att_vals_freq[v] = 0

        for i in samples_values_ids:
            att_vals_freq[dataset.samples_values[i][attribute_id]] += 1

        for item in att_vals_freq:
            if att_vals_freq[item] > 0:
                att_vals_freq[item] /= len(samples_values_ids)
                entropy += (att_vals_freq[item] * np.log2(att_vals_freq[item]))
        print(att_vals_freq)
        print(entropy)
        entropy = entropy * -1
        return entropy

    def getEntropy(self, dataset, samples_values_ids, attribute_id=None, attribute_val=None):
        entropy = 0
        sam_count = 0
        target_val = dict()
        for i in samples_values_ids:
            if attribute_id is None or dataset.samples_values[i][attribute_id] == attribute_val:
                if dataset.target_feature[i] in target_val:
                    target_val[dataset.target_feature[i]] += 1
                else:
                    target_val[dataset.target_feature[i]] = 1
                sam_count += 1

        for t_val in target_val:
            p = target_val[t_val] / sam_count
            entropy += (p * np.log2(p))

        entropy = entropy * -1
        return entropy
    def get_rem_discrete_att(self,dataset, samples_values_ids, attributes_id, attrbute_name):
        rem = 0
        # print(dataset.attributes_values[attrbute_name])
        att_vals_freq = dict()

        for v in dataset.attributes_values[attrbute_name]:
            att_vals_freq[v] = 0

        for i in samples_values_ids:
            att_vals_freq[dataset.samples_values[i][attributes_id]] += 1

        for item in att_vals_freq:
            att_vals_freq[item] /= len(samples_values_ids)
        # print(att_vals_freq)

        for item in att_vals_freq:
            if self.impurity_measure == Mearsure.Entropy_Gain_Info or \
                    self.impurity_measure == Mearsure.Entropy_Gain_Info_Ratio:
                rem += att_vals_freq[item] * self.getEntropy(dataset, samples_values_ids, attributes_id, item)
            else:
                rem += att_vals_freq[item] * self.getGini(dataset, samples_values_ids, attributes_id, item)

        return rem
    def calc_rem_continuous_att(self, dataset, samples_values_ids, attributes_id, threshold_val):
        greater = 0
        smaller = 0
        greater_sample_ids = []
        smaller_sample_ids = []
        for i in samples_values_ids:
            if dataset.samples_values[i][attributes_id] >= threshold_val:
                greater += 1
                greater_sample_ids.append(i)
            else:
                smaller +=1
                smaller_sample_ids.append(i)

        greater /= len(samples_values_ids)
        smaller /= len(samples_values_ids)

        if self.impurity_measure == Mearsure.Entropy_Gain_Info or \
                self.impurity_measure == Mearsure.Entropy_Gain_Info_Ratio:
            rem = greater * self.getEntropy(dataset, greater_sample_ids)
            rem += smaller * self.getEntropy(dataset, smaller_sample_ids)
        else:
            rem = greater * self.getGini(dataset, greater_sample_ids)
            rem += smaller * self.getGini(dataset, smaller_sample_ids)

        return rem

    def get_rem_continuous_att(self, dataset, samples_values_ids, attributes_id, attrbute_name):
        tmp_samples_vals= []

        for id in samples_values_ids:
            tmp_samples_vals.append(dataset.samples_values[id])
            tmp_samples_vals[-1].append(dataset.target_feature[id])

        tmp_samples_vals.sort(key=itemgetter(attributes_id))
        adj_pair = []
        print (tmp_samples_vals)
        for i in range(len(tmp_samples_vals) - 1):
            if tmp_samples_vals[i+1][-1] is not tmp_samples_vals[i][-1]:
                p1 = tmp_samples_vals[i+1][attributes_id]
                p2 = tmp_samples_vals[i][attributes_id]
                adj_pair.append((p1 + p2) / 2)

        rem_list = []
        for th in adj_pair:
            tmp_rem = self.calc_rem_continuous_att(dataset, samples_values_ids, attributes_id, th)
            rem_list.append([tmp_rem, th])

        rem = min(rem_list)
        return rem

    def get_rem(self, dataset, samples_values_ids, attributes_id):
        rem = []
        attrbute_name = dataset.attributes_names[attributes_id]
        if not dataset.isContinuousAttribute(attrbute_name):
            rem = [self.get_rem_discrete_att(dataset, samples_values_ids, attributes_id, attrbute_name)]
        else:
            rem = self.get_rem_continuous_att(dataset, samples_values_ids, attributes_id, attrbute_name)

        return rem

    def get_best_att_GI(self, dataset, samples_values_ids, attributes_ids):
        measure = 0
        if self.impurity_measure == Mearsure.Entropy_Gain_Info or \
                self.impurity_measure == Mearsure.Entropy_Gain_Info_Ratio:
            measure = self.getEntropy(dataset, samples_values_ids)
        else:
            measure = self.getGini(dataset, samples_values_ids)

        # print(entropy)
        gain_info = [0] * len(dataset.attributes_ids)
        rem = [0] * len(dataset.attributes_ids)
        # print("gain info =")
        for att in attributes_ids:
            rem[att] = self.get_rem(dataset, samples_values_ids, att)
            gain_info[att] = (measure - rem[att][0])
            if self.impurity_measure == Mearsure.Entropy_Gain_Info_Ratio:
                gain_info[att] /= self.getAttributeEntropy(dataset, samples_values_ids, att)

        best_att_id = gain_info.index(max(gain_info))
        threshold = None
        print(rem[best_att_id])
        print(len(rem))
        if isinstance(rem[best_att_id], list) and len(rem[best_att_id]) > 1:
            threshold = rem[best_att_id][1]
        return dataset.attributes_names[best_att_id], best_att_id, threshold

    def __id3_rec(self, dataset, samples_values_ids, attributes_ids, root):
        _attributes_ids = attributes_ids[:]
        root = Node()
        # if the current data partition contain only one target feature, then it is a leaf node "no branches"
        if dataset.issamplescontainonetargetval(samples_values_ids):
            root.node_value = dataset.target_feature[samples_values_ids[0]]
            return root
        # check if there is no other attribute to be add in our decision tree
        if len(_attributes_ids) == 0:
            root.node_value = dataset.max_target_freq
            return root
        # Get the maximum gain info which induct that this descriptive feature
        bestAttributNam, bestAttrinutId, threshold = self.get_best_att_GI(dataset, samples_values_ids, _attributes_ids)

        root.node_value = bestAttributNam
        root.node_childs = []
        print(bestAttributNam)

        if threshold is None:
            print(dataset.attributes_values[bestAttributNam])
            attributes_vals = dataset.attributes_values[bestAttributNam]
        else:
            attributes_vals = [('<' + str(threshold)), ('>=' + str(threshold))]
            print(attributes_vals)
        for item in attributes_vals:
            child = Node()
            child.node_value = item
            root.node_childs.append(child)

            childsamplesid = []
            for id in samples_values_ids:
                if threshold is None:
                    if dataset.samples_values[id][bestAttrinutId] == item:
                        childsamplesid.append(id)
                else:
                    if '>=' in item:
                        if dataset.samples_values[id][bestAttrinutId] >= threshold:
                            childsamplesid.append(id)
                    else:
                        if dataset.samples_values[id][bestAttrinutId] < threshold:
                            childsamplesid.append(id)

            if len(childsamplesid) == 0:
                child.node_next = Node()
                child.node_next.node_value = dataset.max_target_freq
            else:
                if len(_attributes_ids) > 0 and bestAttrinutId in _attributes_ids and threshold is None:
                    to_be_remove = _attributes_ids.index(bestAttrinutId)
                    _attributes_ids.pop(to_be_remove)
                child.node_next = self.__id3_rec(dataset, childsamplesid, _attributes_ids, child.node_next)
        return root

    def id3(self, dataset):
        self.root = self.__id3_rec(dataset, dataset.samples_values_ids, dataset.attributes_ids, self.root)

    def printTree(self):
        print('Decision Tree :')
        if self.root:
            tree = deque()
            tree.append(self.root)
            while len(tree) > 0:
                node = tree.popleft()
                print(node.node_value)
                if node.node_childs:
                    for n in node.node_childs:
                        print('({})'.format(n.node_value))
                        tree.append(n.node_next)
                elif node.node_next:
                    print(node.node_next)

    def drawTree(self, database_name):
        g = Digraph('Decision Tree of ' + database_name, filename=database_name + '.gv')
        g.attr(rankdir='TB', size='8,5')

        g.attr('node', shape='ellipse')
        if self.root:
            tree = deque()
            tree.append(self.root)
            while len(tree) > 0:
                node = tree.popleft()
                g.node(node.node_value)
                if node.node_childs:
                    for n in node.node_childs:
                        g.edge(str(node.node_value), str(n.node_next.node_value), label=str(n.node_value))
                        tree.append(n.node_next)
                elif node.node_next:
                    g.node(node.node_next)
        g.view()

    def drawTreeDFS(self, database_name):
        g = Digraph('Decision Tree of ' + database_name, filename=database_name + '.gv')
        g.attr(rankdir='TB', size='8,5')

        g.attr('node', shape='ellipse')

        self.drawTreeRec(self.root, g,0)
        g.view()

    def drawTreeRec(self, root, graph, node_num):
        if not root:
            return node_num
        cur_node = node_num
        if root.node_childs:
            graph.node(str(cur_node), str(root.node_value))
            for n in root.node_childs:
                if not n.node_next.node_childs:
                    graph.node(str(n.node_next.node_value), shape='square')
                    graph.edge(str(cur_node), str(n.node_next.node_value), label=str(n.node_value))
                else:
                    graph.node(str(node_num+1), str(n.node_next.node_value))
                    graph.edge(str(cur_node), str(node_num+1), label=str(n.node_value))
                node_num = self.drawTreeRec(n.node_next, graph, node_num+1)
        # elif root.node_next:
        #     graph.node(str(node_num), str(root.node_next), shape='square')

        return node_num+1

def Test(file_path):
    dataSet = DataSet(file_path)
    # decision_tree = DecisionTree(Mearsure.Entropy_Gain_Info_Ratio)
    # decision_tree = DecisionTree(Mearsure.Gini_Gain_Info)
    decision_tree = DecisionTree()
    decision_tree.id3(dataSet)
    decision_tree.printTree()
    decision_tree.drawTreeDFS(dataSet.data_set_name)


if __name__ == '__main__':
    # Test('./Dataset/Playtennis.csv')
    # Test('./Dataset/vegetation.csv')
    Test('./Dataset/vegetation_Cont.csv')
