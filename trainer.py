import numpy as np
import cv2
import pickle 
import os
from sklearn import tree
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import *

class TrainData:
    def __init__(self):
        self.train_data = {}

    def read_data(self, filename):
        if not os.path.exists(filename):
            self.train_data = {}
        else:
            with open(filename, 'rb') as f:
                self.train_data =  pickle.load(f)

    def save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.train_data, f)

    def add_data(self, key, value):
        if key not in self.train_data.keys():
            self.train_data[key] = []
        self.train_data[key].append(value)

    def delete_data(self, key):
        self.train_data[key] = self.train_data[key][:-1]

    def delete_key(self, key):
        self.train_data.pop(key, None)

class Train:
    def __init__(self):
        self.X = []
        self.Y = []
        self.labels = []
        self.tree_graph = None

    def generate_train_data(self, train_data):
        
        for key in train_data.keys():
            if SKIP_THUMB:
                x_array = [x[1:, :].flatten() for x in train_data[key]]
            else:
                x_array = [x.flatten() for x in train_data[key]]
            self.X.extend(x_array)
            y_array = [len(self.labels) for y in train_data[key]]
            self.Y.extend(y_array)
            self.labels.append(key)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
            

    def train(self, t):
        if t == "tree":
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(self.X, self.Y)
            self.tree_graph = tree.plot_tree(clf)
            model = {}
            model['clf'] = clf 
            model['graph'] = self.tree_graph
            with open("Models/decision_tree.pkl", 'wb') as f:
                pickle.dump(model, f)
        else:
            clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf'))
            clf = clf.fit(self.X, self.Y)
            model = {}
            model['clf'] = clf 
            with open("Models/svm.pkl", 'wb') as f:
                pickle.dump(model, f)

    def show_graph(self):
        with open("Models/decision_tree.pkl", 'rb') as f:
            saved = pickle.load(f)
        #tree.plot_tree()
        from graphviz import Source
        graph = Source( tree.export_graphviz(saved['clf'], out_file=None))
        png_bytes = graph.pipe(format='png')
        with open('dtree_pipe.png','wb') as f:
            f.write(png_bytes)

    def load_model(self):
        with open("Models/decision_tree.pkl", 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, input):
        if SKIP_THUMB:
            x_array = [x[1:,:].flatten() for x in input]
        else:
            x_array = [x.flatten() for x in input]
        y_array = self.model['clf'].predict(x_array)
        return y_array

class Model:
    def __init__(self, t):
        if t == "tree":
            with open("Models/decision_tree.pkl", 'rb') as f:
                self.model = pickle.load(f)
        else:
            with open("Models/svm.pkl", 'rb') as f:
                self.model = pickle.load(f)


    def predict(self, input):
        if SKIP_THUMB:
            x_array = [x[1:,:].flatten() for x in input]
        else:
            x_array = [x.flatten() for x in input]
        y_array = self.model['clf'].predict(x_array)
        return y_array

    def delete_key(self, key):
        self.model.pop(key, None)


if __name__ == "__main__":

    trainer = Train()
    with open("DATASET/train.pkl", 'rb') as f:
        train_data =  pickle.load(f)
    trainer.generate_train_data(train_data)
    trainer.train('svm')

    #trainer.show_graph()

    # model = Model()
    # model.delete_key('PINCH')


    # data = TrainData()
    # data.read_data("DATASET/train.pkl")
    # print(data.train_data.keys())
    # data.delete_key('PINCH')
    # data.save_data("DATASET/train.pkl")
    # print(data.train_data.keys())