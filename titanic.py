import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials


class RFClassifier(RandomForestClassifier):
	def __init__(self,train=None,test=None,ensemble=10,nodes=20):
		super().__init__(n_estimators=ensemble,max_leaf_nodes=nodes)
		self.train=train
		self.test=test
		self.accuracy=None
		self.fit_object=None

	def fit_data(self):
		self.fit_object=super().fit(self.train[["Age","Gender","Class"]],self.train["Survived"])

	def accuracy_test(self):
		result=self.fit_object.predict(self.test[["Age","Gender","Class"]])
		self.accuracy=accuracy_score(self.test["Survived"],result)

remove=lambda s:s[0]

def gender(name):
    firstname=name[name.index(',')+2:]
    sal=firstname.split(' ')[0].lower()
    if sal=='mrs' or sal=='miss':
        return 'Female'
    else:
        return 'Male'

def cattonum(name):
	series=name.astype('category')
	return series.cat.codes

def write_to_file(clf):
	with open("Titanic.dot","w") as f:
		f=tree.export_graphviz(clf,
			feature_names=['Class','Age','Gender'],out_file=f)

def build_and_test_tree(test,train,nodes):
	clf=DecisionTreeClassifier(max_leaf_nodes=nodes)
	clf=clf.fit(train[["Class","Age","Gender"]],train['Survived'])
	predictions=clf.predict(test[["Class","Age","Gender"]])
	accuracy=accuracy_score(test["Survived"],predictions)
	return accuracy

def preprocess_data():
	df=pd.read_csv("Titanic.csv")
	df=df[df.Age.notnull()][['Name','Class','Age','Survived']]
	df['Age']=df['Age'].map(int)
	df['Class']=df['Class'].apply(remove)
	df['Survived']=df['Survived'].map(int)
	df['Gender']=df['Name'].apply(gender)
	s=df.groupby(['Class'])['Survived']
	df['Gender']=df[['Gender']].apply(cattonum)
	train,test=train_test_split(df,test_size=0.2)
	return train,test

def xgbclassifier(train,test):
	xgb=XGBClassifier()
	xgb=xgb.fit(train[["Age","Class","Gender"]],train["Survived"])
	result=xgb.predict(test[["Age","Class","Gender"]])
	accuracy=accuracy_score(result,test["Survived"])
	return accuracy

train,test=preprocess_data()
RandForestClassifierObject=RFClassifier(train,test,ensemble=1000,nodes=15)
RandForestClassifierObject.fit_data()
RandForestClassifierObject.accuracy_test()
print(RandForestClassifierObject.accuracy)

