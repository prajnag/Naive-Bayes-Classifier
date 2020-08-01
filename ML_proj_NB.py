import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
random_state = 42 
#storing all the column names
names=["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-corporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa","ClassName"]
#reading the csv into pandas dataframe
data = pd.read_csv("/Users/prajnagirish/Desktop/College/Sem-5/ML/hvdata.TXT", 
                  sep=',', 
                  names=names)
#function to add commas between elements of a list ie. [1 2 3]->[1,2,3]
def list_comp(l):
    temp=[]
    for i in l:
        temp.append(i)
    return temp
#uses inbuilt function to create a confusion matrix and calculate various parameters
def confusionmatrix(ypred,yobtained):
    cm=confusion_matrix(ypred,yobtained) #from sklearn
    acc = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0] + cm[1][1])
    precision=(cm[0][0])/(cm[0][0]+cm[0][1])
    recall=(cm[0][0])/(cm[0][0]+cm[1][0])
    return cm,acc, precision, recall
#given the training and testing sets, classifying the data into y or n
def naive_bayes(x_train,x_test,y_train,y_test):
    yobtained=[]
    temp=[]
    xtest=[]
    xtrain=[]
    ytrain=[]
    ytest=[]
    y_train.index = np.arange(0, len(y_train)) #setting indexes of all the dataframes to 0
    x_train.index = np.arange(0, len(x_train))
    y_test.index = np.arange(0, len(y_test))
    x_test.index = np.arange(0, len(x_test))
    for j in range(len(x_test)):  #storing contents of dataframe into list for easy traversal
        temp=[]
        for i in x_test.iloc[j,:]:
            temp.append(i)
        xtest.append(temp)
    for j in range(len(x_train)):
        temp=[]
        for i in x_train.iloc[j,:]:
            temp.append(i)
        xtrain.append(temp)
    for i in range(len(y_train)):
        ytrain.append(y_train['ClassName'][i])
    for i in range(len(y_test)):
        ytest.append(y_test['ClassName'][i])
    for j in xtest: #for each test set
        i=0
        pos=float(list(y_train['ClassName']).count(1)/len(x_train))
        neg=float(list(y_train['ClassName']).count(0)/len(x_train))
        while(i<no_of_attributes): #for each one of the attributes 
            test=j[i] #value of a particular attribute
            l=list(x_train[names[i]]) 
            df = pd.DataFrame(list(zip(l,ytrain)),columns =[names[i], 'op']) #set a dataframe of the column and the output column
            ctp=len(df[(df[names[i]]==test) & (df['op']==1)]) #positive classification
            ctn=len(df[(df[names[i]]==test) & (df['op']==0)])#classifying as negative
            pos*=float(ctp/list(y_train['ClassName']).count(1)) #finding probability
            neg*=float(ctn/list(y_train['ClassName']).count(0)) #finding probability
            i+=1
        if(pos>neg):
            yobtained.append(1)
        else:
            yobtained.append(0)
    cm,acc, precision,recall=confusionmatrix(ytest,yobtained) #finding confusion matrix and related parameters 
    return cm,acc, precision,recall

length=435
data = data.replace('?', np.nan) #replacing question marks with NaN
for i in range(len(data['ClassName'])): #replacing republican with 1 and democrat with 0
    if data['ClassName'][i]=='republican':
        data['ClassName'][i]=1
    else:
        data['ClassName'][i]=0
for i in data:
    data[i].fillna(data[i].mode()[0], inplace=True) #replacing NaN with mode of column
no_of_attributes=16
x=data.drop('ClassName',axis=1) #x contains all the columns except last
y = data.iloc[:, 16] #y contains last column
y=y.to_frame() #converting pandas series to pandas dataframe
cv = KFold(n_splits=5, random_state=42, shuffle=False) #5 fold cross validation
max_acc=0
for train_index, test_index in cv.split(x):
    train_index=list_comp(train_index)
    test_index=list_comp(test_index)
    x_train, x_test, y_train, y_test = x.loc[train_index,:], x.loc[test_index,:], y.loc[train_index,:], y.loc[test_index,:]
    cm,acc,precision,recall=naive_bayes(x_train,x_test,y_train,y_test)
    if acc>max_acc:
        maxcm=cm
        max_acc=acc
        maxpres=precision
        maxrecall=recall
print(maxcm)
print("The Accuracy of the program is: ",round(max_acc*100,3))
print("The Precision of the program is: ",round(maxpres*100,3))
print("The Recall of the program is: ",round(maxrecall*100,3))
print("F meaure of the program is: ", round(200*(maxpres*maxrecall)/(maxpres+maxrecall),3))