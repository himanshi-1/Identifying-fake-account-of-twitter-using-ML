
### Detect fake profiles in online social networks using Hybrid approach


# In[57]:
##import system
#import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler    
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow.keras
from keras.models import Sequential #used to initialize the NN
from keras.layers import Dense  #used to build the hidden Layers
from keras.layers import Dropout

import time
start_time = time.time()


####### function for reading dataset from csv files

# In[58]:

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("users.csv")
    fake_users = pd.read_csv("fusers.csv")
    x=pd.concat([genuine_users,fake_users])  
    #print (x.columns)
    y= len(fake_users)*[0] + len(genuine_users)*[1]   
    return x,y
    

####### function for feature engineering

# In[62]:

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x


####### ####plot
# In[63]:

def plotcount(uv,title1):
    labels = ['Genuine','Fake']
    classes = pd.value_counts(uv, sort = True)
    classes.plot(kind = 'bar', rot=0)
    plt.title(title1)
    plt.xticks(range(2), labels)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()


####### function for plotting confusion matrix

# In[65]:

def plot_confusion_matrix(cm):
    LABELS = [ 'Genuine','Fake']
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

####### function for plotting ROC curve

# In[71]:

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,color='orange',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],linestyle='--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


####### Function for training data using Hybrid Classifier

# In[72]:

def train(X_train,y_train,X_test):
    """ Trains and predicts dataset with a Voting classifier """
    # Scaling features
    ##X_train = X_train.fillna(X_train.mean())
    ##X_test = X_test.fillna(X_test.mean())
    st_x= StandardScaler()    
    X_train= st_x.fit_transform(X_train)    
    X_test= st_x.transform(X_test)  
    estimator=[]
    estimator.append(('SVC', SVC(probability=True))) 
    ##estimator.append(('RF', RandomForestClassifier()))  
    ##estimator.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
    ##estimator.append(('DTC', DecisionTreeClassifier())) 
    estimator.append(('LR',LogisticRegression(solver ='lbfgs',  multi_class ='binomial', max_iter = 200))) 
        
    vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
    vot_soft.fit(X_train, y_train) 
    scores = model_selection.cross_val_score(vot_soft, X_train,y_train, cv=5)
    print (scores)
    print('Estimated score: %0.5f ' % (scores.mean()))
    # Predict class
    y_pred = vot_soft.predict(X_test)
    return y_test,y_pred




def traini(X_train,y_train,X_test):
    """ Trains and predicts dataset with a SVM classifier """
    # Scaling features
    X_train=preprocessing.scale(X_train)
    X_test=preprocessing.scale(X_test)
    classifier = Sequential()

    # Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dropout(rate=0.1))

    # Adding the second hidden layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
        
    #y_pred = (y_pred > 0.5)
    _, accuracy = classifier.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))
    y_pred = classifier.predict(X_test)
    # Predict class
  #  y_pred = clf.best_estimator_.predict(X_test.reshape(1,-1))
    
    print('Estimated score: %0.5f ' % (accuracy.mean()))
    return y_test,y_pred
# In[76]:

print ("reading datasets.....\n")
x,y=read_datasets()
plotcount(y,"Class distribution of actual dataset")


# In[77]:

print ("extracting features.....\n")
x=extract_features(x)
print (x.columns)
print (x.describe())


# In[78]:

print ("spliting datasets in train and test dataset...\n")
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)
plotcount(y_train,"Class distribution of training dataset=")
plotcount(y_test,"Class distribution of test dataset")


# In[79]:

print ("training datasets.......\n")
y_test,y_pred = train(X_train,y_train,X_test)



print ("spliting datasets in train and test dataset...\n")


X_train,X_test,y_train,y_test = train_test_split(X_test, y_pred, test_size=0.20, random_state=44)

print ("training datasets.......\n")
y_test,y_pred = traini(X_train,y_train,X_test)
# In[80]:




# In[81]:

cm=confusion_matrix(y_test, y_pred.round())
print('Confusion matrix')
print(cm)
plot_confusion_matrix(cm)



# In[82]:

print(classification_report(y_test, y_pred.round(), target_names=['Fake','Genuine']))



# In[83]:

plot_roc_curve(y_test, y_pred.round())
print("Process finished --- %s seconds ---" % (time.time() - start_time))


