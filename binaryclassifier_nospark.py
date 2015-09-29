

"""
Gradient boosted Trees classification and regression using MLlib.


Clean this file and upload again


Additions
@author: Johannes Bauer

28 July 2015

1 Load file (path; csv or libsvm)
2 Do random split into test and training set
3 Compute classifier (logistic regression, tree, random forest (RF), gradient boosted trees (GBT))
  a) use python scikit learn library
  b) use spark MLLib
4 Compute accuracy (Training error, test error, F1 score)
5 Write results to file

Note difference in input formats:
-scikit learn takes numpy array input 
-mllib takes rdd as list of labeled point format


To do:

- as input make csv file possible; This should be OK
- produce csv for voter data (turnoutmodel)
- add runtime timer
- clarify how to make hyper parameters the same in all cases (not clear)
- run more systematic tests
- change local spark parameters
- find more suitable datasets
- why is mllib logistic regression so bad?, no idea, could not really improve it....


"""

import numpy as np
import pandas as pd
import os
import sys
import json
import csv
import datetime


from operator import add

from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import sklearn.datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# Global variable
metrics_dict={}
timing_dict={}

def npmat_to_rdd_wreadwrite(sc,X,Y,f_name,delete_file=False):
    """
    Takes a data prepared for scikit model X in numpy matrix format, Y one-dimensional numpy array 
    and writes to file in libsvm format with filename string f_name provided (could delete automatically), 
    then reads from file directly into spark RDD object (for given Sparkcontext sc)

    """
    sklearn.datasets.dump_svmlight_file(X,Y,f_name,zero_based=False)
    read_rdd= MLUtils.loadLibSVMFile(sc, f_name)
    if delete_file:
      os.remove(f_name)
    return read_rdd


def scikit_model_accuracy(str_model_name,model,X_train,Y_train,X_test,Y_test,print_screen=True):

    Y_train_pred= model.predict(X_train)
    trainingErr = sum(Y_train != Y_train_pred)/float(len(Y_train))
    Y_test_pred= model.predict(X_test)
    testErr = sum(Y_test != Y_test_pred)/float(len(Y_test))
    F1_score=sm.f1_score(Y_test,Y_test_pred)

    metrics_dict[str_model_name+'_scikit_trainErr']=trainingErr
    metrics_dict[str_model_name+'_scikit_testErr']=testErr
    metrics_dict[str_model_name+'_scikit_F1score']=F1_score

    if print_screen:
      print str_model_name+" scikit training error: "+ str(trainingErr)
      print str_model_name+" scikit test error : "+ str(testErr)
      #print str_model_name+" scikit test F1 score : "+ str(my_F1score(Y_test,Y_test_pred))
      print str_model_name+" scikit test F1 score : "+ str(F1_score)


def mllib_model_accuracy(str_model_name,model,train_rdd,test_rdd,print_screen=True):

    Y_train_pred_rdd = model.predict(train_rdd.map(lambda x: x.features))
    Y_train_labelandpred_rdd = train_rdd.map(lambda lp: lp.label).zip(Y_train_pred_rdd)
    trainingErr = Y_train_labelandpred_rdd.filter(lambda (v, p): v != p).count() / float(train_rdd.count())
    Y_test_pred_rdd = model.predict(test_rdd.map(lambda x: x.features))
    Y_test_labelandpred_rdd = test_rdd.map(lambda lp: lp.label).zip(Y_test_pred_rdd)
    testErr = Y_test_labelandpred_rdd.filter(lambda (v, p): v != p).count() / float(test_rdd.count())
    F1_score=sm.f1_score(test_rdd.map(lambda lp: lp.label).collect(),Y_test_pred_rdd.collect())

    metrics_dict[str_model_name+'_mllib_trainErr']=trainingErr
    metrics_dict[str_model_name+'_mllib_testErr']=testErr
    metrics_dict[str_model_name+'_mllib_F1score']=F1_score

    if print_screen:
      print str_model_name+' MLLib training Error = ' + str(trainingErr)
      print str_model_name+ ' MLLib test error = ' + str(testErr)
      print str_model_name+" MLLib test F1 score : "+ str(F1_score)

def runtime_write(str_model_name,t_start,print_screen=True):
    delta_t=datetime.datetime.now()-t_start
    timing_dict[str_model_name]=delta_t
    if print_screen:
      print str_model_name+' time taken: ' + str(delta_t)


def my_F1score(Y_test,Y_test_pred):
    # Results for confusion matrix, True Positive etc (test set)
    NP=0
    NTP=0
    NFP=0
    # Assume binary 0,1 label
    for k in xrange(len(Y_test)):
      if Y_test[k]==1:
        NP+=1
        if Y_test_pred[k]==1:
          NTP+=1
      else:
        if Y_test_pred[k]==1:
          NFP+=1
    
    if NTP+NFP !=0:
      precision = NTP/float(NTP+NFP)
    else:
      print "Problem when computing precision."
      precision=1
    if NP!=0:
      recall = NTP/float(NP)
    else:
      print "Problem when computing recall."
      recall=1
    F1_score=2*precision*recall/(precision+recall)
    return F1_score





    ###############################################################################
    ###############################################################################
    ###############################################################################


        




if __name__ == "__main__":

    ###############################################################################
    # Input parameters
    ###############################################################################
    # For npmat_to_rdd function
    temp_test_filename='temptest.txt'
    temp_train_filename='temptrain.txt'
    delete_temp_file=True

    input_data_path='data/mllib/sample_libsvm_data.txt'
#    input_data_path='data/mllib/svmguide1.t'
#    input_data_path='data/mllib/spambase.csv'

    dataformat_libsvm=True
    dataformat_csv=False



    # Assume that classification label is the last entry
   
    test_size=0.3

    run_scikit = True
    #run_spark_mllib = True

    # Output
    write_dict2file = False
    format_json = False
    format_csv = True
    # could be more specific here, possibly include timestamp
    outfilename='results'

    ###############################################################################
    # Read X, Y for classification problem
    if dataformat_libsvm:
      X_temp, Y = load_svmlight_file(input_data_path)
      X=X_temp.toarray()

    if dataformat_csv:
      X_temp=np.loadtxt(open(input_data_path,"rb"),delimiter=",",skiprows=0)
      # Assume that classification label is the last entry
      ncolm1=X_temp.shape[1]-1
      X=X_temp[:,:ncolm1]
      Y=X_temp[:,ncolm1]

    if dataformat_libsvm or dataformat_csv:
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)




    

    if run_scikit:
      ###############################################################################
      # Scikit learn models
      ###############################################################################
      # Simple tree
      t0=datetime.datetime.now()
      model_scikit_tree=tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

      model_scikit_tree.fit(X_train,Y_train)

      scikit_model_accuracy('Tree',model_scikit_tree,X_train,Y_train,X_test,Y_test)

      runtime_write('Tree_scikit',t0)
      ###############################################################################
      # Random forest
      t0=datetime.datetime.now()
      model_scikit_RFT=RandomForestClassifier(max_depth=5,n_estimators=10,
            criterion='gini')

      model_scikit_RFT.fit(X_train,Y_train)

      scikit_model_accuracy('RF',model_scikit_RFT,X_train,Y_train,X_test,Y_test)

      runtime_write('RF_scikit',t0)
      ###############################################################################
      # Gradient boosted trees
      t0=datetime.datetime.now()
      model_scikit_GBT=GradientBoostingClassifier(
            loss='deviance', learning_rate=0.1, n_estimators=10,
            subsample=1, min_samples_split=5, min_samples_leaf=5, max_depth=5,
            init=None, random_state=1, verbose=0,
            max_leaf_nodes=None, warm_start=False)

#      model_scikit_GBT=GradientBoostingClassifier(
#            loss='deviance', learning_rate=0.1, n_estimators=10,
#            max_depth=5)

      model_scikit_GBT.fit(X_train,Y_train)

      scikit_model_accuracy('GBT',model_scikit_GBT,X_train,Y_train,X_test,Y_test)

      runtime_write('GBT_scikit',t0)
      ###############################################################################
      # Logistic regression
      t0=datetime.datetime.now()
      #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
      #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0)

      #model_scikit_logreg=LogisticRegression(penalty='l2', tol=0.0001, C=1.0, solver='liblinear', max_iter=100, multi_class='ovr')
      model_scikit_logreg=LogisticRegression(penalty='l2', tol=0.0001, C=1.0, solver='lbfgs', max_iter=100, multi_class='ovr')

      model_scikit_logreg.fit(X_train,Y_train)

      scikit_model_accuracy('logreg',model_scikit_logreg,X_train,Y_train,X_test,Y_test)

      runtime_write('logreg_scikit',t0)

      ###############################################################################
      # Support vector machines (SVM)
      t0=datetime.datetime.now()
      #model_scikit_SVM =svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
      #                      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      #                      shrinking=True, tol=0.001, verbose=False)

      #model_scikit_SVM.fit(X_train,Y_train)

      #scikit_model_accuracy('SVM',model_scikit_SVM,X_train,Y_train,X_test,Y_test)

      runtime_write('SVM_scikit',t0)

    ###############################################################################


    print metrics_dict
    print timing_dict
    if write_dict2file:
      if format_json:
        f_name=outfilename+'.json'
        with open(f_name,'w') as f:
          f.write(json.dumps(metrics_dict))
      if format_csv:
        f_name=outfilename+'.csv'
        writer=csv.writer(open(f_name,'wb'),delimiter=':')
        for key in sorted(metrics_dict.keys()):
          writer.writerow([key, metrics_dict[key]])
        f_name='timing'+'.csv'
        writer=csv.writer(open(f_name,'wb'),delimiter=':')
        for key in sorted(timing_dict.keys()):
          writer.writerow([key, timing_dict[key]])
        #with open(f_name,'w') as f:
        #  for key in sorted(metrics_dict.keys()):
        #    f.write(key+', '+ str(metrics_dict[key])+'\n')

