
"""

@author: Johannes Bauer
First version: 28 July 2015

Run with
> <spark-directory> /bin/spark-submit binaryclassifier.py
replace <spark-directory> with path to your spark installation

Test scikit learn vs Spark Machine Learning libraries (MLLib): accuracy and timing

Testing and comparing different classifiers
with some input data in libsvm or csv format

function mllib_accuracy is taken from example script in Apache spark distribution

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


"""

import numpy as np
import pandas as pd
import os
import sys
import json
import csv
import datetime

from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import LogisticRegressionWithSGD,LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD, SVMModel

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
modelparameter_dict={}
# entries are dictionaries of parameters for each model

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

    # possible input data files, either libsvm or csv format
    #input_data_path='data/mllib/sample_libsvm_data.txt'
    input_data_path='data/mllib/svmguide1.t'
    #input_data_path='data/mllib/spambase.csv'

    dataformat_libsvm=True
    dataformat_csv=False


    # In csv format, assume that classification label is the last entry
   
    test_size=0.3

    run_scikit = True
    run_spark_mllib = True


    # List of models used to run
    model_list=['Tree_scikit','RF_scikit','GBT_scikit','logreg_scikit','SVM_scikit','Tree_mllib','RF_mllib','GBT_mllib','logreg_mllib','SVM_mllib']

    # Write output to files
    write_dict2file = False

    format_json = False
    format_csv = True

    # This name could be more specific here
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
      if 'Tree_scikit' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        tree_scikit_par_dict={}
        #tree_scikit_par_dict['numClasses'] = 2
        tree_scikit_par_dict['criterion'] = 'gini'
        tree_scikit_par_dict['max_depth'] = 2

        model_scikit_tree=tree.DecisionTreeClassifier(criterion=tree_scikit_par_dict['criterion'], max_depth=tree_scikit_par_dict['max_depth'])

        model_scikit_tree.fit(X_train,Y_train)

        scikit_model_accuracy('Tree',model_scikit_tree,X_train,Y_train,X_test,Y_test)

        modelparameter_dict['Tree_scikit']=tree_scikit_par_dict

        runtime_write('Tree_scikit',t0)    


      ###############################################################################
      ###############################################################################
      # Random forest
      if 'RF_scikit' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        RF_scikit_par_dict={}
        RF_scikit_par_dict['criterion'] = 'gini'
        RF_scikit_par_dict['max_depth'] = 10
        RF_scikit_par_dict['n_estimator'] = 80
        RF_scikit_par_dict['max_features'] = None
        #RF_scikit_par_dict['max_features'] = 'sqrt'
        #RF_scikit_par_dict['bootstrap'] = False
        RF_scikit_par_dict['bootstrap'] = True
        RF_scikit_par_dict['n_jobs'] = 4 
        # currently not used

        model_scikit_RF=RandomForestClassifier(max_depth=RF_scikit_par_dict['max_depth'], max_features=RF_scikit_par_dict['max_features'],
                                                bootstrap=RF_scikit_par_dict['bootstrap'], n_jobs=RF_scikit_par_dict['n_jobs'],
                                                n_estimators=RF_scikit_par_dict['n_estimator'], criterion=RF_scikit_par_dict['criterion'])

        model_scikit_RF.fit(X_train,Y_train)

        scikit_model_accuracy('RF',model_scikit_RF,X_train,Y_train,X_test,Y_test)

        modelparameter_dict['RF_scikit']=RF_scikit_par_dict

        runtime_write('RF_scikit',t0)
      ###############################################################################
      # Gradient boosted trees
      if 'GBT_scikit' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        GBT_scikit_par_dict={}
        GBT_scikit_par_dict['loss'] = 'deviance'
        GBT_scikit_par_dict['max_depth'] = 10
        GBT_scikit_par_dict['n_estimator'] = 40
        GBT_scikit_par_dict['learning_rate']=0.1
        GBT_scikit_par_dict['subsample']=1
        GBT_scikit_par_dict['min_samples_split']=5
        GBT_scikit_par_dict['min_samples_leaf']=5

        model_scikit_GBT=GradientBoostingClassifier(
            loss=GBT_scikit_par_dict['loss'], learning_rate=GBT_scikit_par_dict['learning_rate'], n_estimators=GBT_scikit_par_dict['n_estimator'],
            subsample=GBT_scikit_par_dict['subsample'], min_samples_split=GBT_scikit_par_dict['min_samples_split'], 
             min_samples_leaf=GBT_scikit_par_dict['min_samples_leaf'], max_depth=GBT_scikit_par_dict['max_depth'],
            init=None, random_state=1, verbose=0,  max_leaf_nodes=None, warm_start=False)

 
        model_scikit_GBT.fit(X_train,Y_train)

        scikit_model_accuracy('GBT',model_scikit_GBT,X_train,Y_train,X_test,Y_test)

        modelparameter_dict['GBT_scikit']=GBT_scikit_par_dict

        runtime_write('GBT_scikit',t0)

      ###############################################################################
      # Logistic regression
      if 'logreg_scikit' in model_list:
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
      if 'SVM_scikit' in model_list:
        t0=datetime.datetime.now()
        model_scikit_SVM =svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
                              kernel='rbf', max_iter=-1, probability=False, random_state=None,
                              shrinking=True, tol=0.001, verbose=False)

        model_scikit_SVM.fit(X_train,Y_train)

        scikit_model_accuracy('SVM',model_scikit_SVM,X_train,Y_train,X_test,Y_test)

        runtime_write('SVM_scikit',t0)


    ###############################################################################


    if run_spark_mllib:
      ###############################################################################
      # Spark MLLib part
      ###############################################################################

      sc = SparkContext("local[2]",appName="Sparkclassifiers")
    
      trainingData_rdd=npmat_to_rdd_wreadwrite(sc,X_train,Y_train,temp_train_filename,delete_file=delete_temp_file)
      testData_rdd=npmat_to_rdd_wreadwrite(sc,X_test,Y_test,temp_test_filename,delete_file=delete_temp_file)

      #print "Training size: "+str(trainingData_rdd.count())

      trainingData_rdd.cache()

      ###############################################################################
      if 'Tree_mllib' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        tree_mllib_par_dict={}
        tree_mllib_par_dict['numClasses'] = 2
        tree_mllib_par_dict['impurity'] = 'gini'
        tree_mllib_par_dict['maxDepth'] = 10

        model_mllib_tree = DecisionTree.trainClassifier(trainingData_rdd, numClasses=tree_mllib_par_dict['numClasses'], categoricalFeaturesInfo={},
                                     impurity=tree_mllib_par_dict['impurity'], maxDepth=tree_mllib_par_dict['maxDepth'])

        mllib_model_accuracy('Tree',model_mllib_tree,trainingData_rdd,testData_rdd)

        modelparameter_dict['Tree_mllib']=tree_mllib_par_dict

        runtime_write('Tree_mllib',t0)

      ###############################################################################
      # Random forest
      if 'RF_mllib' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        RF_mllib_par_dict={}
        RF_mllib_par_dict['numClasses'] = 2
        RF_mllib_par_dict['impurity'] = 'gini'
        RF_mllib_par_dict['maxDepth'] = 10
        RF_mllib_par_dict['numTrees'] = 40
        RF_mllib_par_dict['maxBins'] = 100
        RF_mllib_par_dict['featureSubsetStrategy'] = 'all'

        model_mllib_RF = RandomForest.trainClassifier(trainingData_rdd, numClasses=RF_mllib_par_dict['numClasses'],
                                         categoricalFeaturesInfo={},
                                         maxBins=RF_mllib_par_dict['maxBins'],
                                         numTrees=RF_mllib_par_dict['numTrees'], featureSubsetStrategy=RF_mllib_par_dict['featureSubsetStrategy'],
                                         impurity=RF_mllib_par_dict['impurity'], maxDepth=RF_mllib_par_dict['maxDepth'])

        mllib_model_accuracy('RF',model_mllib_RF,trainingData_rdd,testData_rdd)

        modelparameter_dict['RF_mllib']=RF_mllib_par_dict

        runtime_write('RF_mllib',t0)

      ###############################################################################
      # Gradient boosted trees
      if 'GBT_mllib' in model_list:
        t0=datetime.datetime.now()

        # Only change parameters here
        GBT_mllib_par_dict={}
        #GBT_mllib_par_dict['numClasses'] = 2
        GBT_mllib_par_dict['loss'] = 'logLoss'
        GBT_mllib_par_dict['maxDepth'] = 10
        GBT_mllib_par_dict['numIterations'] = 40
        GBT_mllib_par_dict['learningRate'] = 0.1

        model_mllib_GBT = GradientBoostedTrees.trainClassifier(trainingData_rdd, categoricalFeaturesInfo={},
                                                        learningRate=GBT_mllib_par_dict['learningRate'], loss=GBT_mllib_par_dict['loss'],
                                                    numIterations=GBT_mllib_par_dict['numIterations'], maxDepth=GBT_mllib_par_dict['maxDepth'])

        mllib_model_accuracy('GBT',model_mllib_GBT,trainingData_rdd,testData_rdd)

        modelparameter_dict['GBT_mllib']=GBT_mllib_par_dict

        runtime_write('GBT_mllib',t0)

      ###############################################################################
      # Logistic regression
      if 'logreg_mllib' in model_list:
        t0=datetime.datetime.now()
        #classmethod train(data, iterations=100, step=1.0, miniBatchFraction=1.0, initialWeights=None, 
        #regParam=0.01, regType='l2', intercept=False, validateData=True)

        iterations=100
        #model_mllib_logreg = LogisticRegressionWithSGD.train(trainingData_rdd,iterations,step=1, miniBatchFraction=1.0, initialWeights=None, 
        # regParam=1, regType='l2')

        model_mllib_logreg = LogisticRegressionWithLBFGS.train(trainingData_rdd, iterations=200, initialWeights=None, 
         regParam=0.01, regType='l2', intercept=False, corrections=10, tolerance=0.0001)

        mllib_model_accuracy('logreg',model_mllib_logreg,trainingData_rdd,testData_rdd)

        runtime_write('logreg_mllib',t0)
      ###############################################################################

      ###############################################################################
      # Support Vector machines (SVM) only linear kernel (dont use now)
      if 'SVM_mllib' in model_list:
        t0=datetime.datetime.now()
        model_mllib_SVM = SVMWithSGD.train(trainingData_rdd)
        #model_mllib_SVM = SVMWithSGD.train(trainingData_rdd, iterations)

        mllib_model_accuracy('SVM',model_mllib_SVM,trainingData_rdd,testData_rdd)

        runtime_write('SVM_mllib',t0)

      ###############################################################################

      sc.stop()

    #print metrics_dict
    #print timing_dict
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
      f_name='modelparameters'+'.json'
      with open(f_name,'w') as f:
        f.write(json.dumps(modelparameter_dict,indent=2))


