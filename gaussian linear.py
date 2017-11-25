# -*- coding: utf-8 -*-
__author__ = 'luyi-pamela'
import pandas as pd
import numpy as np
import math
from dataclass import *
from sklearn.cross_validation import *
from sklearn import metrics

from methods import Methodset

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

"""
第一次作业 LDF
在类别条件分布服从等协方差高斯分布假设下，分界面为现行分界面的分类器实现
"""
class GaussianClassifiers(Dataset, Methodset):

    def __init__(self):
        Dataset.__init__(self)
        Methodset.__init__(self)

        #self.file_name = None

        self.prior_probability = {}#类别：先验概率
        self._data_by_class = {}
        self.training_data = TrainData()
        self.test_data = TestData()


        self.posterior_probability = {}#类别：后验概率结果
        self.classconditional_probability = {} # 类条件概率
        self.method_class = {"lda":self.lda_classification,
                             "qdf":self.qdf_classification,
                             "mqdf":self.mqdf_classification,
                             "RDA": self.rda_classification
        }
        # todo：没有将数据分为训练集及测试集，后续可写一个函数，随机选择部分数据分别进入test_data和training_data中

        pass









    def discriminant_function(self, x, covMatrix, mean_i, p_i):
        #covMatrix = self.data.cov().as_matrix(columns=None)
       # print covMatrix
        cov_inverse = np.linalg.pinv(covMatrix)
        a = x-mean_i
        b = np.dot(a.T, cov_inverse)
        c = -np.dot(b, a)

        discriminant_function_value = c + 2 * math.log(p_i)
        #print  "result:", surface_fuction_value
        return discriminant_function_value

    def lda_classification(self,test_data):

        self.training_data.get_group()
        for class_i in self.training_data.classnamelist:
            covMatrix, mean= self.get_sample_covariance_matrix(self.training_data.data)
            mean_i = self.training_data._data_by_class[class_i].mean()
            self.training_data.classCovMatrix[class_i] = covMatrix
            self.training_data.classMean[class_i] = mean_i
        right_num = 0
        for i in range(test_data.num):
            x = test_data.data._get_values[i]

            result_i = self.point_classification(x)
            if result_i == test_data.target_list[i]:
                right_num += 1

            self.test_data.result.append(result_i)
        score = float(right_num) / float(test_data.num)
        return score

    def get_parameter_by_cross_validate(self, method_name, folds_n):
        beta_list = []
        gamma_list = []
        result_score = []
        self.training_data.split_into_folds(folds_n)
        for time_beta in range(10):
            for time_gamma in range(10):
                scores = []
                gamma_1 = 0.1 * float(time_gamma)
                beta_1 = 0.1 * float(time_beta)
                # gamma_1 = 0.1
                # beta_1 = 0.1
                for i in range(folds_n):
                    training_data_i = self.training_data.folds_data[i][0]
                    test_data_i = self.training_data.folds_data[i][1]


                    score = self.method_class[method_name](test_data_i, gamma = gamma_1 , beta = beta_1, training_data = training_data_i)
                    #print score
                    scores.append(score)

                score_mean = np.mean(scores)
                # print score_mean
                result_score.append(score_mean)
                beta_list.append(beta_1)
                gamma_list.append(gamma_1)

        fig = plt.figure()

        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        # ax.plot_surface(np.array(beta_list), np.array(gamma_list), np.array(result_score), rstride=1, cstride=1, cmap='rainbow')
        ax.scatter(beta_list, gamma_list, result_score, c='y')
        ax.set_zlabel('score')  # 坐标轴
        ax.set_ylabel('gamma')
        ax.set_xlabel('beta')
        #ax.set_title(text=self.name, loc="center")
        # 绘制数据点
        plt.savefig(str(self.name+".png"))

        gamma_best = gamma_list[result_score.index(max(result_score))]
        beta_best = beta_list[result_score.index(max(result_score))]
        return gamma_best, beta_best

    def rda_classification(self,test_data, gamma = 0.1, beta = 0.00, training_data = None):
        if training_data == None:
            training_data = self.training_data

        training_data.get_group()
        classCovMatrix_1 ={}
        for class_i in training_data._data_by_class:
            covMatrix, mean= self.get_sample_covariance_matrix(training_data._data_by_class[class_i])
            mean_i = training_data._data_by_class[class_i].mean()
            classCovMatrix_1[class_i] = covMatrix
            training_data.classMean[class_i] = mean_i


        d = len(training_data.indexnamelist)
        training_data.classCovMatrix = self.regularize_covMtrix(classCovMatrix_1, training_data.prior_probability, gamma, beta, d)
        right_num = 0
        for i in range(test_data.num):
            x = test_data.data._get_values[i]

            result_i = self.point_classification(x, training_data)
            if result_i == test_data.target_list[i]:
                right_num += 1

            self.test_data.result.append(result_i)
        score = float(right_num) / float(test_data.num)
        return score




    def qdf_classification(self,test_data):
        #print "1",type(test_data)
        self.training_data.get_group()

        for class_i in self.training_data.classnamelist:
            covMatrix, mean_i = self.get_sample_covariance_matrix(self.training_data._data_by_class[class_i])
            self.training_data.classCovMatrix[class_i] = covMatrix
            self.training_data.classMean[class_i] = mean_i
        right_num  = 0
        for i in range(test_data.num):
            x = test_data.data._get_values[i]

            result_i = self.point_classification(x)
            if result_i == test_data.target_list[i]:

                right_num += 1

            self.test_data.result.append(result_i)
        score = float(right_num)/float(test_data.num)
        return score



    def mqdf_classification(self,test_data):

        self.training_data.get_group()
        for class_i in self.training_data.classnamelist:
            covMatrix, mean_i = self.get_sample_covariance_matrix(self.training_data._data_by_class[class_i])
            modify_covmatrix = self.modify_covariance(covMatrix)
            self.training_data.classCovMatrix[class_i] = modify_covmatrix
            self.training_data.classMean[class_i] = mean_i
        right_num = 0
        for i in range(test_data.num):
            x = test_data.data._get_values[i]

            result_i = self.point_classification(x)
            if result_i == test_data.target_list[i]:
                right_num += 1

            self.test_data.result.append(result_i)
        score = float(right_num) / float(test_data.num)
        return score


    def point_classification(self, x, training_data = None):
        if training_data == None:
            training_data = self.training_data
        class_value = {}

        for class_i in training_data.classnamelist:
            covMatrix = training_data.classCovMatrix[class_i]
            mean_i = training_data.classMean[class_i]
            p_i = training_data.prior_probability[class_i]
            class_value[class_i] = self.discriminant_function(x, covMatrix, mean_i, p_i)

        result_x = class_value.keys()[class_value.values().index(max(class_value.values()))]
        return result_x

    def split_data(self,sample_data):
        self.name = sample_data.name
        train_data, test_data =  train_test_split(sample_data.data, test_size = 0.2, random_state=1)
        self.training_data.name = sample_data.name
        self.test_data.name = sample_data.name
        self.training_data.importdata(framedata=train_data)
        self.test_data.importdata(framedata=test_data)

    def get_score(self,test_data, methodname, best_gamma = None, best_beta = None):
        test_data.result = []
        if methodname == "RDA":
            result = self.rda_classification(test_data, gamma = float(best_gamma), beta= (best_beta))
        else:

            result = self.method_class[methodname](test_data)
#        print "sklearn score:",metrics.accuracy_score(self.test_data.target_list, self.test_data.result)


        return result

def url_classification_work(data_set):

    lda_1 = GaussianClassifiers()

    lda_1.split_data(data_set)

    # lda_1.url_data_input(data_url, 4, attribute_information)
    lda_1.get_group()

    print "LDA------------------------------------------------------"
    score = lda_1.get_score(lda_1.test_data, "lda")
    print score
    print "QDF------------------------------------------------------"
    score = lda_1.get_score(lda_1.test_data, "qdf")
    print score
    print "MQDF------------------------------------------------------"
    score = lda_1.get_score(lda_1.test_data, "mqdf")
    print score
    print "RDA------------------------------------------------------"
    gamma, beta = lda_1.get_parameter_by_cross_validate( "RDA", 5)
    print gamma, beta
    score = lda_1.get_score(lda_1.test_data, "RDA", gamma, beta)
    print score




if __name__ == "__main__":

    print "实验1------------------------------------------iris---------------------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
    attribute_information = ["sepal length", "sepal width" ,"petal length", "petal width"]
    usecol = [i for i in range(5)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col = 4, index_name=attribute_information, usecol=usecol)
    data_set.name = "iris_1"
    url_classification_work(data_set)
    print "实验2----------------------------------------------Balance Scale Data Set---------------------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    attribute_information = ["Left-Weight", "Left-Distance","Right-Weight","Right-Distance"]
    usecol = [i for i in range(5)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col=0, index_name=attribute_information, usecol=usecol)
    data_set.name = "balance_scale_1"
    url_classification_work(data_set)


    print "实验3-----------------------------------------------------------------------wine---------------------------------------------------------------------------------"
    f_name = "wine.csv"
    #f_name ="/home/pamela/Documents/python space/machine learninghomework/first/wine.csv"
    attribute_information = ["Alcohol", "malic acid", "ash","alcalinity of ash","mag","tp","fla","nonfla", "proan","colour","hue","oo","proline"]
    usecol1 = [i for i in range(14)]
    data_set_1 = Dataset()
    #data_set.importdata(url=data_url, index_col = 0, index_name=attribute_information, usecol=usecol1)
    data_set_1.importdata(filename=f_name ,index_col=0 )
    data_set_1.name = "wine_1"
    url_classification_work(data_set_1)
    print "实验4------------------------------------------------Breast Cancer Wisconsin (Diagnostic) Data Set---------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    attribute_information = ["radius","texture", "perimeter", "area","smoothness",
                             "compactness","concavity","concave points","symmetry","fractal dimension"]

    usecol = [i for i in range(1,11)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col = 0, index_name=attribute_information, usecol=usecol)
    data_set.name = "wdbc_1"
    url_classification_work(data_set)
    print "实验5-------------------------------Glass Identification Data Set---------------------------------------------------------------------"


    data_url="http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    #
    attribute_information = ["1", "2", "3"]
    data_set = Dataset()
    usecol = [i for i in range(4)]
    data_set.importdata(url=data_url, index_col=3, index_name=attribute_information, usecol=usecol)
    data_set.name = "haberman_1"
    url_classification_work(data_set)
