#encoding:utf-8
__author__ = 'LuYi'
"""
朴素贝叶斯分类器
parzen 窗来估计概率密度
数据结构采用pandas DateFrame
"""
import math
import pandas as pd
import numpy as np
import pandas.util.testing as tm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Probability_density_estimation import *


class Bayes:

    def __init__(self):
        self._data = None
        self.class_list = None
        self.key_list=None
        self.file_name = None
        self.prior_probability={}
        self.posterior_probability = {}
        self.classconditional_probability={}
        #todo：没有将数据分为训练集及测试集，后续可写一个函数，随机选择部分数据分别进入a_sample和training_data中

    def get_prior_probability(self):
        """
        获得先验概率
        :return: self.prior_probability={类别：概率}
        """
        pass

    def get_conditional_probability(self):
        """
        条件概率密度self.classconditional_probability={类别：概率}
        :return:
        """
        pass





    # def class_joint_probability_density(self,a_sample,class_name):
    #     """
    #
    #     :param a_sample:
    #     :param class_name:
    #     :return:
    #     """
    #     dd=self._data.groupby(level="dataclass")
    #     d_2 = dd.describe()
    #     print d_2
    #
    #
    #     result_y=A
    #     for i in range(len(self.key_list)):
    #         str_i="y"+str(i)
    #         x=a_sample[i]
    #         c_i=selAf.key_list[i]
    #         if c_i=="c_7":
    #             dd=0
    #         mean_1 = d_2[c_i].xs((class_name,"mean"))
    #         std_1 = d_2[c_i].xs((class_name,"std"))
    #         y=self.normal_distribution(x,mean_1,std_1)*(10**10)
    #         print class_name,c_i,y
    #         exec(str_i+'=y')
    #         result_y*=eval(str_i)
    #     print result_y
    #     return result_y
    #
    # def get_class_probability(self,a_sample):
    #     self.posterior_probability={}
    #     self.class_probability={}
    #     y=0
    #     for i_class in self.class_list:
    #         self.posterior_probability[i_class]=self.class_joint_probability_density(a_sample,i_class)
    #         y+=self.class_joint_probability_density(a_sample,i_class)
    #     for i_class in self._data_by_class:
    #         self.class_probability[i_class]=self.posterior_probability[i_class]
    #     print self.class_probability
    #
    # def get_probability _by_parzen(self,a_sample):
    #     testt = D_Parzen(self.file_name,self.index_num)
    #
    #     probability =testt.get_posterior_probability(c_1=a_sample[0], c_2=a_sample[1],c_3=a_sample[2],c_4=a_sample[3])
    #     return probability
    #
    # def get_decision_function_value(self,a_sample):
    #     """
    #     计算决策函数g(x)值，x为当前观测值
    #     :param a_sample:
    #     :return:
    #     """
    #
    #     pass
class ParzenBayes(Bayes,D_Parzen):

    def __init__(self):
        self.name  = None
        Bayes.__init__(self)
        D_Parzen.__init__(self)
        self.training_data = TrainData()
        self.test_data = TestData()


    def get_prior_probability(self):
        pass

    def init_data_by_file(self,file_name,index_num):
        self._data = Dataset()
        self._data.importdata(filename=file_name,dim=index_num)

    def init_data_by_Dateframe(self,dataset):
        pass


    def get_conditional_probability(self, a_sample, training_data = None, h = None):
        if training_data == None:
            training_data = self.training_data
        training_data.get_group()
        for class_i in self.training_data.classnamelist:
            class_data=training_data._data_by_class[class_i]
            class_dataset=Dataset()
            class_dataset.importdata(framedata = class_data)
            training_data._data_by_class[class_i] =  class_dataset
            if h == None:
                h = self.get_h(class_dataset)
            probability = self.get_classconditional_probability(a_sample, class_dataset, h )
            self.classconditional_probability[class_i] = float(probability )* training_data.prior_probability[class_i]
        # print self.classconditional_probability

    def get_decision_function_value(self, a_sample, training_data, h):
        self.get_conditional_probability(a_sample, training_data, h)
        index_name=self.get_max_decision(self.classconditional_probability)
        return index_name

    def paren_classification(self, test_data, training_data = None ,h = None ):
        if training_data ==  None:
            training_data = self.training_data
        right_num = 0
        for i in range(test_data.num):
            x = test_data.data._get_values[i]
            #print x,type(x)

            result_i = self.get_decision_function_value(x, training_data, h)
            if result_i == test_data.target_list[i]:
                right_num += 1

            self.test_data.result.append(result_i)
        score = float(right_num) / float(test_data.num)
        return score

    def plot_test(self):
        x_1=[]
        x_2=[]
        y_1=[]
        y_2=[]
        z_1=[]
        z_2=[]
        n=0
        false_set = []
        for i in range(1000):

            x=self._data.data["c_1"][i]
            y=self._data.data["c_2"][i]
            a_sample={'c_1':x, 'c_2':y}
            d=self.get_decision_function_value(a_sample)
            real_d=self._data.class_list[i]
            # z=math.log(self.classconditional_probability[d])
            z=self.classconditional_probability[d]
            # print i,z,d,real_d
            if d=="class1":
                x_1.append(x)
                y_1.append(y)
                z_1.append(z)

            elif d=="class2":
                 x_2.append(x)
                 y_2.append(y)
                 z_2.append(z)


            if d == real_d:
                n+=1

            else:
                false_set.append((x,y,i))

        # print n,false_set
        # print "x",x_1
        # print "y",y_1
        # print "z",z_1
        fig = plt.figure()
        # ax = Axes3D(fig)
        #
        #
        # p1=ax.plot_surface(x_1,y_1,z_1,rstride=1, cstride=1, cmap='rainbow')
        # # p2=ax.plot_surface(x_2,y_2,z_2,rstride=1, cstride=1, cmap='rainbow')
        # plt.show()

        ax=plt.subplot(111,projection='3d')
        p1=ax.scatter(x_1,y_1,z_1,color='r')
        p2=ax.scatter(x_2,y_2,z_2,color='g')

        ax.set_zlabel('Z') #坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.show()


    def plot_data(self):
        x=60*np.random.random(100)-30
        y=60*np.random.random(100)-30
        # plt.scatter(x,y,color = 'm')
        x_1=[]
        x_2=[]
        y_1=[]
        y_2=[]
        z_1=[]
        z_2=[]
        for i in range(100):
            a_sample={'c_1':x[i], 'c_2':x[2]}
            d=self.get_decision_function_value(a_sample)
            z=self.classconditional_probability[d]
            if d=="class1":
                x_1.append(x[i])
                y_1.append(y[i])
                z_1.append(z)

            elif d=="class2":
                 x_2.append(x[i])
                 y_2.append(y[i])
                 z_1.append(z)



        f1 = plt.figure(1)
        p1=plt.scatter(x_1,y_1,s = 50,marker = 'o',color = 'm')
        p2=plt.scatter(x_2,y_2,s = 50,marker = 'o',color = 'y')
        plt.show()

    def split_data(self,dataset):
        train_data, test_data =  train_test_split(dataset.data, test_size = 0.2, random_state = 0)
        self.training_data.name = data_set.name
        self.test_data.name = data_set.name
        self.training_data.importdata(framedata=train_data)
        # self.training_data.indexnamelist = dataset.indexnamelist
        self.test_data.importdata(framedata=test_data)
        # self.test_data.indexnamelist = dataset.indexnamelist

    def get_parameter_by_cross_validate(self, folds_n, training_data):
        h_list = []
        result_score = []
        index_h = {}
        self.training_data.split_into_folds(folds_n)
        c=1
        index_h_i = {}
        #cc= (0.1**(float(len(training_data.indexnamelist))))*c
        for h_i in range(1,10):
            h =   0.1*h_i
            scores = []
            for index_name in training_data.indexnamelist:
                index_max = max(training_data.data[index_name])
                index_min = min(training_data.data[index_name])
                index_h_i[index_name] = (index_max - index_min) * h

            # gamma_1 = 0.1
            # beta_1 = 0.1
            for i in range(folds_n):
                training_data_i = training_data.folds_data[i][0]
                test_data_i = training_data.folds_data[i][1]

                score = self.paren_classification(test_data_i, training_data_i,index_h_i)
                # print score
                scores.append(score)

            score_mean = np.mean(scores)

            #score_mean, h
            result_score.append(score_mean)
            h_list.append(h)



        plt.plot(h_list, result_score)  # 创建一个三维的绘图工程
        # ax.plot_surface(np.array(beta_list), np.array(gamma_list), np.array(result_score), rstride=1, cstride=1, cmap='rainbow')
        #plt.title(self.name)

        # 绘制数据点
        plt.savefig(str(self.name+".png"))

        #plt.show()

        h_best = h_list[result_score.index(max(result_score))]
        print "h_best", h_best
        for index_name in training_data.indexnamelist:
            index_max=max(training_data.data[index_name])
            index_min=min(training_data.data[index_name])
            index_h[index_name]=(index_max-index_min) * h_best



        return index_h


def parzen_bayes(data_set):


    exa_1 = ParzenBayes()
    exa_1.name = data_set.name



    exa_1.split_data(data_set)
    #
    # print score
    h_best = exa_1.get_parameter_by_cross_validate(5, exa_1.training_data)
    #h_best = {0: 0.07, 1: 0.03799999999999999, 2: 0.114, 3: 0.048}

    score = exa_1.paren_classification(exa_1.test_data, h=h_best)
    print score



if __name__ == "__main__":
    # exa_1=ParzenBayes()
    # print "----------------------------------"
    # # 鸢尾花
    # data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
    # attribute_information = ["sepal length", "sepal width", "petal length", "petal width"]
    # data_set = Dataset()
    # data_set.importdata(url=data_url, index_col=4, index_name=attribute_information)
    #
    #
    # exa_1.split_data(data_set)
    # #
    # # print score
    # # h_best = exa_1.get_parameter_by_cross_validate(5, exa_1.training_data)
    # h_best = {0: 0.07, 1: 0.03799999999999999, 2: 0.114, 3: 0.048}
    # score = exa_1.paren_classification(exa_1.test_data,h=h_best)
    # print score
    # a_sample={'c_1':5.6, 'c_2':2.9, 'c_3':3.6, 'c_4':1.4}
       # # exa_1.class_joint_probability_density(5.6,3.0,4.2,1.3,u"Iris-virginica")
       # d=exa_1.get_decision_function_value(a_sample)
       # print d
       # print "-----------------------------------------"
    #    #the survival of patients who had undergone surgery for breast cancer.
    #    exa_1.init_data_by_file("data1.csv",2)
    #
    #
    #
    # # -0.221445015	-9.340940605
    #
    #    a_sample={'c_1':-3.924379375, 'c_2':-0.35612799499999997}
    #
    #    d=exa_1.get_decision_function_value(a_sample)
    #
    #    print d
    #   exa_1.plot_test()
       # exa_1.plot_result()

    print "实验1------------------------------------------iris---------------------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
    attribute_information = ["sepal length", "sepal width", "petal length", "petal width"]
    usecol = [i for i in range(5)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col=4, index_name=attribute_information, usecol=usecol)
    data_set.name = "iris"
    parzen_bayes(data_set)
    print "实验2----------------------------------------------Balance Scale Data Set---------------------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    attribute_information = ["Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
    usecol = [i for i in range(5)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col=0, index_name=attribute_information, usecol=usecol)
    data_set.name = "balance_scale"
    parzen_bayes(data_set)

    print "实验3-----------------------------------------------------------------------wine---------------------------------------------------------------------------------"

    f_name = "wine.csv"
    attribute_information = ["Alcohol", "malic acid", "ash", "alcalinity of ash", "mag", "tp", "fla", "nonfla", "proan",
                             "colour", "hue", "oo", "proline"]
    usecol1 = [i for i in range(14)]
    data_set_1 = Dataset()
    # data_set.importdata(url=data_url, index_col = 0, index_name=attribute_information, usecol=usecol1)
    data_set_1.importdata(filename=f_name, index_col=0)
    data_set_1.name = "wine"
    parzen_bayes(data_set_1)
    print "实验4------------------------------------------------Breast Cancer Wisconsin (Diagnostic) Data Set---------------------------------------------------------------------"
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    attribute_information = ["radius", "texture", "perimeter", "area", "smoothness",
                             "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]

    usecol = [i for i in range(1, 12)]
    data_set = Dataset()
    data_set.importdata(url=data_url, index_col=0, index_name=attribute_information, usecol=usecol)
    data_set.name = "wdbc"
    parzen_bayes(data_set)
    print "实验5-------------------------------Glass Identification Data Set---------------------------------------------------------------------"

    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    #
    attribute_information = ["1", "2", "3"]
    data_set = Dataset()
    usecol = [i for i in range(4)]
    data_set.importdata(url=data_url, index_col=3, index_name=attribute_information, usecol=usecol)
    data_set.name = "haberman"
    parzen_bayes(data_set)

    #

