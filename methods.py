#encoding:utf-8
__author__ = 'LuYi'
"""
参数集
"""
from pandas import *
from scipy import stats
import numpy as np
from numpy import *
import pandas as pd
import math
import scipy.io as scio
from sklearn.cross_validation import *

class Methodset:

    def __init__(self):

        pass

    def convert_mat_to_csv(self,file_name):
        result = scio.loadmat(file_name+".mat")
        data_result=result['eyes']
        data_result1=np.ndarray.transpose(data_result)
        print data_result1,type(data_result1)
        df=pd.DataFrame(data_result1)
        df.to_csv(file_name+".csv")
        #, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')[source]
             # columns=['val'])

        print df

    def input_data_by_csv(self,file_name,index_col_num):
        data = pd.read_csv(file_name,index_col=index_col_num, header=None)
        class_list = data.axes[0]
        class_set = list(set(data.axes[0]))
        key_list = list(set(data.keys()))
        index_num = index_col_num
        sample_num = len(data.axes[0])
        # print sample_num
        return data,class_set,key_list,index_num,sample_num,class_list

    def get_max_decision(self,a_dic):
        # print a_dic
        key_list = a_dic.keys()
        value_list = a_dic.values()
        max_value = max(value_list)
        max_key=key_list[value_list.index(max_value)]
        return max_key


    # def get_group(self,dataset):
    #     """
    #     初始化_data_by_class和prior_probability
    #     将导入的_data 按类别分组，获得_data_by_class
    #
    #     :return:
    #     _data_by_class格式为字典，key为类别标记，value为DataFrame格式的样本数据
    #
    #     """
    #
    #     data_by_class={}
    #
    #     for class_i in dataset.classnamelist:
    #         # print i_index
    #
    #         data_by_class[class_i]=dataset.data.xs(class_i)
    #
    #     return data_by_class

    def kernel_function(self,x,name):

        if name == "Gauss":
            pp=self.gauss_kernel(x)

        return pp

    def get_sample_covariance_matrix(self, sample_data):
        mean_i = sample_data.mean()
        #print type(sample_data)
        if isinstance(sample_data,pandas.core.series.Series):
            sample_data_1 = sample_data.to_frame()
        else:
            sample_data_1= sample_data
        #print type(sample_data_1)

        covMatrix = sample_data_1.cov().as_matrix(columns=None)
        #covMatrix = np.cov(sample_data)
        return covMatrix,mean_i

    def modify_covariance(self,cov):
        a, b = np.linalg.eig(cov)
        a_modeified = [0 for i in range(len(a))]
        sum_a = sum(a)
        c = 0
        for i in range(len(a)):
            index_i = np.where(a==sorted(a,reverse=True)[i])[0][0]
            a_modeified[index_i] = a[index_i]
            c += a[index_i]
            cc = float(c)/sum_a
            delta = float(sum_a-c)/float(len(a) - index_i + 1)
            while 0.99 <= cc:
                a_modeified[index_i] = delta
                index_i +=1
                if len(a) == index_i:
                    break

        cov_modefied = np.dot(np.dot(np.linalg.inv(b), mat(diag(a))),b)

        return cov_modefied



    def normal_distribution(self,x,mean,std):
        #正态分布
        y=stats.norm.pdf(x,mean,std)
        return y

    def gauss_kernel(self,x):
        a=1.00/(0.5*math.pi)
        t = -(0.5*x**2)
        # print "t:",t
        pp=a*math.e**(t)

        return pp

    def normal_distribution_1(self,x,mean,std):
        x_1=((2*3.14)**0.5)*std
        print "x_1", x_1
        y_1=(x-mean)**2
        y_2=2*(std**2)
        t_1=-y_1/y_2
        print "t_1",t_1
        x_2=math.e**(t_1)
        print "x_2",x_2
        tt=1/x_1
        print "tt",tt
        # tt_1=1/x_2
        # print "7",tt_1
        return tt*x_2

    def regularize_covMtrix(self, classCovMtrix, prior_probability, gamma, beta, d):
        regularize_class_conMtrix = {}
        matix_0 = 0
        sigma2 = {}

        for class_i in classCovMtrix:
            matix_0 += classCovMtrix[class_i] * float(prior_probability[class_i])
            sigma2[class_i] = np.trace(classCovMtrix[class_i])/float(d)
        c1 = float(1-gamma)
        c2 = float(1-beta)
        for class_i in classCovMtrix:
            regularize_class_conMtrix[class_i] = c1*(c2 * classCovMtrix[class_i] + float(beta) * matix_0)+\
                                                 float(gamma)* sigma2[class_i]*mat(eye(d, d, dtype=int))
        return regularize_class_conMtrix


    # def split_into_folds(self,dataset, n_flods, ):
    #




if __name__ == "__main__":
    test=Methodset()
    # c=test.input_data("Pimaindians.csv",8)
    # print c

    c1=test.convert_mat_to_csv("D:\workspace\iris\DATAA\Eyes")
    print c1