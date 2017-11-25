
#encoding:utf-8
from __future__ import division
from dataclass import *
__author__ = 'LuYi'

"""
主要实现概率密度估计方法
1、有监督的参数估计，采用矩估计方式
2、无监督采用parzen窗和k近邻
"""
from methods import *

class D_Parzen(Methodset):
    """
    d维
    """
    def __init__(self):
        Methodset.__init__(self)







    def get_classconditional_probability(self, x, dataset, h_result):
        """


        :param x:
        :return:
        """
        # h_result=self.get_h(dataset)
        index_h =h_result
        cc = 1
        for index_name in dataset.indexnamelist:
            cc *= index_h[index_name]
        coefficient = 1.0 / (dataset.num * cc)
        #coefficient=h_result[1]
        pp=0
        for i in range(dataset.num):
            p=1
            for index_name in dataset.data.keys():
                #print dataset.data[index_name]
                #print dataset.data.keys()
                x_i=dataset.data[index_name].iloc[i]
                #print index_name,index_h,x,i,dataset.num

                XX=float(x[index_name-dataset.data.keys()[0]]-x_i)/float(index_h[index_name])
                p_kerel=self.kernel_function(XX,"Gauss")
                p*=p_kerel
                # if p_kerel!=0:
                #     print p_kerel

            pp+=p



        probility=coefficient*pp
        # if probility!=0:
        #     print probility

        return probility





    def get_h(self,dataset):
        index_h={}
        coefficient = 1
        cc=1
        for index_name in dataset.indexnamelist:
            index_max=max(dataset.data[index_name])
            index_min=min(dataset.data[index_name])
            index_h[index_name]=(index_max-index_min)/float(3000**0.5)
            # index_h[index_name]=0.001

        # print coefficient
        return index_h,coefficient









if __name__ == "__main__":
    testt = D_Parzen("iris.csv",4)
    x={'c_1':5.1, 'c_3':3.5, 'c_2':1.4, 'c_4':0.2}
    #5.1,3.5,1.4,0.2
#6.2,3.4,5.4,2.3
    #(5.6,3.0,4.5,1.5)
    testt.get_posterior_probability(c_1=5.6, c_3=3.0,c_2=3.5,c_4=0.2)