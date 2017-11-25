#encoding:utf-8
__author__ = 'LuYi'
from methods import Methodset
import pandas as pd
import copy
from sklearn.cross_validation import *


class Dataset(Methodset):
    def __init__(self):
        Methodset.__init__(self)
        self.name = None
        self.data = None
        self.classnamelist = []  #存放类别名称
        self.indexnamelist = []  #存放指标名称
        self.classnum = None       #类别的数量
        self.indexnum = None       # 指标数量
        self.num = None            #样本个数
        self._data_by_class ={}
        self.prior_probability = {}


    def importdata(self,**kwargs):
        if "filename" in kwargs:
            fname=kwargs["filename"]
            #dim=kwargs["dim"]
            data = self.input_data_by_csv(fname, kwargs["index_col"] )
            self.data = data[0]

            # print data[2]
            self.indexnum = data[3]
            self.num = data[4]
            self.indexnamelist = data[2]
            self.classnamelist = data[1]
            self.class_list=data[5]
            return None


        elif "framedata" in kwargs:
            self.data  = kwargs["framedata"]
            self.classnamelist=list(set(self.data.axes[0]))
            self.indexnamelist=list(set(self.data.keys()))
            self.indexnum=len(self.indexnamelist)
            self.classnum=len(self.classnamelist)
            self.num=len(self.data.axes[0])
            return None

        elif "url" in kwargs:
            dataset = pd.read_csv(kwargs["url"],index_col=kwargs["index_col"],header=None, usecols= kwargs["usecol"])
            # dataset.to_csv(kwargs["url"]+".csv")
            # pd.read_csv(kwargs["url"]+".csv")
            self.importdata(framedata=dataset)
            self.indexnamelist = kwargs["index_name"]
            return None

        elif "data_dic" in kwargs:
            return None
        # self.split_dataset()


    def get_group(self):
        """
        初始化_data_by_class和prior_probability
        将导入的_data 按类别分组，获得_data_by_class

        :return:
        _data_by_class格式为字典，key为类别标记，value为DataFrame格式的样本数据

        """


        for i_index in self.classnamelist:
            # print i_index
            self._data_by_class[i_index] = self.data.xs(i_index)
            self.prior_probability[i_index] = 1.0/ float(len(self.classnamelist)) # 先验概率值
        #print self._data_by_class


class TrainData(Dataset):

    def __init__(self):
        Dataset.__init__(self)
        self.covMatrix = None
        self.mean = None
        self.classCovMatrix = {}
        self.classMean = {}
        self.folds_data = []

    def split_into_folds(self, n_folds):
        kf = KFold(self.num, n_folds, shuffle = False)
        for iteration, index_data in (kf):
            # print iteration, index_data
            # print type(iteration),type(index_data)
            # print "88:",type(self.data),self.num
            # print "index0",index_data[0]
            # print "iteration",iteration
            a = self.data.iloc[iteration]
            b = self.data.iloc[index_data]
            a1 =copy.deepcopy(a)
            b1 =copy.deepcopy(b)
            # print a1
            # print b1
            train_data_i = TrainData()
            train_data_i.importdata(framedata = a1)

            train_data_i.indexnamelist = self.indexnamelist
            test_data_i = TestData()
            train_data_i.name = self.name
            test_data_i.name =self.name
            test_data_i.importdata(framedata = b1)
            test_data_i.indexnamelist = self.indexnamelist
            self.folds_data.append([train_data_i, test_data_i])

        # print 0



class TestData(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.result = []
        self.target_list = []

    def importdata(self,**kwargs):
        self.data = kwargs["framedata"]
        self.classnamelist = list(set(self.data.axes[0]))
        self.indexnamelist = list(set(self.data.keys()))
        self.indexnum = len(self.indexnamelist)
        self.classnum = len(self.classnamelist)
        self.num = len(self.data.axes[0])
        self.target_list = list(self.data.index)
