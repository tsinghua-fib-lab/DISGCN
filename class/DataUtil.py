

import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf

    def initializeRankingHandle(self):
        self.createTrainHandle()
    
    def createTrainHandle(self):
        data_dir = self.conf.data_dir
        train_filename = f'{data_dir}/train'
        val_filename = f'{data_dir}/val'
        test_filename = f'{data_dir}/test'
        filename_link = f'{data_dir}/social'


        self.train = DataModule(self.conf, train_filename, filename_link)# train_filename_link)
        self.val = DataModule(self.conf, val_filename, filename_link)# val_filename_link)
        self.test = DataModule(self.conf, test_filename, filename_link)# test_filename_link)
        if self.conf.test:
            self.test_eva = DataModule(self.conf, test_filename, filename_link)# test_filename_link)
        else:
            self.test_eva = DataModule(self.conf, val_filename, filename_link)
