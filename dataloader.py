import util
import numpy as np

class Dataloader(object):
    def __init__(self, dataset_A, dataset_B, shape=(36, 128)):
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.shape = shape
        
    def loadData(self):
        data_A = util.loadPickle(self.dataset_A)[0]
        data_B = util.loadPickle(self.dataset_B)[0]
        data_A = np.concatenate([*map(lambda x:x.flatten(), data_A)])
        data_B = np.concatenate([*map(lambda x:x.flatten(), data_B)])
        data_A = np.reshape(data_A, (self.shape[0], -1))
        data_B = np.reshape(data_B, (self.shape[0], -1))
        self.data_A = data_A
        self.data_B = data_B
        self.data_len = min(data_A.shape[1], data_B.shape[1]) 
        self.data_len = self.data_len - ( self.data_len // self.shape[1] )
    
    def loadBatch(self):
        for i in range(0, self.data_len, self.shape[1]):
            yield self.data_A[:, i:i+self.shape[1]].reshape((1,)+self.shape), self.data_B[:, i:i+self.shape[1]].reshape((1,)+self.shape)

    
