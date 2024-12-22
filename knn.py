import torch

def randomize_tensor(tensor):
    return tensor[torch.randperm(len(tensor))]

def distance_matrix(x, y=None, p = 2): # pairwise distance of vectors x.shape=[50000, 10]
    
    y = x if type(y) == type(None) else y  

    n = x.size(0) # 100
    m = y.size(0) # 50000
    d = x.size(1) # 10

    x = x.unsqueeze(1).expand(n, m, d)  # x.shape=[100, 10]   => x.unsqueeze(1).shape=[100, 1, 10]   => x.unsqueeze(1).expand(n, m, d).shape=[100, 50000, 10]
    # x = x.unsqueeze(1).repeat(1, y.shape[0], 1)  # x.shape=[100, 10]   => x.unsqueeze(1).shape=[100, 1, 10]   => x.unsqueeze(1).expand(n, m, d).shape=[100, 50000, 10]

    dist = torch.pow(x - y, p).sum(2)   # dist.shape=[100, 50000]
    
    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X    
        self.train_label = Y   # 0, 1, 2, ..., len(X) 

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2, d = 1e-3):
        self.k = k
        self.d = d
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        

        dist = distance_matrix(x.cpu(), self.train_pts, self.p) ** (1/self.p)  

        # value.shape = indice.shape = [100, 1]
        value, indice = dist.topk(self.k, largest=False)  
        # value, indice = torch.max(dist, dim=1, keepdim=True)

        mask = value < self.d

        votes = self.train_label[indice] 

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) -1   
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)      

        for lab in self.unique_labels:
            vote_count = torch.logical_and((votes == lab), mask).sum(1)   
            who = vote_count > count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

if __name__ == '__main__':
    import torch, glob, os
    import config.config_common as config_common
    dataset = 'cifar10'
    
    _DIR =  os.path.join(config_common.DIR_TO_SAVE_SOFTMAX_CLASSIFIER, f'{dataset}_targetmodel')
    _PATH_D0 = os.path.join(_DIR, f"{dataset}_targetmodel_D0.pth") # 

    softmax_member = torch.load(_PATH_D0)['softmax']
    label = torch.Tensor(list(range(len(softmax_member)))).long()
    knn = KNN(softmax_member.cpu(), label.cpu(), k=1, p=2, d=0.000001)

    files = glob.glob(f"{_DIR}/*")

    files = sorted(files, key=lambda x:x)[:3]
    # print(files)
    
    for file in files:
        num, batch_size = 0, 1000
        softmax_to_test = torch.load(file)['softmax']
        length = len(softmax_to_test)
        num_idx = length // batch_size
        for idx in range(num_idx):
            begin = idx * batch_size
            end = (idx + 1) * batch_size if ((idx + 1) * batch_size) < length else length
            to_test = softmax_to_test[begin:end]
            result = knn(to_test)
            num += (result!=-1).sum()
            print(f"member: {num}/{length}, non-member: {length-num}/{length} in {os.path.basename(file)}")
            
        
        

    print()
