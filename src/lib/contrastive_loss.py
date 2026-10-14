import torch
import numpy as np
import torch.nn as nn
import math


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion_h = nn.CrossEntropyLoss(reduction="none")
        self.similarity_h = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
    #初始化对角矩阵（2N*2N）
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask


    def forward(self, z_i, z_j): #[64,512]
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0) #[128,512]
        sim = torch.matmul(z, z.T) / self.temperature #余弦相似性? [128,128]
        sim_i_j = torch.diag(sim, self.batch_size) #[64]
        sim_j_i = torch.diag(sim, -self.batch_size) #[64]

        a = sim[self.mask]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #[128,1]
        negative_samples = sim[self.mask].reshape(N, -1) #[128,126]  [16128] [128,128]
        labels = torch.zeros(N).to(positive_samples.device).long() #128个0  [128]
        logits = torch.cat((positive_samples, negative_samples), dim=1)  #[128,127]

        # w = self.criterion_h(logits, labels)  # [128]
        # weight = del_tensor_ele(w, N - 1)
        # w = torch.tensor(weight, dtype=torch.float)

        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        # mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)#求数组每一列的和，然后将矩阵变成1维
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j #H(Y)

        c_i = c_i.t() #[10,64]
        c_j = c_j.t()

        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0) #[20,64]

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature #扩展维度 [20,1,64] [1,20,64] [20,20]
        sim_i_j = torch.diag(sim, self.class_num) #[10]
        sim_j_i = torch.diag(sim, -self.class_num) #[10]

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #[20,1]
        negative_clusters = sim[self.mask].reshape(N, -1) #[20,19]

        labels = torch.zeros(N).to(positive_clusters.device).long() #[20]
        logits = torch.cat((positive_clusters, negative_clusters), dim=1) #[20,20]

        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
