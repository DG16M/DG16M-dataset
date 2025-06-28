import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, num_points=2500, input_chann=3):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class SimpleSTN3d(nn.Module):
    def __init__(self, num_points=2500, input_chann=3):
        super(SimpleSTN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class DualPointNetfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=6, global_feat=True):
        super(DualPointNetfeat, self).__init__()
        self.stn1 = SimpleSTN3d(num_points=num_points, input_chann=input_chann // 2)
        self.stn2 = SimpleSTN3d(num_points=num_points, input_chann=input_chann // 2)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans1 = self.stn1(x[:, 0:3, :])
        trans2 = self.stn2(x[:, 3:6, :])
        x = x.transpose(2, 1)
        x = torch.cat([torch.bmm(x[..., 0:3], trans1), torch.bmm(x[..., 3:6], trans2)], dim=-1)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans1 + trans2
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans1 + trans2


# Epoch 1: Train loss: 0.4793356829428739, Train acc: 0.7731627823912849
# Epoch 1: Val loss: 0.5300648773829066, Val acc: 0.6813367039946515


'''
Epoch 1: Train loss: 0.4905876951899068, Train acc: 0.7717117599785696
Epoch 1: Val loss: 0.44569133866136884, Val acc: 0.8483202406819321
AdaptiveAvgPool1d
'''

class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 512, 1)
        # self.conv6 = torch.nn.Conv1d(1024, 2048, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.bn5 = nn.BatchNorm1d(512)
        # self.bn6 = nn.BatchNorm1d(2048)
        # self.mp1 = torch.nn.MaxPool1d(num_points)
        self.mp1 = torch.nn.AdaptiveAvgPool1d(1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        # x = torch.cat([torch.bmm(x[..., 0:3], trans), torch.bmm(x[..., 3:6], trans)], dim=-1)
        x = x.transpose(2, 1)
        
        x = F.elu(self.bn1(self.conv1(x))) # (bs, 64, N)
        pointfeat = x
        x = F.elu(self.bn2(self.conv2(x))) # (bs, 128, N)
        
        x = F.elu(self.bn3(self.conv3(x))) # (bs, 256, N)
        x = self.dropout(x)
        x = F.elu(self.bn4(self.conv4(x))) # (bs, 512, N)
        x = F.elu(self.bn5(self.conv5(x))) # (bs, 512, N)
        # x = F.elu(self.bn6(self.conv6(x))) # (bs, 1024, N)
        
        # x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 512)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class DualPointNetCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2, output_logits=True):
        super(DualPointNetCls, self).__init__()
        self.num_points = num_points
        self.output_logits = output_logits
        self.feat = DualPointNetfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x) # batch, 1024
        x = F.relu(self.bn1(self.fc1(x))) # batch, 512
        x = F.relu(self.bn2(self.fc2(x))) # batch, 256
        x = self.fc3(x) # batch, k
        if self.output_logits:
            return x, trans
        else:
            return F.log_softmax(x, dim=-1), trans


class PointNetCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


class PointNetDenseCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, input_chann=input_chann, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans


if __name__ == '__main__':
    pass
