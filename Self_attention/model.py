from calendar import day_abbr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGPooling
from torch_geometric.nn import global_add_pool as gap,global_max_pool as gmp

class SAGPooling_Global(nn.Module):
    def __init__(self,args,num_features,num_classes):
        super(SAGPooling_Global, self).__init__()
        self.hid=args.hid
        self.pooling_ratio=args.pooling_ratio
        self.dropout=args.dropout

        self.conv1=GCNConv(num_classes,self.hid)
        self.conv2=GCNConv(self.hid,self.hid)
        self.conv3=GCNConv(self.hid,self.hid)

        self.pool=SAGPooling(self.hid*3,ratio=self.pooling_ratio,
                             GNN=GCNConv, min_score=None,
                             nonlinearity=torch.tanh)

        self.lin1=nn.Linear(self.hid*6,self.hid)
        self.lin2=nn.Linear(self.hid,self.hid//2)

        self.classifier=nn.Linear(self.hid/2,num_features)

    def forward(self,data):
        x,edge_index,batch=data.x,data.edge_index,data.batch

        x1=F.relu(self.conv1(x,edge_index))
        x2=F.relu(self.conv2(x1,edge_index))
        x3=F.relu(self.conv3(x2,edge_index))

        x=torch.cat([x1,x2,x3],dim=1)

        x,edge_index,_,batch,perm,score=self.pool(x,edge_index,None,batch)

        x=torch.cat([gmp(x,batch),gap(x,batch)],dim=1)

        x=F.relu(self.hid(x))
        x=F.dropout(x,p=self.dropout_ration,training=self.training)
        x=F.relu(self.lin2(x))

        x=self.classifier(x)

        x=F.log_softmax(x,dim=1)

        return x


class SAGPooling_Hierarchical(nn.Module):
    def __init__(self,args,num_features,num_classes):
        super(SAGPooling_Hierarchical, self).__init__()
        self.hid=args.hid
        self.pooling_ratio=args.pooling_ratio
        self.dropout_ratio=args.dropout_ratio

        self.conv1=GCNConv(num_classes,self.hid)
        self.conv2=GCNConv(self.hid,self.hid)
        self.conv3=GCNConv(self.hid,self.hid)

        self.pool=SAGPooling(self.hid,ratio=self.pooling_ratio,GNN=GCNConv,min_score=None,nonlinearity=torch.tanh)

        self.lin1=nn.Linear(self.hid*2,self.hid)
        self.lin2=nn.Linear(self.hid,self.hid//2)

        self.classifier=nn.Linear(self.hid//2,num_classes)

    def forward(self,data):
        x,edge_index,batch=data.x,data.edge_index,data.batch

        x=F.relu(self.conv1(x,edge_index))
        x,edge_index,_,batch,perm,score=self.pool(x,edge_index,None,batch)

        x1=torch.cat([gmp(x,batch),gap(x,batch)],dim=1)

        x=F.relu(self.conv2(x,edge_index))
        x,edge_index,_,batch,perm,score=self.pool(x,edge_index,None,batch)

        x2=torch.cat([gmp(x,batch),gap(x,batch)],dim=1)

        x=F.relu(self.conv3(x,edge_index))
        x,edge_index,_,batch,perm,score=self.pool(x,edge_index,None,batch)

        x3=torch.cat([gmp(x,batch),gap(x,batch)],dim=1)

        x=x1+x2+x3

        x=F.relu(self.lin1(x))
        x=F.dropout(x,p=self.dropout_ratio,training=self.training)
        x=F.relu(self.lin2(x))

        x=F.log_softmax(x,dim=1)

        return x




