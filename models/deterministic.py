import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from heapq import heappush, heappop
from tqdm import tqdm
import hyperparams

class RankModel(nn.Module):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True):
        """
        @input:
        - embeddings: pretrained item embeddings
        - u_embeddings: pretrained user embeddings
        - slate_size: number of items in a slate
        - no_user: true if user embeddings are ignored during training/inference
        - device: cpu/cuda:x
        - fine_tune: true if want to fine tuning item/user embedding
        """
        super(RankModel, self).__init__()
        assert embeddings.weight.shape[1] == feature_size
        assert u_embeddings.weight.shape[1] == feature_size
        self.candidateFlag = False
        self.slate_size = slate_size
        self.feature_size = feature_size
        self.device = device
        print("\tdevice: " + str(self.device))
        
        with torch.no_grad():
            # doc embedding
            print("\tLoad pretrained document latent embedding")
            self.docEmbed = nn.Embedding(embeddings.weight.shape[0], embeddings.weight.shape[1])
            self.docEmbed.weight.data.copy_(F.normalize(embeddings.weight, p = 2, dim = 1))
    #         self.docEmbed.weight.data.copy_(embeddings.weight)
            self.docEmbed.weight.requires_grad=fine_tune
            print("\t\tDoc embedding shape: " + str(self.docEmbed.weight.shape))

            # user embedding
            print("\tCopying user latent embedding")
            self.userEmbed = nn.Embedding(u_embeddings.weight.shape[0], u_embeddings.weight.shape[1])
            self.userEmbed.weight.data.copy_(F.normalize(u_embeddings.weight, p = 2, dim = 1))
#             self.userEmbed.weight.data.copy_(u_embeddings.weight)
            self.userEmbed.weight.requires_grad=fine_tune
            print("\t\tUser embedding shape: " + str(self.userEmbed.weight.shape))
            
        self.m = nn.Sigmoid()
    
    def point_forward(self, users, items):
        raise NotImplemented
    
    def forward(self, s, r, candidates = None, u = None):
        pred = torch.zeros_like(s).to(torch.float)
        for i in range(self.slate_size):
            pred[:,i] = self.point_forward(u, s[:,i])
        return pred
        
    def recommend(self, r, u = None, return_item = False):
        raise NotImplemented
    
    def get_recommended_item(self, embeddings):
        candidateEmb = self.docEmbed.weight.data.view((-1,self.feature_size))
        p = torch.mm(embeddings, candidateEmb.t())
        values, indices = torch.max(p,1)
        return indices
        
    def log(self, logger):
        logger.log("\tfeature size: " + str(self.feature_size))
        logger.log("\tslate size: " + str(self.slate_size))
        logger.log("\tdevice: " + str(self.device))

class MF(RankModel):
    """
    Biased MF
    """
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize MF framework...")
        super(MF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        
        print("Initialize user and item biases")
        # user bias
        self.userBias = nn.Embedding(self.userEmbed.weight.shape[0], 1)
        self.userBias.weight.data.fill_(0.001)
        
        # item bias
        self.docBias = nn.Embedding(self.docEmbed.weight.shape[0], 1)
        self.docBias.weight.data.fill_(0.001)
        
        print("Done.")
        
    def point_forward(self, users, items):
        """
        forward pass of MF
        """
        # extract latent embeddings
        uE = self.userEmbed(users.view(-1))
        uB = self.userBias(users.view(-1))
        iE = self.docEmbed(items.view(-1))
        iB = self.docBias(items.view(-1))

        # positive example
        output = torch.mul(uE,iE)\
                            .sum(1).view(-1,1)
        output = output + uB + iB
        return output.view(-1)
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i] = self.point_forward(u[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None

class DiverseMF(MF):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True, diversity_alpha = 0.5):
        print("Initialize MF framework...")
        super(DiverseMF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        self.alpha = diversity_alpha
    
    def item_similarity(self, items):
        iE = self.docEmbed(items.view(-1))
        iB = self.docBias(items.view(-1))
        
        item_sim,_ = torch.max(torch.matmul(iE,iE.transpose(0,1)),1)       
        return item_sim
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i] = self.point_forward(u[i], candItems)
        
        item_sim = self.item_similarity(candItems)
        mmr = self.alpha*p - (1-self.alpha)*item_sim
        
        _, recItems = torch.topk(mmr, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None


class NeuMF(RankModel):
    """
    Biased MF
    """
    def __init__(self, embeddings, u_embeddings, struct, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize NeuMF model...")
        super(NeuMF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        # MLP structure hyperparams
        self.structure = struct
        assert len(self.structure) >= 2
        assert self.structure[0] == 2 * feature_size
        
        # GMF part
        self.gmf = nn.Linear(feature_size, 1)
        
        print("\tCreating embedding")
        # MLP user embedding
        self.userMLPEmbed = nn.Embedding(u_embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.userMLPEmbed.weight)
        # MLP item embedding
        self.docMLPEmbed = nn.Embedding(embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.docMLPEmbed.weight)

        # setup model structure
        print("\tCreating predictive model")
        # mlp part
        self.mlp_modules = list()
        for i in range(len(self.structure) - 1):
            module = nn.Linear(self.structure[i], self.structure[i+1])
            torch.nn.init.kaiming_uniform_(module.weight)
            self.mlp_modules.append(module)
            self.add_module("mlp_" + str(i), module)
        # final output
        self.outAlpha = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad = True)

    def point_forward(self, users, items):
        # user embeddings
        uME = self.userMLPEmbed(users.view(-1))
        uGE = self.userEmbed(users.view(-1))
        # item embeddings
        iME = self.docMLPEmbed(items.view(-1))
        iGE = self.docEmbed(items.view(-1))
        # GMF output
        gmfOutput = self.gmf(torch.mul(uGE, iGE))
        # MLP output
        X = torch.cat((uME, iME), 1)
        for i in range(len(self.mlp_modules) - 1):
            layer = self.mlp_modules[i]
            X = F.relu(layer(X))
        mlpOutput = self.mlp_modules[-1](X)
        # Integrate GMF and MLP outputs
        out = self.outAlpha * gmfOutput + (1 - self.outAlpha) * mlpOutput
        return out.view(-1)
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        users = torch.ones(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i] = self.point_forward(users * u.view(-1)[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None