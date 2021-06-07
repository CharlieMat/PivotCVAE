import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm

def sample_users(environment, batch_size):
    up = torch.ones(environment.maxUserId + 1)
    sampledU = torch.multinomial(up, batch_size, replacement = True).reshape(-1).to(environment.device)
    return sampledU

class Environment(nn.Module):
    def __init__(self, maxIID, maxUID, f_size, s_size, device, no_user):
        super(Environment, self).__init__()
        self.maxItemId = maxIID
        self.maxUserId = maxUID
        self.featureSize = f_size
        self.slateSize = s_size
        # device: gpu / cpu
        self.device = device
        print("\tdevice: " + str(self.device))
        # True if ignore user
        self.noUser = no_user
        # doc embedding
        print("\tCreating document latent embedding")
        a = math.sqrt(2.0 / f_size)
        self.docEmbed = nn.Embedding(maxIID + 1, f_size)
        self.docEmbed.weight.data.uniform_(-a,a)
        print("\t\tDoc embedding sample: " + str(self.docEmbed.weight[0]))
        if not no_user:
            # user embedding
            print("\tCreating user latent embedding")
            self.userEmbed = nn.Embedding(maxUID + 1, f_size)
            self.userEmbed.weight.data.uniform_(-a,a)
            print("\t\tUser embedding sample: " + str(self.userEmbed.weight[0]))
        
##################################################################
#                  Response Model for Datasets                   #
#   For offline training, log datasets cannot provide responses  #
# for  unseen  slates.  Learn  a  URM  based on the log data can #
# can provide this information.                                  #
##################################################################

class UserResponseModel_MLP(Environment):
    """
    User's response model for slates
    """
    def __init__(self, maxIID, maxUID, f_size, s_size, struct, device, no_user):
        """
        @input:
        - maxIId: maximum item ID
        - maxUId: maximum user ID
        - f_size: latent feature size
        - s_size: number of item in a slate
        - struct: mlp structure of prediction model
        - device: "cpu","cuda:0",etc.
        """
        super(UserResponseModel_MLP, self).__init__(maxIID, maxUID, f_size, s_size, device, no_user)
        
        # mlp structure
        if no_user:
            assert struct[0] == s_size * f_size
        else:
            assert struct[0] == (s_size + 1) * f_size
        assert struct[-1] == s_size
        self.mlp = list()
        for i in range(len(struct)-1):
            module = nn.Linear(struct[i], struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.mlp.append(module)
            self.add_module("mlp_" + str(i+1), module)
        
    def forward(self, slates, users):
        # get embedding
        dEmb = F.normalize(self.docEmbed(slates).reshape((slates.shape[0], -1)), p = 2, dim = 1)
        if self.noUser:
            output = dEmb
        else:
            uEmb = F.normalize(self.userEmbed(users), p = 2, dim = 1).reshape((slates.shape[0], -1))
            output = torch.cat([dEmb, uEmb], 1)
        for i in range(len(self.mlp)-1):
            output = F.relu(self.mlp[i](output))
        output = self.mlp[-1](output)
        return output
    
    
##################################################################
#                  Response Model for Simulation                 #
#   Simulation is used to check the effectiveness of the learned #
# user response model. Generative model will be tested on both   #
# learned URM and the simulator to observe the difference.       #
##################################################################

class URM(Environment):
    """
    User's response model for slates
    Only contains a matrix factorization between user and item
    Resulting user responses = p(r | user, item) only depends on the item the user is looking at
    """
    def __init__(self, maxIID, maxUID, slate_size, latent_size, device, no_user):
        super(URM, self).__init__(maxIID, maxUID, latent_size, slate_size, device, no_user)
        # must have user
        assert not no_user
        
        # max item id and user id
        
        self.maxIID = torch.tensor(maxIID).to(self.device)
        self.maxUID = torch.tensor(maxUID).to(self.device)
        
        # item bias
        self.itemBias = nn.Embedding(maxIID + 1, 1)
        self.itemBias.weight.data = torch.zeros_like(self.itemBias.weight.data)
        
        # user bias
        self.userBias = nn.Embedding(maxUID + 1, 1)
        self.userBias.weight.data = torch.zeros_like(self.userBias.weight.data)
        
        # output to probability
        self.m = nn.Sigmoid()
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.device = args[0]
        return self
    
    def core_forward(self, slates, users):
        """
        This is the first step of user response simulation before multinomial sampling
        @input:
        - slates: 
        @output:
         - p: probability of click
         - 
        """
        B,_ = slates.shape
        # get embedding (B, slateSize, dim)
        dEmb = F.normalize(self.docEmbed(slates), p = 2, dim = -1).view(B,self.slateSize,-1)
        # get item bias (B, slateSize)
        dBias = self.itemBias(slates).view(B, self.slateSize)
        # get user embedding (B, slateSize)
        uEmb = F.normalize(self.userEmbed(users.view(B)), p = 2, dim = -1)
        uEmb = self.userEmbed(users.view(B))
        uBias = self.userBias(users.view(B))
        output = torch.bmm(dEmb, uEmb.view(B, self.featureSize, 1)).view(B,self.slateSize) + dBias
        output = output.transpose(0,1) + uBias.view(-1)
        output = output.transpose(0,1)
        return self.m(output), dEmb, dBias, uEmb, uBias
        
    def forward(self, slates, users):
        p, dEmb, dBias, uEmb, uBias = self.core_forward(slates, users)
        return p
    
    def sample_response(self, slate_p):
        """
        Sample user click responses by binomial distribution
        Not using multinomial because: 
        - to be able to sample all zeros; 
        - multinomial may need another sampling for number of trials
        @input:
        - slate_p: the probability of click
        """
        slate_p[slate_p >= 0.5] = 1.0
        slate_p[slate_p < 0.5] = 0.0
#         m = Bernoulli(slate_p)
#         return m.sample()
        return slate_p
    
    def generate_response_for_dataset(self, sampledU, sampledSlates):
        L = len(sampledU)
        sampledR = torch.zeros_like(sampledSlates)
        pbar = tqdm(total = L)
        batchSize = 128
        loc = 0
        while loc <= L:
            end = min(loc + batchSize, L)
            batchUsers = sampledU[loc:end]
            batchSlates = sampledSlates[loc:end]
            p, dEmb, dBias, uEmb, uBias = self.core_forward(batchSlates, batchUsers)
            sampledR[loc:end,:] = self.sample_response(p.reshape(-1,self.slateSize))
            pbar.update(end - loc)
            loc = loc + batchSize
        pbar.close()
        return sampledR
    
    def generate_dataset(self, min_user_hist = 20, min_item_hist = 20, n_record = 1000000):
        """
        Generating dummy dataset, each sample of type (user, slate(list of item), slate responses)
        @input:
         - min_user_hist: minimum number of slates(not item) of each user
         - min_item_hist: minimum number of users of each item(not slate)
         - n_record: number of samples to generate, must be larger than or equals to \
                 (min_user_hist * nUsers + min_item_hist * nItems)
        """
        
        # the number of records is controlled by all three inputs
        assert min_user_hist * self.maxUID + min_item_hist * self.maxIID < n_record
        n_record = max(max(n_record, (self.maxUID + 1) * min_user_hist), (self.maxIID + 1) * min_item_hist)
        
        # init empty dataset
        genUList = torch.zeros((n_record)).to(torch.long).to(self.device)
        genSList = torch.zeros((n_record, self.slateSize)).to(torch.long).to(self.device)
        genRList = torch.zeros((n_record, self.slateSize)).to(torch.float).to(self.device)
        
        # keep track of how many data points have been generated
        offset = 0
        
        # sampling probability of users and items
        up = torch.ones(self.maxUID + 1)
        ip = torch.ones(self.maxIID + 1)
        
        # guarantee min_user_hist requirement
        with torch.no_grad():
            if min_user_hist > 0:
                print("Guarantee min_user_hist requirement:")
                sampledU = torch.arange(self.maxUID + 1).reshape(-1,1).repeat(1,1,min_user_hist).reshape(-1,1).to(self.device)
                sampledSlates = torch.multinomial(ip, (self.maxUID + 1) * self.slateSize * min_user_hist, replacement = True)\
                    .reshape(-1,self.slateSize).to(self.device)
                sampledR = self.generate_response_for_dataset(sampledU, sampledSlates)
                # inject into datasets
                L = len(sampledU)
                genUList[offset:offset+L] = sampledU[:,0]
                genSList[offset:offset+L, :] = sampledSlates
                genRList[offset:offset+L, :] = sampledR
                offset = offset + L

            # guarantee min_item_hist requirement
            if min_item_hist > 0:
                print("Guarantee min_item_hist requirement:")
                sampledU = torch.multinomial(up, (self.maxIID + 1) * min_item_hist, replacement = True).reshape(-1,1).to(self.device)
                sampledSlates = torch.multinomial(ip, (self.maxIID + 1) * self.slateSize * min_item_hist, replacement = True)\
                    .reshape(-1,min_item_hist,self.slateSize).to(self.device)
                allI = torch.arange(self.maxIID + 1)
                for i in range(min_item_hist):
                    sampledSlates[:,i,0] = allI
                sampledSlates = sampledSlates.reshape((-1, self.slateSize))
                sampledR = self.generate_response_for_dataset(sampledU, sampledSlates)
                # inject into datasets
                L = len(sampledU)
                genUList[offset:offset+L] = sampledU[:,0]
                genSList[offset:offset+L, :] = sampledSlates
                genRList[offset:offset+L, :] = sampledR
                offset = offset + L

            # generate the rest of the record to fulfill n_record requirement
            print("Generate the remaining data")
            if offset < n_record:
                sampledU = torch.multinomial(up, n_record - offset, replacement = True).reshape(-1,1).to(self.device)
                sampledSlates = torch.multinomial(ip, (n_record - offset) * self.slateSize, replacement = True)\
                        .reshape(-1, self.slateSize).to(self.device)
            sampledR = self.generate_response_for_dataset(sampledU, sampledSlates)
            L = len(sampledU)
            genUList[offset:] = sampledU[:,0]
            genSList[offset:, :] = sampledSlates
            genRList[offset:, :] = sampledR
        
        respCount = torch.sum(sampledR, dim = 1).detach().cpu().numpy()
        print("Number of click distribution: " + str(np.unique(respCount, return_counts = True)))
        return (genUList.detach().cpu().numpy(), genSList.detach().cpu().numpy(), genRList.detach().cpu().numpy())


class URM_P(URM):
    """
    User's response model for slates
    Compare to URM, positional bias is added
    Resulting user responses = p(slate_r | user, slate items, positional bias)
    """
    def __init__(self, maxIID, maxUID, slate_size, latent_size, device, no_user, \
                p_bias_max, p_bias_min):
        super(URM_P, self).__init__(maxIID, maxUID, slate_size, latent_size, device, no_user)
        self.p_bias_max = p_bias_max
        self.p_bias_min = p_bias_min
        
        # independent positional bias
        self.posBias = torch.tensor(\
                            [p_bias_max - i * (p_bias_max - p_bias_min) / slate_size \
                             for i in range(slate_size)]).to(self.device)
        # positional bias that depends on user
        a = math.sqrt(0.5 / latent_size) # about 1/4 of the volumn of user/item latent
        self.posDependentBias = torch.FloatTensor(\
            slate_size * latent_size).uniform_(-a,a)\
            .reshape(slate_size, latent_size).to(self.device)
    
    def core_forward(self, slates, users):
        if self.noUser:
            users = sample_users(self, slates.shape[0])
        # independent item score
        rawScore, dEmb, dBias, uEmb, uBias = super(URM_P, self).core_forward(slates, users)
        # positional bias
        posBiases = torch.mm(uEmb.view(-1, self.featureSize), self.posDependentBias.view(self.featureSize, self.slateSize))
        posBiases = posBiases + self.posBias.view(-1)
        finalScore = rawScore.reshape(-1, self.slateSize) + posBiases
        return finalScore, dEmb, dBias, uEmb, uBias
        
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.posBias = self.posBias.to(args[0])
        self.posDependentBias = self.posDependentBias.to(args[0])
        return self


class URM_P_MR(URM_P):
    """
    User's response model for slates
    Compare to URM_P, more complecated item relations can be added, here we only applied a slate bias computed by the average of item embedding, and the slate bias is used as the attention of the user for each of the items 
    Resulting user responses = p(slate_r | user, slate items, positional bias), where the item relations are contained inside the slate.
    """
    def __init__(self, maxIID, maxUID, slate_size, latent_size, device, no_user, p_bias_max, p_bias_min, mr_factor):
        super(URM_P_MR, self).__init__(maxIID, maxUID, slate_size, latent_size, device, no_user, p_bias_max, p_bias_min)
        self.mrFactor = mr_factor
    
    def core_forward(self, slates, users):
        if self.noUser:
            users = sample_users(self, slates.shape[0])
        rawScore, dEmb, dBias, uEmb, uBias = super(URM_P_MR, self).core_forward(slates, users)
        slateAttention = self.m(torch.mean(dEmb.view(-1, self.slateSize, self.featureSize), dim = 1))
        relationOffset = torch.bmm(dEmb.view(-1, self.slateSize, self.featureSize), slateAttention.view(-1,self.featureSize,1))\
                            .view(-1,self.slateSize)
        finalScore = rawScore + relationOffset * self.mrFactor
        return finalScore, dEmb, dBias, uEmb, uBias
