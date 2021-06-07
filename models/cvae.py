import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

class BaseCVAE(nn.Module):    
    def __init__(self, embeddings, u_embeddings, slate_size, latent_size, 
                 no_user, device, fine_tune = False):
        """
        @input:
        - embeddings: pretrained item embeddings
        - u_embeddings: pretrained user embeddings
        - slate_size: number of items in a slate
        - no_user: true if user embeddings are ignored during training/inference
        - device: cpu/cuda:x
        - fine_tune: true if want to fine tuning item/user embedding
        """
        super(BaseCVAE, self).__init__()
        self.candidateFlag = False    # in forward, it can predict scores for either candidate items or all items
        self.slate_size = slate_size
        self.latent_size = latent_size
        self.noUser = no_user
        self.device = device
        print("\tdevice: " + str(self.device))
        
        with torch.no_grad():
            # doc embedding
            print("\tLoad pretrained document latent embedding")
            self.docEmbed = nn.Embedding(embeddings.weight.shape[0], embeddings.weight.shape[1])
            self.docEmbed.weight.data.copy_(F.normalize(embeddings.weight, p = 2, dim = 1))
            self.docEmbed.weight.requires_grad=fine_tune
            print("\t\tDoc embedding shape: " + str(self.docEmbed.weight.shape))

            if not no_user:
                # user embedding
                print("\tCopying user latent embedding")
                self.userEmbed = nn.Embedding(u_embeddings.weight.shape[0], u_embeddings.weight.shape[1])
                self.userEmbed.weight.data.copy_(F.normalize(u_embeddings.weight, p = 2, dim = 1))
                self.userEmbed.weight.requires_grad=fine_tune
                print("\t\tUser embedding shape: " + str(self.userEmbed.weight.shape))
            
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def encode(self, emb, c, u_emb = None): # Q(z|s)
        '''
        Encoder forward
        emb: (bs, slate raw features)
        c: (bs, condition size)
        '''
        raise NotImplemented
    
    def decode(self, z, c, u_emb = None): # P(x|z)
        '''
        Decoder
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        raise NotImplemented
    
    def get_prior(self, r, u = None):
        raise NotImplemented

#     def set_candidate(self, flag = False):
#         self.candidateFlag = flag
        
    def forward(self, s, r, candidates = None, u = None):
        """
        Encoder-decoder forward
        s: (bs, slate items)
        r: (bs, slate responses)
        condidates: (bs, slate size, #candidate)
        u: (bs, )
        """
        raise NotImplemented

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std) + mu
        return z

    def get_condition(self, r):
        """
        get condition vector
        """
        # condition is the number of clicks as onehot
        condition = torch.zeros(len(r),self.slate_size + 1).to(self.device)
        condition = condition.scatter_(1,torch.sum(r, dim=1).reshape(-1,1).to(torch.long),1)
        return condition

    def recommend(self, r, u = None, return_item = False):
        raise NotImplemented
    
    def get_recommended_item(self, embeddings):
        candidateEmb = self.docEmbed.weight.data.view((-1,self.feature_size))
        p = torch.mm(embeddings, candidateEmb.t())
        values, indices = torch.max(p,1)
        return indices
    
    def sample_encoding(self, s, r, u = None):
        cond = self.get_condition(r)
        # get embedding of shape (batch_size, emb_size * slate_size)
        originalShape = s.shape
        emb = self.docEmbed(s.reshape(-1)).reshape((originalShape[0], -1))
        if self.noUser:
            uEmb = None
            z_mu, z_logvar = self.encode(emb, cond)
        else:
            uEmb = self.userEmbed(u.reshape(-1)).reshape((originalShape[0], -1))
            # encoder forward
            z_mu, z_logvar = self.encode(emb, cond, uEmb)
        return z_mu, z_logvar
    
    def log(self, logger):
        raise NotImplemented
