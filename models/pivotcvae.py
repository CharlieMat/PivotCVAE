import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from models.cvae import BaseCVAE

'''
4 training types:
* gt: ground truth pivot
* pt: best pivot output
* sgt: sampled ground truth pivot
* spt: sampled pivot output

2 inference types:
* pi: best pivot output
* spi: sampled pivot output

Models (4 training * 2 inference):
* UserPivotCVAE: gt, pi
* UserPivotCVAE2: pt, pi
* UserPivotCVAE_PrePermute: spt, pi
* UserPivotCVAE_PrePermute2: sgt, pi
* UserPivotCVAE_PrePermute3: gt, spi
* UserPivotCVAE_PrePermute4: pt, spi
* UserPivotCVAE_PrePermute5: spt, spi
* UserPivotCVAE_PrePermute6: sgt, spi
'''


class UserPivotCVAE(BaseCVAE):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        """
        @input:
        embeddings - pretrained item embeddings
        u_embeddings - pretrained user embeddings
        slate_size - number of items in a slate
        feature_size - item embedding size
        latent_size - encoding z size
        condition_size - the condition vector size
        encoder_struct
        psm_struct - pivot selection model structure
        scm_struct - slate completion model structure
        prior_struct - prior network struture
        no_user - True if user embeddings are ignored during training/inference
        device - cpu/cuda:x
        fin_tune - True if want to fine tune item/user embedding
        """
        super(UserPivotCVAE, self).__init__(embeddings, u_embeddings, \
                                            slate_size, latent_size, no_user, device, fine_tune)
        
        # model structure is different according to whether user embedding included
        # - encoder structure: 
        if no_user:
            assert encoder_struct[0] == slate_size * feature_size + condition_size
            assert psm_struct[0] == latent_size + condition_size
            assert psm_struct[-1] == feature_size
            assert scm_struct[0] == latent_size + condition_size + feature_size
            assert scm_struct[-1] == (slate_size - 1) * feature_size
            assert prior_struct[0] == condition_size
        else:
            assert encoder_struct[0] == slate_size * feature_size + condition_size + feature_size
            assert psm_struct[0] == latent_size + condition_size + feature_size
            assert psm_struct[-1] == feature_size
            assert scm_struct[0] == latent_size + condition_size + feature_size + feature_size
            assert scm_struct[-1] == (slate_size - 1) * feature_size
            assert prior_struct[0] == condition_size + feature_size
            
#         self.slate_size = slate_size
        self.feature_size = feature_size
#         self.latent_size = latent_size
        self.condition_size = condition_size
#         self.noUser = no_user
#         self.device = device
        self.encoderStruct = encoder_struct
        self.psmStruct = psm_struct
        self.scmStruct = scm_struct
        self.priorStruct = prior_struct
        print("\tdevice: " + str(self.device))
        
#         # doc embedding
        
#         print("\tCopying document latent embedding")
#         self.docEmbed = nn.Embedding(embeddings.weight.shape[0], feature_size)
# #         self.docEmbed.weight.data.copy_(F.normalize(embeddings.weight, p = 2, dim = 1))
#         self.docEmbed.weight.data.copy_(embeddings.weight)
#         self.docEmbed.weight.requires_grad=False
#         print("\t\tDoc embedding shape: " + str(self.docEmbed.weight.shape))
#         print("\t\tDoc embedding sample: " + str(self.docEmbed.weight[0]))
        
#         # user embedding
#         if not no_user:
#             print("\tCopying user latent embedding")
#             self.userEmbed = nn.Embedding(u_embeddings.weight.shape[0], feature_size)
# #             self.userEmbed.weight.data.copy_(F.normalize(u_embeddings.weight, p = 2, dim = 1))
#             self.userEmbed.weight.data.copy_(u_embeddings.weight)
#             self.userEmbed.weight.requires_grad=False
#             print("\t\tDoc embedding shape: " + str(self.userEmbed.weight.shape))
#             print("\t\tDoc embedding sample: " + str(self.userEmbed.weight[0]))
        
        # encoder
        
        print("\tSetting up encoder")
        self.encMLP = list()
        for i in range(len(encoder_struct) - 1):
            module = nn.Linear(encoder_struct[i], encoder_struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.add_module("enc_" + str(i+1), module)
            self.encMLP.append(module)
        # encoding to latent
        self.encmu = nn.Linear(encoder_struct[-1], latent_size)
        self.enclogvar = nn.Linear(encoder_struct[-1], latent_size)
        print("\tdone")

        print("\tSetting up decoder")
        
        
        # decoder contains a pivot selection model (psm) and a slate completion model (scm)
        
        self.psmMLP = list()
        for i in range(len(psm_struct) - 1):
            module = nn.Linear(psm_struct[i], psm_struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.add_module("psm_" + str(i+1), module)
            self.psmMLP.append(module)
        print("\tdone")
        
        self.scmMLP = list()
        for i in range(len(scm_struct) - 1):
            module = nn.Linear(scm_struct[i], scm_struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.add_module("scm_" + str(i+1), module)
            self.scmMLP.append(module)
        print("\tdone")
        
        # prior
        
        print("\tSetting up prior")
#         assert params["prior_struct"][0] = condition_size
        self.priorMLP = list()
        for i in range(len(prior_struct) - 1):
            module = nn.Linear(prior_struct[i], prior_struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.add_module("prior_" + str(i+1), module)
            self.priorMLP.append(module)
        # encoding to latent
        self.priorMu = nn.Linear(prior_struct[-1], latent_size)
        self.priorLogvar = nn.Linear(prior_struct[-1], latent_size)
        print("\tdone")
        
        # GPU on/off
        print("\tMoving model to " + str(self.device))
        self.to(self.device)

    def encode(self, emb, c, u_emb = None): # Q(z|s)
        '''
        Encoder forward
        emb: (bs, slate raw features)
        c: (bs, condition_size)
        u_emb: (bs, feature_size)
        '''
        if self.noUser:
            output = torch.cat([emb, c], 1) # (bs, feature_size * slate_size + condition_size)
        else:
            output = torch.cat([emb, c, u_emb], 1)
        for i in range(len(self.encMLP)):
            output = self.relu(self.encMLP[i](output))
        z_mu = self.encmu(output)
        z_var = self.enclogvar(output)
        return z_mu, z_var

    def sample_pivot_ranking(self, z, c):
        '''
        Pivot ranking selection
        '''
        output = torch.cat([z, c], 1) # (bs, latent_size + condition_size)
        for i in range(len(self.psmMLP) - 1):
            output = self.relu(self.psmMLP[i](output))
        pivot_output = self.psmMLP[-1](output)
        torch.mm(self.docEmbed.weight, pivot_output.t())
        
    def pick_pivot(self, pivot_output, true_pivot):  
        '''
        UserPivotCVAE: ground truth pivot during training, best pivot during inference
        '''
        if len(true_pivot) == 0: # inference time, selected best pivot
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        else: # at training time, ground truth as pivot
            pivot_emb = self.docEmbed(true_pivot)  
        return pivot_emb
    
    def decode(self, z, c, u_emb = None, true_pivot = []): # P(x|z)
        '''
        Decoder
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        # pivot selection
        if self.noUser:
            output = torch.cat([z, c], 1) # (bs, latent_size + condition_size)
        else:
            output = torch.cat([z, c, u_emb], 1) # (bs, latent_size + condition_size + feature_size)
        for i in range(len(self.psmMLP) - 1):
            output = self.relu(self.psmMLP[i](output))
        pivot_output = self.psmMLP[-1](output)
        pivot_emb = self.pick_pivot(pivot_output, true_pivot)
        
        # slate completion
        if self.noUser:
            output = torch.cat([z,c,pivot_emb], 1)
        else:
            output = torch.cat([z,c,pivot_emb,u_emb], 1)
        for i in range(len(self.scmMLP) - 1):
            output = self.relu(self.scmMLP[i](output))
        output = self.scmMLP[-1](output)
        output = output.reshape((len(z), self.slate_size - 1, self.feature_size))
#         output = output.transpose(0,1) + pivot_emb
#         output = output.transpose(0,1)
        output = torch.cat([pivot_emb.reshape((len(z), 1, self.feature_size)), output], 1)
#         output = torch.cat([pivot_output.reshape((len(z), 1, self.feature_size)), output], 1)
        
        return output
    
    def get_prior(self, r, u = None):
        cond = self.get_condition(r)
        if self.noUser:
            output = cond
        else:
            uEmb = self.userEmbed(u.reshape(-1))
            output = torch.cat([cond,uEmb], 1)
        for i in range(len(self.priorMLP)):
            output = self.relu(self.priorMLP[i](output))
        prior_mu = self.priorMu(output)
        prior_logvar = self.priorLogvar(output)
        return prior_mu, prior_logvar

    def forward(self, s, r, candidates = None, u = None):
        """
        Encoder-decoder forward
        s: (bs, slate items)
        r: (bs, slate responses)
        candidates: (bs, slate size, #candidate)
        u (bs,)
        """
        cond = self.get_condition(r)
        # get embedding of shape (batch_size, emb_size * slate_size)
        originalShape = s.shape
        emb = self.docEmbed(s.reshape(-1)).view((originalShape[0], -1))
        # encoder and decoder forward
        if self.noUser:
            uEmb = None
        else:
            uEmb = self.userEmbed(u.reshape(-1)).view((originalShape[0], -1))
        z_mu, z_logvar = self.encode(emb, cond, uEmb)
        z = self.reparametrize(z_mu, z_logvar)
        rx = self.decode(z, cond, u_emb = uEmb, true_pivot = s[:,0])
        # get ranking score for all doc
        prox_emb = rx.reshape(-1, self.feature_size)
        # only find items from candidate set for each position
        if self.candidateFlag:
            # BS * slate_size * #candidate
            nCandidate = candidates.shape[-1]
            # (BS * slate_size, #candidate, dim)
            candidateEmb = self.docEmbed(candidates).view((-1,nCandidate,self.feature_size))
            p = torch.bmm(candidateEmb, prox_emb.view((-1,self.feature_size,1))).view((-1,nCandidate))
            return p, rx, z, emb, z_mu, z_logvar
        # entire itemset as candidate set
        else:
            p = torch.mm(prox_emb.view(-1,self.feature_size), self.docEmbed.weight.t())
#             return prox_emb, rx, z, emb, z_mu, z_logvar
            return p, rx, z, emb, z_mu, z_logvar

    def recommend(self, r, u = None, return_item = False, random_pivot = False):
        cond = self.get_condition(r)
        if self.noUser:
            uEmb = None
            output = cond
        else:
            uEmb = self.userEmbed(u.reshape(-1))
            output = torch.cat([cond, uEmb], 1)
        for i in range(len(self.priorMLP)):
            output = self.relu(self.priorMLP[i](output))
        z_mu = self.priorMu(output)
        z_logvar = self.priorLogvar(output)
        z = self.reparametrize(z_mu, z_logvar)
        rx = self.decode(z, cond, uEmb)
        if return_item:
            recItems = self.get_recommended_item(rx.view((-1, self.feature_size)))
            return recItems, z_mu
        else:
            return rx, z_mu
    
    def log(self, logger):
        logger.log("\tfeature size: " + str(self.feature_size))
        logger.log("\tslate size: " + str(self.slate_size))
        logger.log("\tz size: " + str(self.latent_size))
        logger.log("\tcondition size: " + str(self.condition_size))
        logger.log("\tuser is ignored: " + str(self.noUser))
        logger.log("\tencoder struct: " + str(self.encoderStruct))
        logger.log("\tpsm struct: " + str(self.psmStruct))
        logger.log("\tscm struct: " + str(self.scmStruct))
        logger.log("\tprior struct: " + str(self.priorStruct))
        logger.log("\tdevice: " + str(self.device))

class UserPivotCVAE2(UserPivotCVAE):
    '''
    UserPivotCVAE2: best pivot during training, best pivot during inference
    '''
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE2, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot):
        if len(true_pivot) == 0:
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        else:
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        return pivot_emb
'''
* UserPivotCVAE2: best pivot during training, best pivot during inference
'''
        
class UserPivotCVAE_PrePermute(UserPivotCVAE):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot = []):
        '''
        UserPivot_PrePermute: sampled pivot during training, best pivot during inference
        '''
        if len(true_pivot) == 0: # inference time, select best pivot
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        else: # at training time, sampled pivot
            p = self.sigmoid(torch.mm(pivot_output, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        return pivot_emb
    
class UserPivotCVAE_PrePermute2(UserPivotCVAE):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute2, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot = []):
        '''
        UserPivot_PrePermute2: sampled ground truth during training, best pivot during inference
        '''
        if len(true_pivot) == 0:
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        else:
            gt_pivot_emb = self.docEmbed(true_pivot)
            p = self.sigmoid(torch.mm(gt_pivot_emb, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        return pivot_emb
    
class UserPivotCVAE_PrePermute3(UserPivotCVAE):
    '''
    UserPivot_PrePermute3: ground truth pivot during training, sampled pivot during inference
    '''
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute3, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot = []):
        if len(true_pivot) == 0: # inference time, sample pivot
            p = self.sigmoid(torch.mm(pivot_output, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        else: # at training time, sampled ground truth as pivot
            pivot_emb = self.docEmbed(true_pivot) 
        return pivot_emb
        
class UserPivotCVAE_PrePermute4(UserPivotCVAE):
    '''
    UserPivot_PrePermute4: best pivot during training, sampled pivot during inference
    '''
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute4, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot):
        if len(true_pivot) == 0:
            p = self.sigmoid(torch.mm(pivot_output, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        else:
            p = torch.mm(self.docEmbed.weight, pivot_output.t()).max(0)[1]
            pivot_emb = self.docEmbed(p)
        return pivot_emb
    
class UserPivotCVAE_PrePermute5(UserPivotCVAE):
    '''
    UserPivot_PrePermute5: sampled pivot during both training and inference
    '''
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute5, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot):
        p = self.sigmoid(torch.mm(pivot_output, self.docEmbed.weight.t()))
        samp = Categorical(p).sample()
        pivot_emb = self.docEmbed(samp)
        return pivot_emb
    
class UserPivotCVAE_PrePermute6(UserPivotCVAE):
    '''
    UserPivot_PrePermute6: sampled ground truth during training and sampled pivot during inference
    '''
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False):
        super(UserPivotCVAE_PrePermute6, self).__init__(embeddings, u_embeddings, \
                             slate_size, feature_size, latent_size, condition_size, \
                             encoder_struct, psm_struct, scm_struct, prior_struct, no_user, device, fine_tune = False)

    def pick_pivot(self, pivot_output, true_pivot):
        if len(true_pivot) == 0:
            p = self.sigmoid(torch.mm(pivot_output, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        else:
            gt_pivot_emb = self.docEmbed(true_pivot)
            p = self.sigmoid(torch.mm(gt_pivot_emb, self.docEmbed.weight.t()))
            samp = Categorical(p).sample()
            pivot_emb = self.docEmbed(samp)
        return pivot_emb

    
PIVOTCVAE_MODELS = {"pivotcvae_gt_pi": UserPivotCVAE, "pivotcvae_pt_pi": UserPivotCVAE2, \
                    "pivotcvae_spt_pi": UserPivotCVAE_PrePermute, "pivotcvae_sgt_pi": UserPivotCVAE_PrePermute2, \
                    "pivotcvae_gt_spi": UserPivotCVAE_PrePermute3, "pivotcvae_pt_spi": UserPivotCVAE_PrePermute4, \
                    "pivotcvae_spt_spi": UserPivotCVAE_PrePermute5, "pivotcvae_sgt_spi": UserPivotCVAE_PrePermute6} 