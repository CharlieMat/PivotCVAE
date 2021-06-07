import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from models.cvae import BaseCVAE

class UserListCVAEWithPrior(BaseCVAE):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, latent_size, condition_size, \
                 encoder_struct, decoder_struct, prior_struct, no_user, device, fine_tune = False):
        """
        @input:
        embeddings - pretrained item embeddings
        u_embeddings - pretrained user embeddings
        slate_size - number of items in a slate
        feature_size - item embedding size
        latent_size - encoding z size
        condition_size - the condition vector size
        encoder_struct - encoder MLP structure
        decoder_struct - decoder MLP structure
        prior_struct - prior network MLP structure
        no_user - true if user embeddings are ignored during training/inference
        device - cpu/cuda:x
        fine_tune - true if want to fine tuning item/user embedding
        """
        super(UserListCVAEWithPrior, self).__init__(embeddings, u_embeddings, \
                                                    slate_size, latent_size, no_user, device, fine_tune)
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.encoderStruct = encoder_struct
        self.decoderStruct = decoder_struct
        self.priorStruct = prior_struct
        
        if no_user:
            assert encoder_struct[0] == slate_size * feature_size + condition_size
            assert decoder_struct[0] == latent_size + condition_size
            assert decoder_struct[-1] == slate_size * feature_size
            assert prior_struct[0] == condition_size
        else:
            assert encoder_struct[0] == slate_size * feature_size + condition_size + feature_size
            assert decoder_struct[0] == latent_size + condition_size + feature_size
            assert decoder_struct[-1] == slate_size * feature_size
            assert prior_struct[0] == condition_size + feature_size
        
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

        # decoder
        
        print("\tSetting up decoder")
        self.decMLP = list()
        for i in range(len(decoder_struct) - 1):
            module = nn.Linear(decoder_struct[i], decoder_struct[i+1])
            nn.init.kaiming_uniform_(module.weight)
            self.add_module("dec_" + str(i+1), module)
            self.decMLP.append(module)
        print("\tdone")
        
        # prior
        
        print("\tSetting up prior")
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
        c: (bs, condition size)
        '''
        if self.noUser:
            output = torch.cat([emb, c], 1) # (bs, embedding_size + condition_size)
        else:
            output = torch.cat([emb, c, u_emb], 1) # (bs, embedding_size + condition_size + user_embedding_size)
        for i in range(len(self.encMLP)):
            output = self.relu(self.encMLP[i](output))
        z_mu = self.encmu(output)
        z_var = self.enclogvar(output)
        return z_mu, z_var

    def decode(self, z, c, u_emb = None): # P(x|z)
        '''
        Decoder
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        if self.noUser:
            output = torch.cat([z, c], 1) # (bs, latent_size + condition_size)
        else:
            output = torch.cat([z, c, u_emb], 1) # (bs, latent_size + condition_size + feature_size)
        for i in range(len(self.decMLP) - 1):
            output = self.relu(self.decMLP[i](output))
        output = self.decMLP[-1](output)
        return output
    
    def get_prior(self, r, u = None):
        cond = self.get_condition(r)
        if self.noUser:
            output = cond
        else:
            uEmb = self.userEmbed(u.reshape(-1))
            output = torch.cat([cond,uEmb],1)
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
        condidates: (bs, slate size, #candidate)
        u: (bs, )
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
        rx = self.decode(z, cond, uEmb)
        # reconstructed slate embedding, (BS * slate_size, dim)
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
            p = torch.mm(prox_emb, self.docEmbed.weight.t())
#             return prox_emb, rx, z, emb, z_mu, z_logvar
            return p, rx, z, emb, z_mu, z_logvar

    def recommend(self, r, u = None, return_item = False):
        cond = self.get_condition(r)
        if self.noUser:
            output = cond
            uEmb = None
        else:
            uEmb = self.userEmbed(u.reshape(-1))
            output = torch.cat([cond,uEmb],1)
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
        logger.log("\tdecoder struct: " + str(self.decoderStruct))
        logger.log("\tprior struct: " + str(self.priorStruct))
        logger.log("\tdevice: " + str(self.device))