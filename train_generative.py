import argparse
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import time as timer

import data_extract as dae
from data_loader import UserSlateResponseDataset
from env.response_model import UserResponseModel_MLP, sample_users
from models.listcvae import UserListCVAEWithPrior
from models.pivotcvae import PIVOTCVAE_MODELS
import my_utils as utils
import settings



###################################################
#                train simulation                 #
###################################################

def train_simluation(args):
    return

###########################################
#                training                 #
###########################################

def downsample(pred, slate, n_neg = 1000.0):
    mask = torch.zeros_like(pred, device = pred.device)
    mask.scatter_(1,slate.reshape(-1,1),1)
    neg_sample = torch.bernoulli(torch.ones_like(pred) * (n_neg / pred.shape[1]))
    mask = mask + neg_sample
    mask[mask == 2] = 1
    return pred * mask

def get_gen_loss(batch_data, model, lossFun, beta, n_neg = 1000):
    # get input and target and forward
    slates = torch.LongTensor(batch_data["slates"]).to(model.device)
    users = torch.LongTensor(batch_data["users"]).to(model.device)
    targets = torch.tensor(batch_data["responses"]).to(torch.float).to(model.device)
    pMu, pLogvar = model.get_prior(targets, users)

    # loss
    if model.candidateFlag:
        sampleCandidates = torch.LongTensor(batch_data["sample_candidates"]).to(model.device)
        pred, rSlates, z, emb, mu, logvar = model.forward(slates, targets, candidates = sampleCandidates, u = users)
        sampleTargets = torch.LongTensor(batch_data["sample_targets"]).to(model.device)
        recLoss = lossFun(pred, sampleTargets.reshape(-1))
    else:
        pred, rSlates, z, emb, mu, logvar = model.forward(slates, targets, u = users)
        recLoss = lossFun(downsample(pred, slates, n_neg = n_neg), slates.reshape(-1))
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = - 0.5 * torch.sum(1 + logvar - pLogvar - (logvar.exp() + (mu - pMu).pow(2)) / pLogvar.exp())
#     KLD = - 0.5 * torch.sum(1 - logvar + pLogvar - (pLogvar.exp() + (mu - pMu).pow(2)) / logvar.exp())
    loss = recLoss + beta * KLD
    
    return loss, recLoss, KLD

def train_on_dataset(trainset, valset, model, model_path, logger, resp_model, \
                    bs, epochs, lr, decay, beta):
    '''
    @input:
    - trainset and valset: data_loader.UserSlateResponseDataset
    - f_size: embedding size for item and user
    - s_size: slate size
    - model: generative model (list_cvae_with_prior, slate_cvae)
    - bs: batch size
    - epochs: number of epoch
    - lr: learning rate
    - decay: weight decay
    - beta: trade-off between reconstruction loss and KLD
    '''
    
    logger.log("----------------------------------------")
    logger.log("Train user response model as simulator")
    logger.log("\tbatch size: " + str(bs))
    logger.log("\tnumber of epoch: " + str(epochs))
    logger.log("\tlearning rate: " + str(lr))
    logger.log("\tweight decay: " + str(decay))
    logger.log("\tbeta: " + str(beta))
    logger.log("----------------------------------------")
    model.log(logger)
    logger.log("----------------------------------------")

    # data loaders
    trainLoader = DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 0)
    valLoader = DataLoader(valset, batch_size = bs, shuffle = False, num_workers = 0)
    speedupTrain = False #trainset.sampleSpeedup
    speedupVal = False #valset.sampleSpeedup
    

    # loss function and optimizer
    BCE = nn.BCELoss()
    m = nn.Sigmoid()
    optimizer = opt.Adam(model.parameters(), lr = lr)
    
    CEL = nn.CrossEntropyLoss()
    
    runningLoss = []  # step loss history
    trainHistory = [] # epoch training loss
    valHistory = []   # epoch validation loss
    bestLoss = np.float("inf")
    bestValLoss = np.float("inf")
    # optimization
    temper = 2
#     currentBeta = 0.0 # annealing
    for epoch in range(epochs):
        logger.log("Epoch " + str(epoch + 1))
        # training
        batchLoss = []
        pbar = tqdm(total = len(trainset))
        for i, batchData in enumerate(trainLoader):
            if speedupTrain and np.random.random() < 0.9:
                pbar.update(len(batchData["users"]))
                continue
            optimizer.zero_grad()
#             loss, recLoss, kld = get_gen_loss(batchData, model, CEL, currentBeta)
            loss, recLoss, kld = get_gen_loss(batchData, model, CEL, beta)

            batchLoss.append(loss.item())
            if len(batchLoss) >= 50:
                runningLoss.append(np.mean(batchLoss[-50:]))

            # backward and optimize
            loss.backward()
            optimizer.step()

            # update progress
            pbar.update(len(batchData["users"]))
            
#             # beta annealing
#             currentBeta = 0.9 * currentBeta + 0.1 * beta

        # record epoch loss
        trainHistory.append(np.mean(batchLoss))
        pbar.close()
        logger.log("train loss: " + str(trainHistory[-1]))

        # validation
        batchRecLoss = []
        batchKLDLoss = []
        batchLoss = []
        with torch.no_grad():
            pbar = tqdm(total = len(valset))
            for i, batchData in enumerate(valLoader):
                if speedupVal and np.random.random() < 0.99:
                    pbar.update(len(batchData["users"]))
                    continue
                loss, recLoss, KLD = get_gen_loss(batchData, model, CEL, beta, n_neg = trainset.nCandidate)
                batchRecLoss.append(recLoss.item())
                batchKLDLoss.append(KLD.item())
                batchLoss.append(loss.item())
                pbar.update(len(batchData["users"]))
            pbar.close()
        valHistory.append(np.mean(batchLoss))
        logger.log("validation Loss: " + str(valHistory[-1]) + \
                   " = " + str(np.mean(batchRecLoss)) + " + " + str(beta) + " * " + str(np.mean(batchKLDLoss)))
#                    " = " + str(np.mean(batchRecLoss)) + " + " + str(currentBeta) + " * " + str(np.mean(batchKLDLoss)))
        
        # recommendation test
        n_test_trial = 100
        enc = torch.zeros(5, n_test_trial)
        maxnc = torch.zeros(5, n_test_trial)
        minnc = torch.zeros(5, n_test_trial)
        with torch.no_grad():
            # repeat for several trails
            for k in tqdm(range(n_test_trial)):
                # sample users for each trail
                sampledUsers = sample_users(resp_model, bs)
                # test for different input condition/context
                context = torch.zeros(bs, 5).to(model.device)
                for i in range(5):
                    # each time set one more target response from 0 to 1
                    context[:,i] = 1
                    # recommend should gives slate features of shape (B, L)
                    rSlates, mu = model.recommend(context, sampledUsers, return_item = True)
                    resp = m(resp_model(rSlates.view(bs, -1), sampledUsers))
                    # the expected number of click
                    nc = torch.sum(resp,dim=1)
                    enc[i,k] = torch.mean(nc).detach().cpu()
                    maxnc[i,k] = torch.max(nc).detach().cpu()
                    minnc[i,k] = torch.min(nc).detach().cpu()
        for i in range(5):
            logger.log("Expected response (" + str(i+1) + "): " + \
                       str(torch.mean(minnc[i]).numpy()) + "; " + \
                       str(torch.mean(enc[i]).numpy()) + "; " + \
                       str(torch.mean(maxnc[i]).numpy()))

        # save best model and early termination
        if epoch == 0 or valHistory[-1] < bestValLoss - 1e-3:
            torch.save(model, open(model_path, 'wb'))
            logger.log("Save best model")
            temper = 3
            bestValLoss = valHistory[-1]
        else:
            temper -= 1
            logger.log("Temper down to " + str(temper))
#             if temper == 0:
#                 logger.log("Out of temper, early termination.")
#                 break
    logger.log("Move model to cpu before saving")
    bestModel = torch.load(open(model_path, 'rb'))
    bestModel.to("cpu")
    bestModel.device = "cpu"
    torch.save(bestModel, open(model_path, 'wb'))
    return

#######################################
#                main                 #
#######################################

    
def get_model(args, response_model):
    if args.model == "listcvae":
        encoderStruct = [int(v) for v in args.enc_struct[1:-1].split(",")]
        decoderStruct = [int(v) for v in args.dec_struct[1:-1].split(",")]
        priorStruct = [int(v) for v in args.prior_struct[1:-1].split(",")]
        model = UserListCVAEWithPrior(response_model.docEmbed, None if response_model.noUser else response_model.userEmbed, \
                                      args.s, args.dim, args.z_size, args.s + 1, \
                                      encoderStruct, decoderStruct, priorStruct, args.nouser, args.device)
    elif args.model in PIVOTCVAE_MODELS:
        encoderStruct = [int(v) for v in args.enc_struct[1:-1].split(",")]
        psmStruct = [int(v) for v in args.psm_struct[1:-1].split(",")]
        scmStruct = [int(v) for v in args.scm_struct[1:-1].split(",")]
        priorStruct = [int(v) for v in args.prior_struct[1:-1].split(",")]
        model = PIVOTCVAE_MODELS[args.model](response_model.docEmbed, \
                                             None if response_model.noUser else response_model.userEmbed, \
                                             args.s, args.dim, args.z_size, args.s + 1, \
                                             encoderStruct, psmStruct, scmStruct, priorStruct, args.nouser, args.device)
    elif args.model == "randomlistcvae":
        model = None
    return model

def main(args):
    
    logPath = utils.make_gen_model_path(args, "log/")
    logger = utils.Logger(logPath)
    if args.dataset != "yoochoose" and args.dataset != "movielens": # simulation envirionment
        respModel, trainset, valset = dae.load_simulation(args, logger)
    else: # real-world datasets
        if args.dataset == "yoochoose":
            train, val, test = dae.read_yoochoose(entire_set = False)
            args.nouser == True
            trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
            trainset.balance_n_click()
            valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
        elif args.dataset == "movielens":
            train, val = dae.read_movielens(entire = False)
            trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
            valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
        respModel = torch.load(open(args.resp_path, 'rb'))
        
    # train generative model
    
    # do sampling softmax
    trainset.init_sampling(args.nneg)
    valset.init_sampling(args.nneg)
    respModel.to(args.device)
    respModel.device = args.device
    # generative model
    gen_model = get_model(args, respModel)
    if not args.mask_train:
        logger.log("Candidate training")
        gen_model.candidateFlag = True
    else:
        logger.log("Mask training")
    # beta grid search
    import setproctitle 
    if args.beta > 0:
        setproctitle.setproctitle("Socrate")
        modelPath = utils.make_gen_model_path(args, "trained_gen/")
        train_on_dataset(trainset, valset, gen_model, modelPath, logger, respModel, \
                    args.batch_size, args.epochs, args.lr, args.wdecay, args.beta)
    # single beta test
    else:
        betaList = settings.BETA_LIST
        setproctitle.setproctitle("Socrate(0/" + str(len(betaList)) + ")")
        logger.log("Beta test")
        
        for i in range(len(betaList)):
            beta = betaList[i]
            args.beta = beta
            logPath = utils.make_gen_model_path(args, "log_beta/")
            modelPath = utils.make_gen_model_path(args, "trained_beta/")
            betaLogger = Logger(logPath)
            betaLogger.log("beta = " + str(beta))
            train_on_dataset(trainset, valset, gen_model, betaModelPath, betaLogger, respModel, \
                    args.batch_size, args.epochs, args.lr, args.wdecay, beta)
            setproctitle.setproctitle("Socrate(" + str(i+1) + "/" + str(len(betaList)) + ")")
            logger.log("Done, model saved to: " + modelPath)
            
            
def add_gen_model_parse(parser):
    parser.add_argument('--dim', type=int, default=8, help='should be the same as --dim of response model')
    parser.add_argument('--model', type=str, default='pivotcvae', help='model keyword from [listcvae, pivotcvae]')
    parser.add_argument('--z_size', type=int, default=16, help='encoding size')
    parser.add_argument('--mask_train', action='store_true', help='set this to do sampled softmax, otherwise candidates will be selected by data loader, the over-concentration case will not appear')
    
    # used by all cvae models
    parser.add_argument('--enc_struct', type=str, default="[54,256,256]", help='mlp structure for prediction')
    parser.add_argument('--prior_struct', type=str, default="[14,128,128]", help='mlp structure for prediction')
    parser.add_argument('--beta', type=float, default=-1, help='trade-off term between reconstruction loss and KLD for CVAE models; do beta-test if -1 by default')
    
    # unique for listcvae models
    parser.add_argument('--dec_struct', type=str, default="[30,256,256,40]", help='mlp structure for prediction')
    
    # unique for pivotcvae models
    parser.add_argument('--psm_struct', type=str, default="[30,256,256,8]", help='mlp structure for prediction')
    parser.add_argument('--scm_struct', type=str, default="[38,256,256,32]", help='mlp structure for prediction')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # real-world dataset configuration [--dataset, --s, --nouser]
    parser = dae.add_data_parse(parser)
    
    # simulation configuration 
    # [--sim_root, --sim_dim, --n_user, --n_item, --n_train, --n_val, --n_test, 
    # --pbias_min, --pbias_max, --mr_factor, --balance]
    parser = dae.add_sim_parse(parser)
    
    # training configuration [--batch_size, --epochs, --lr, --wdecay, --device, --nneg]
    parser = utils.add_training_parse(parser)
    
    # generative model configuration
    parser = add_gen_model_parse(parser)
    
    # load pretrained user response model
    parser.add_argument('--resp_path', type=str, default="resp/resp_[48,256,256,5]_spotify_BS64_dim8_lr0.00030_decay0.00010", help='trained user response model, only valid when training generative rec model')
    
    
    args = parser.parse_args()
    main(args)
        