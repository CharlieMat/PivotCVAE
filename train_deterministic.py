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
from response_model import UserResponseModel_MLP, sample_users
from models.deterministic import MF, NeuMF, DiverseMF
from my_utils import make_model_path, Logger, add_sim_parse, ms2str
import settings


def get_ranking_loss(batch_data, model, lossFun):
    # get input and target and forward
    slates = torch.LongTensor(batch_data["slates"]).to(model.device)
    users = torch.LongTensor(batch_data["users"]).to(model.device)
    targets = torch.tensor(batch_data["responses"]).to(torch.float).to(model.device)

    # loss
    pred = model.forward(slates, targets, u = users)
    loss = lossFun(model.m(pred), targets)
    
    return loss

def train_ranking(trainset, valset, model, model_path, logger, resp_model, \
                    bs, epochs, lr, decay):
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
    '''
    
    logger.log("----------------------------------------")
    logger.log("Train user response model as simulator")
    logger.log("\tbatch size: " + str(bs))
    logger.log("\tnumber of epoch: " + str(epochs))
    logger.log("\tlearning rate: " + str(lr))
    logger.log("\tweight decay: " + str(decay))
    logger.log("----------------------------------------")
    model.log(logger)
    logger.log("----------------------------------------")

    # data loaders
    trainLoader = DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 0)
    valLoader = DataLoader(valset, batch_size = bs, shuffle = False, num_workers = 0)    

    # loss function and optimizer
    BCE = nn.BCELoss()
    m = nn.Sigmoid()
    optimizer = opt.Adam(model.parameters(), lr = lr)
    
    runningLoss = []  # step loss history
    trainHistory = [] # epoch training loss
    valHistory = []   # epoch validation loss
    bestLoss = np.float("inf")
    bestValLoss = np.float("inf")
    # optimization
    temper = 2
    for epoch in range(epochs):
        logger.log("Epoch " + str(epoch + 1))
        # training
        batchLoss = []
        pbar = tqdm(total = len(trainset))
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()
            loss = get_ranking_loss(batchData, model, BCE)

            batchLoss.append(loss.item())
            if len(batchLoss) >= 50:
                runningLoss.append(np.mean(batchLoss[-50:]))

            # backward and optimize
            loss.backward()
            optimizer.step()

            # update progress
            pbar.update(len(batchData["users"]))
            
        # record epoch loss
        trainHistory.append(np.mean(batchLoss))
        pbar.close()
        logger.log("train loss: " + str(trainHistory[-1]))

        # validation
        batchLoss = []
        with torch.no_grad():
            pbar = tqdm(total = len(valset))
            for i, batchData in enumerate(valLoader):
                loss = get_ranking_loss(batchData, model, BCE)
                batchLoss.append(loss.item())
                pbar.update(len(batchData["users"]))
            pbar.close()
        valHistory.append(np.mean(batchLoss))
        logger.log("validation Loss: " + str(valHistory[-1]))
        
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
                    rSlates, _ = model.recommend(context, sampledUsers, return_item = True)
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
            if temper == 0:
                logger.log("Out of temper, early termination.")
                break
    logger.log("Move model to cpu before saving")
    bestModel = torch.load(open(model_path, 'rb'))
    bestModel.to("cpu")
    bestModel.device = "cpu"
    torch.save(bestModel, open(model_path, 'wb'))
    return

def get_ranking_model(args, response_model):
    if args.model == "mf":
        model = MF(response_model.docEmbed, response_model.userEmbed, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == "diverse_mf":
        model = DiverseMF(response_model.docEmbed, response_model.userEmbed, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == "neumf":
        mlpStruct = [int(v) for v in args.struct[1:-1].split(",")]
        model = NeuMF(response_model.docEmbed, response_model.userEmbed, mlpStruct, \
                   args.s, args.dim, args.device, fine_tune = True)
    else:
        raise NotImplemented
    return model

def main(args):
    assert not args.nouser
    logPath = make_model_path(args, "log/")
    logger = Logger(logPath)
    if args.dataset != "spotify" and args.dataset != "yoochoose" and args.dataset != "movielens":
        args.sim_root = True
        respModel, trainset, valset = dae.load_simulation(args, logger)
    elif args.dataset == "movielens":
        train, val = dae.read_movielens(entire = False)
        trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
        valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
        respModel = torch.load(open(args.resp_path, 'rb'))
            
#     # do sampling softmax
#     trainset.init_sampling(args.nneg)
#     valset.init_sampling(args.nneg)
    
    respModel.to(args.device)
    respModel.device = args.device
            
    # generative model
    gen_model = get_ranking_model(args, respModel)
    gen_model.to(args.device)
#     if not args.mask_train:
#         logger.log("Candidate training")
#         gen_model.candidateFlag = True
#     else:
#         logger.log("Mask training")

    modelPath = make_model_path(args, "model/")
    train_ranking(trainset, valset, gen_model, modelPath, logger, respModel, \
                args.batch_size, args.epochs, args.lr, args.wdecay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='spotify', help='dataset keyword from ' + str(dae.DATA_KEYS))
    parser.add_argument('--dim', type=int, default=8, help='number of latent features')
    parser.add_argument('--s', type=int, default=5, help='number of items in a slate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--model', type=str, default='mf', help='model keyword')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/cuda:0/...')
    parser.add_argument('--nneg', type=int, default=1000, help='number of negative samples for softmax during training')
    parser.add_argument('--nouser', action='store_true', help='user may or may not be considered as input, make sure to change the corresponding model structure and environment')
    
    # used by NeuMF models
    parser.add_argument('--struct', type=str, default="[16,256,256,1]", help='mlp structure for prediction')
    
    # if training generative model
    parser.add_argument('--response', action='store_true', help='training response model for the generation model')
    parser.add_argument('--resp_path', type=str, default="resp/resp_[48,256,256,5]_spotify_BS64_dim8_lr0.00030_decay0.00010", help='trained user response model, only valid when training generative rec model')
    
    # used when simulation
    parser = add_sim_parse(parser)
    
    args = parser.parse_args()
    main(args)
        