import os
import sys
import torch
import csv
import numpy as np
from tqdm import tqdm

from settings import SIMDATA_ROOT

def ms2str(ms):
    seconds=int(ms/1000)%60
    minutes=int(ms/(1000*60))%60
    hours=int(ms/(1000*60*60))%24
    days=int(ms/(1000*60*60*24))
    return str(days) + "d" + str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s"

def read_lines(fname, row_types = [int, int, float], debug = False, limit = -1):
    print("Load data from \"" + fname + "\"")
    with open(fname, "r") as csvFile:
        fr = csv.reader(csvFile, delimiter = ",")
        data = []
        for rowStr in tqdm(fr):
            if len(row_types) == 1:
                row = row_types[0](rowStr[0])
            else:
                row = []
                for t in range(len(row_types)):
                    row.append(row_types[t](rowStr[t]))
            data.append(row)
    return np.array(data)

def write_lines(fname, data):
    print("Save data to \"" + fname + "\"")
    with open(fname, "w") as csvFile:
        if len(data) > 0:
            fw = csv.writer(csvFile, delimiter = ",")
            if type(data[0]) == int:
                for i in tqdm(range(len(data))):
                    fw.writerow([data[i]])
            else:
                for i in tqdm(range(len(data))):
                    fw.writerow(data[i])

def make_sim_path(args, root = SIMDATA_ROOT):
    model_path = root
    model_path += (args.dataset + "_")
    model_path += ("nouser_" if args.nouser else "")
    model_path += 'dim%d_' % args.sim_dim
    model_path += 'u%d_' % args.n_user
    model_path += 'i%d_' % args.n_item
    model_path += 'n%d' % args.n_train
    if args.dataset == "urmpmr":
        model_path += '_mr%.2f' % args.mr_factor
    return model_path

def make_data_path(args, root):
    if args.dataset == "urmpmr" or args.dataset == "urmp" or args.dataset == "urm":
        model_path = make_sim_path(args, root)
    else:
        model_path = root + args.dataset 
        model_path += ("_nouser" if args.nouser else "")
    return model_path
    
def make_config_path(args, root):
    model_path = root
    model_path += 'BS%d_' % args.batch_size
    model_path += 'lr%.5f_' % args.lr
    model_path += 'decay%.5f' % args.wdecay
    return model_path

def make_resp_model_path(args, root):
    model_path = make_data_path(args, root)
    model_path += "/"
    model_path += ("resp_" + str(args.resp_struct) + "_dim" + str(args.dim)) + "_"
    model_path = make_config_path(args, model_path)
    build_path(model_path)
    return model_path

def make_gen_model_path(args, root):
    model_path = make_data_path(args, root) + "/"
    model_path += args.model
    model_path += '_beta%.5f' % args.beta
    model_path += "_enc" + str(args.enc_struct)
    if "listcvae" in args.model:
        model_path += "_dec" + str(args.dec_struct)
    elif "pivotcvae" in args.model:
        model_path += "_psm" + args.psm_struct + "_scm" + args.scm_struct + "_"
    model_path += make_config_path(args, model_path)
    build_path(model_path)
    return model_path

def make_result_path(args, root = "results/"):
    if not os.path.exists(root):
        os.makedirs(root)
    model_path = root
    model_path += (args.dataset + "_")
    if args.dataset == "urmpmr":
        model_path += 'mr%.2f_' % args.mr_factor
    model_path += args.test_key + "_"
    if args.test_key == "ranking":
        model_path += args.plot_feature + "_"
#     model_path += ("nouser_" if args.nouser else "")
#     model_path += args.model_path.split('/')[-1] + "_"
    if args.all_beta:
        model_path += "allbeta"
    elif args.single_beta:
        model_path += "singlebeta"
    elif args.sample_k:
        model_path += "samplek"
    else:
        raise NotImplemented
    return model_path

def make_ranking_result_path(args, root = "results/"):
    if not os.path.exists(root):
        os.makedirs(root)
    model_path = root
    model_path += (args.dataset + "_")
    if args.dataset == "urmpmr":
        model_path += 'mr%.2f_' % args.mr_factor
    model_path += args.test_key + "_"
    model_path += "discriminative"
    return model_path

def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print("dir \"" + fpath + "\" existed")
    else:
        try:
            os.mkdir(fpath)
        except:
            print("error when creating \"" + fpath + "\"") 
            
def build_path(fpath):
    dirs = [p for p in fpath.split("/")]
    curP = ""
    for p in dirs[:-1]:
        curP += p
        check_folder_exist(curP)
        curP += "/"

class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on
        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'
        build_path(log_path)
        print("Log file path:\n" + self.log_path)

    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
            
def add_training_parse(parser):
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/cuda:0/...')
    parser.add_argument('--nneg', type=int, default=1000, help='number of negative samples for softmax during training')
    return parser