import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join    
import pickle
import torch

from my_utils import make_sim_path, read_lines
from env.response_model import URM, URM_P, URM_P_MR
from data_loader import UserSlateResponseDataset
from settings import DATA_ROOT

DATA_KEYS = ["yoochoose", "movielens", "urm", "urmp", "urmpmr"]


def balance_n_click(slates, users, responses):
    '''
    Response type: #click.
    Distribution may be skewed over types.
    Augment the data by duplicating records of minority response type.
    '''
    clickRecord = np.sum(responses, axis=1)
    nClick = np.unique(clickRecord, return_counts = True)
    print("Before augmentation: ")
    print(nClick)
    nRepeat = ((np.max(nClick[1]) - nClick[1])/2).astype(int)
    N = np.sum(nRepeat)
    slateSize = slates.shape[1]
    newSlates = np.zeros((N, slateSize))
    newResponses = np.zeros((N, slateSize))
    newUsers = np.zeros((N,))
    loc = 0
    for i in range(slateSize + 1):
        print("Augmenting data for #click == " + str(i))
        indices = (clickRecord == i)
        candSlates = slates[indices]
        candUsers = users[indices]
        candResponses = responses[indices]
        N = len(candSlates)
        print("Number of new record: " + str(nRepeat[i]))
        for j in tqdm(range(nRepeat[i])):
            selectedRow = np.random.choice(N)
            newSlates[loc + j] = candSlates[selectedRow]
            newUsers[loc + j] = candUsers[selectedRow]
            newResponses[loc + j] = candResponses[selectedRow]
        loc = loc + nRepeat[i]
    shuffledIndex = np.arange(len(newSlates))
    np.random.shuffle(shuffledIndex)
    finalSlates = np.concatenate([slates, newSlates[shuffledIndex]], axis = 0).astype(int)
    finalUsers = np.concatenate([users, newUsers[shuffledIndex]], axis = 0).astype(int)
    finalResponses = np.concatenate([responses, newResponses[shuffledIndex]], axis = 0)
    nClick = np.unique(np.sum(finalResponses, axis = 1), return_counts = True)
    print("After augmentation: ")
    print(nClick)
    return finalSlates, finalUsers, finalResponses

def read_movielens(entire = False):
    testset = {
        "features": read_lines(DATA_ROOT + "movielens/test_slate.csv", [int, int, int, int, int]),
        "sessions": read_lines(DATA_ROOT + "movielens/test_user.csv", [int]),
        "responses": read_lines(DATA_ROOT + "movielens/test_resp.csv", [int, int, int, int, int])
    }
    if entire:
        trainset = {
            "features": read_lines(DATA_ROOT + "movielens/entire_slate.csv", [int, int, int, int, int]),
            "sessions": read_lines(DATA_ROOT + "movielens/entire_user.csv", [int]),
            "responses": read_lines(DATA_ROOT + "movielens/entire_resp.csv", [int, int, int, int, int])
        }
    else:
        trainset = {
            "features": read_lines(DATA_ROOT + "movielens/train_slate.csv", [int, int, int, int, int]),
            "sessions": read_lines(DATA_ROOT + "movielens/train_user.csv", [int]),
            "responses": read_lines(DATA_ROOT + "movielens/train_resp.csv", [int, int, int, int, int])
        }
    return trainset, testset

def read_yoochoose(from_encoded = True, entire_set = False):
    import pickle
    if not from_encoded:
        train = pickle.load(open(DATA_ROOT + "yoochoose-data/slate_data_size5_min20_train.pkl", 'rb'))
        val = pickle.load(open(DATA_ROOT + "yoochoose-data/slate_data_size5_min20_val.pkl", 'rb'))
        test = pickle.load(open(DATA_ROOT + "yoochoose-data/slate_data_size5_min20_test.pkl", 'rb'))
    else:
        if entire_set:
            train = pickle.load(open(DATA_ROOT + "yoochoose-data/encoded_entire.pkl", 'rb'))
        else:
            train = pickle.load(open(DATA_ROOT + "yoochoose-data/encoded_train.pkl", 'rb'))
        val = pickle.load(open(DATA_ROOT + "yoochoose-data/encoded_val.pkl", 'rb'))
        test = pickle.load(open(DATA_ROOT + "yoochoose-data/encoded_test.pkl", 'rb'))
    return train, val, test

def do_encode(original_list, mapping, flag):
    newData = np.zeros_like(original_list)
    if flag == "i":
        for i in tqdm(range(len(original_list))):
            s = original_list[i]
            for j in range(len(s)):
                newData[i,j] = mapping[s[j]]
    elif flag == "u":
        for i in tqdm(range(len(original_list))):
            newData[i] = mapping[original_list[i]]
    return newData
            
def encode_yoochoose():
    train,val,test = read_yoochoose(from_encoded = False)
    
    print("Original feature shape")
    print("Train: " + str(train["features"].shape))
    print("Val: " + str(val["features"].shape))
    print("Test: " + str(test["features"].shape))
    entireSlates = np.concatenate([train["features"], val["features"], test["features"]])
    entireUsers = np.concatenate([train["sessions"], val["sessions"], test["sessions"]])
    entireResps = np.concatenate([train["responses"], val["responses"], test["responses"]])
    items = np.unique(entireSlates)
    print("Unique items: " + str(len(np.unique(entireSlates, return_counts = True))))
    users = np.unique(entireUsers)
    print("Unique users: " + str(len(np.unique(entireUsers))))
    # find encoding for users and items
    itemMap = {items[i]: i for i in range(len(items))}
    userMap = {users[i]: i for i in range(len(users))}
    
    itemInTrain = {}
    userInTrain = {}
    trainIndices = []
    valIndices = []
    testIndices = []
    N = len(entireSlates)
    nTrain = 0.8 * N
    nVal = 0.1 * N
    for i in range(N):
        s = entireSlates[i]
        u = entireUsers[i]
        # make sure each item and user appeared in trainset at least once
        toTrain = False
        if u not in userInTrain:
            toTrain = True
        else:
            for item in s:
                if item not in itemInTrain:
                    toTrain = True
        if toTrain:
            trainIndices.append(i)
            for item in s:
                itemInTrain[item] = 1
            userInTrain[u] = 1
        # split dataset by 8-1-1
        else:
            K = len(trainIndices)
            if np.random.random() < (nTrain - K) / (N - K):
                trainIndices.append(i)
            elif np.random.random() < 0.5:
                valIndices.append(i)
            else:
                testIndices.append(i)
    
    finalFeatureTrain = entireSlates[trainIndices]
    finalFeatureVal = entireSlates[valIndices]
    finalFeatureTest = entireSlates[testIndices]
    finalSessionTrain = entireUsers[trainIndices]
    finalSessionVal = entireUsers[valIndices]
    finalSessionTest = entireUsers[testIndices]
    finalResponseTrain = entireResps[trainIndices]
    finalResponseVal = entireResps[valIndices]
    finalResponseTest = entireResps[testIndices]
    
    print("New feature shape")
    print("Train: " + str(finalFeatureTrain.shape))
    print("Val: " + str(finalFeatureVal.shape))
    print("Test: " + str(finalFeatureTest.shape))
    
    # encode
    traindata = {"features": do_encode(finalFeatureTrain, itemMap, "i"), \
                 "sessions": do_encode(finalSessionTrain, userMap, "u"), \
                 "responses": finalResponseTrain}
    valdata = {"features": do_encode(finalFeatureVal, itemMap, "i"), \
                 "sessions": do_encode(finalSessionVal, userMap, "u"), \
                 "responses": finalResponseVal}
    testdata = {"features": do_encode(finalFeatureTest, itemMap, "i"), \
                 "sessions": do_encode(finalSessionTest, userMap, "u"), \
                 "responses": finalResponseTest}
    print("Unique items: " + str(len(np.unique(traindata["features"]))))
    print("Unique users: " + str(len(np.unique(traindata["sessions"]))))
    
    pickle.dump(traindata, open(DATA_ROOT + "yoochoose-data/encoded_train.pkl", 'wb'))
    pickle.dump(valdata, open(DATA_ROOT + "yoochoose-data/encoded_val.pkl", 'wb'))
    pickle.dump(testdata, open(DATA_ROOT + "yoochoose-data/encoded_test.pkl", 'wb'))
        
            
def load_simulation(args, logger):
    simulatorPath = make_sim_path(args)
    if args.sim_root:
        logger.log("load from existing simulation data")
        simulator = torch.load(open(simulatorPath, 'rb'))
        trainset = pickle.load(open(simulatorPath + "_train", 'rb'))
        valset = pickle.load(open(simulatorPath + "_val", 'rb'))
    else:
        logger.log("Construct simulation data")
        # simulator
        if args.dataset == "urm":
            simulator = URM(args.n_item, args.n_user, args.s, args.sim_dim, "cpu", args.nouser)
        if args.dataset == "urmp":
            simulator = URM_P(args.n_item, args.n_user, args.s, args.sim_dim, "cpu", args.nouser, \
                            args.pbias_max, args.pbias_min)
        if args.dataset == "urmpmr":
            simulator = URM_P_MR(args.n_item, args.n_user, args.s, args.sim_dim, "cpu", args.nouser, \
                            args.pbias_max, args.pbias_min, args.mr_factor)

        # generate dataset from simulator
        logger.log("generating training set")
        sampledU, sampledS, sampledR = simulator.generate_dataset(\
                min_user_hist = 10, min_item_hist = 10, n_record = args.n_train)
        if args.balance:
            augSlates, augUsers, augResps = balance_n_click(sampledS, sampledU, sampledR)
            trainset = UserSlateResponseDataset(augSlates, augUsers, augResps, args.nouser)
        else:
            trainset = UserSlateResponseDataset(sampledS, sampledU, sampledR, args.nouser)
        logger.log("generating validation set")
        sampledU, sampledS, sampledR = simulator.generate_dataset(\
                min_user_hist = 1, min_item_hist = 1, n_record = args.n_val)
        valset = UserSlateResponseDataset(sampledS, sampledU, sampledR, args.nouser)
        
        # save simulator for reuse
        logger.log("save simulator and generated datasets")
        torch.save(simulator, open(simulatorPath + "_simulator", 'wb'))
        pickle.dump(trainset, open(simulatorPath + "_train", 'wb'))
        pickle.dump(valset, open(simulatorPath + "_val", 'wb'))
    return simulator, trainset, valset

def add_data_parse(parser):
    parser.add_argument('--dataset', type=str, default='spotify', help='dataset keyword from ' + str(DATA_KEYS))
    parser.add_argument('--s', type=int, default=5, help='number of items in a slate')
    parser.add_argument('--nouser', action='store_true', help='user may or may not be considered as input, make sure to change the corresponding model structure and environment')
    return parser

def add_sim_parse(parser):
    parser.add_argument('--sim_root', action='store_true', help='set this to load simulation dataset from existing files')
    parser.add_argument('--sim_dim', type=int, default=8, help='number of latent features')
    parser.add_argument('--n_user', type=int, default=1000, help='number of user in the simulation')
    parser.add_argument('--n_item', type=int, default=3000, help='number of item in the simulation')
    parser.add_argument('--n_train', type=int, default=100000, help='number of records when generating dataset from simulation')
    parser.add_argument('--n_val', type=int, default=10000, help='number of records when generating dataset from simulation')
    parser.add_argument('--n_test', type=int, default=10000, help='number of records when generating dataset from simulation')
    parser.add_argument('--pbias_min', type=float, default=-0.2, help='min value of positional bias')
    parser.add_argument('--pbias_max', type=float, default=0.2, help='max value of positional bias')
    parser.add_argument('--mr_factor', type=float, default=0.2, help='max value of positional bias')
    parser.add_argument('--balance', action='store_true', help='apply response balancing')
    return parser