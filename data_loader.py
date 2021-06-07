import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class UserSlateResponseDataset(Dataset):
    def __init__(self, slates, users, responses, no_user = False):
        print("Initialize dataset")
        self.slates = slates.reshape(slates.shape[0], -1)
        self.slateSize = self.slates.shape[1]
        self.users = users.reshape(slates.shape[0], -1)
        self.responses = responses.reshape(slates.shape[0], -1).astype(float)
        self.noUser = no_user
        self.sampleSpeedup = False
        print("Slates shape: " + str(slates.shape))
        print("Users shape: " + str(users.shape))
        print("Response shape: " + str(responses.shape))
        print("Unique items: " + str(len(np.unique(slates))))
        print("Unique users: " + str(len(np.unique(users))))
        print("User embedding is ignored" if no_user else "User embedding is used")
        # max item id
#         self.items = np.unique(slates).astype(int)
        self.max_iid = np.max(slates).astype(int)
        # max user id
#         self.users = np.unique(sessions.astype(int))
        self.max_uid = np.max(users).astype(int)
        # total number of slates
        self.L = len(slates)
        # whether generate softmax data (too many items, GPU may not hold, do down-sampling for softmax)
        self.sampling = False
    def __len__(self):
        return self.L
    def __getitem__(self, idx):
        '''
        @return:
        - slates: [iid] of size s
        - users: uid
        - responses: [r]
        if using sampling:
        - sample_candidates: [[iid]] of size s-by-ncandidate
        - sample_target: [col id in sample_candidates]
        '''
        feature = self.slates[idx]
        user = self.users[idx]
        resp = self.responses[idx]
        if not self.sampling:
            return {"slates": feature, "users": user, "responses": resp}
        else:
            candidates = np.random.randint(self.max_iid + 1, size = (len(feature),self.nCandidate))
            targetIds = []
            for i in range(self.slates.shape[1]):
                if feature[i] in candidates[i]:
                    targetIds.append(np.where(candidates[i]==feature[i])[0][0])
                else:
                    candidates[i,0] = feature[i]
                    targetIds.append(0)
            return {"slates": feature, "users": user, "responses": resp, \
                    "sample_candidates": candidates, "sample_targets": np.array(targetIds)}
    def init_sampling(self, n_candidate = 1000):
        print("sampling with nNeg = " + str(n_candidate))
        self.sampling = True
        self.nCandidate = n_candidate
    def balance_n_click(self):
        clickRecord = np.sum(self.responses, axis=1)
        nClick = np.unique(clickRecord, return_counts = True)
        print("Before augmentation: ")
        print(nClick)
        nRepeat = ((np.max(nClick[1]) - nClick[1])/2).astype(int)
        newSlates = np.zeros((np.sum(nRepeat), self.slateSize))
        newResponses = np.zeros((np.sum(nRepeat), self.slateSize))
        newUsers = np.zeros((np.sum(nRepeat),1))
        loc = 0
        for i in range(self.slateSize + 1):
            print("Augmenting data for #click == " + str(i))
            indices = (clickRecord == i)
            candSlates = self.slates[indices]
            candResponses = self.responses[indices]
            candUsers = self.users[indices]
            N = len(candSlates)
            print("Number of new record: " + str(nRepeat[i]))
            for j in tqdm(range(nRepeat[i])):
                selectedRow = np.random.choice(N)
                newSlates[loc + j] = candSlates[selectedRow]
                newResponses[loc + j] = candResponses[selectedRow]
                newUsers[loc + j] = candUsers[selectedRow]
            loc = loc + nRepeat[i]
        shuffledIndex = np.arange(len(newSlates))
        np.random.shuffle(shuffledIndex)
#         self.originalSlates = self.slates
        self.slates = np.concatenate([self.slates, newSlates[shuffledIndex]], axis = 0).astype(int)
#         self.originalResponses = self.originalResponses
        self.responses = np.concatenate([self.responses, newResponses[shuffledIndex]], axis = 0).astype(float)
        self.users = np.concatenate([self.users, newUsers[shuffledIndex]], axis = 0).astype(int)
        self.L = len(self.slates)
        nClick = np.unique(np.sum(self.responses, axis = 1), return_counts = True)
        print("After augmentation: ")
        print(nClick)
        print("Number of records: " + str(len(self.slates)))
        
    
class SimulationDataset(Dataset):
    """
    Simulated data only, not augmentation of existing dataset
    """
    def __init__(self, nUsers = 10000, nItems = 100000, model = "urm", sparsity = 0.97):
        """
        @input:
         - nUsers: number of users
         - nItems: number of items
         - model: one of:
             - "urm": independent item response"
             - "urm_p": item response contains positional biases
             - "urm_p_br": item response contains positional biases and binaru item relations
             - "urm_p_br": item response contains positional biases and multivariate item relations
         - sparsity: is used to determine the size of the dataset each round, which is sparsity * nUsers * nItems
        """
        # simulators are implemented as user response model
        from env.response_model import URM, URM_P, URM_P_BR, URM_P_MR
        modelList = {"urm":URM, "urm_p":URM_P, "urm_p_br":URM_P_BR, "urm_p_mr": URM_P_MR, "urm_p_mr_bigrho": URM_P_MR}
        import hyperparams
        params = hyperparams.DEFAULT_PARAMS[model]
        
        self.simulator = modelList[model](nItems, nUsers, params)
        self.simulator.to(self.simulator.device)
        
        # generate temporary dataset, this dataset may change if another round of generate_dataset is called
        print("generating training set")
        sampledU, sampledS, sampledR = self.simulator.generate_dataset(\
                min_user_hist = 10, min_item_hist = 10, n_record = int((1-sparsity) * nUsers * nItems))
        self.trainDataset = {"users": sampledU, "slates": sampledS, "resp": sampledR}
        
        print("generating validation set")
        sampledU, sampledS, sampledR = self.simulator.generate_dataset(\
                min_user_hist = 0, min_item_hist = 0, n_record = int((1-sparsity) * (1-sparsity) * nUsers * nItems))
        self.valDataset = {"users": sampledU, "slates": sampledS, "resp": sampledR}
        
        self.L = len(self.trainDataset["users"])
        print("Dataset size: " + str(self.L))
        self.isTrain = True
        
    def __len__(self):
        return self.L
    def __getitem__(self, idx):
        if self.isTrain:
            dtst = self.trainDataset 
        else:
            dtst = self.valDataset
        user = dtst["users"][idx]
        slate = dtst["slates"][idx]
        resp = dtst["resp"][idx]
        return {"features": slate, "users": user, "responses": resp}
    def set_train(self, is_train = True):
        self.isTrain = is_train
        if self.isTrain:
            self.L = len(self.trainDataset["users"])
        else:
            self.L = len(self.valDataset["users"])
    def get_response(self, users, slates):
#         users = users.to(self.simulator.device)
#         slates = slates.to(self.simulator.device)
        p, dEmb, dBias, uEmb, uBias = self.simulator(users, slates)
        return p