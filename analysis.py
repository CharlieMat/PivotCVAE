import torch
import torch.nn.functional as F


def get_coverage(slates, N):
    """
    Item coverage for give slates
    @input:
     - slates: list of slates
     - N: the total number of items in D
    """
    return len(torch.unique(slates)) * 1.0 / N

def get_ILS(slates, embeds, normalize = False):
    """
    Intra-List Similarity, diversity can be calculated as (1 - ILS)
    @input:
     - slates: list of slates
     - embeds: nn.Embedding for all possible items
    """
    assert slates.shape[1] == 5
    # obtain embeddings for all items
    emb = F.normalize(embeds(slates), p = 2, dim = 2)
    # calculate similarities for each pair of items in each slate
    sims = torch.bmm(emb, emb.transpose(1,2)).reshape(slates.shape[0], -1)
#     if normalize:
#         sims /= torch.max(sims)
    # take the average for each slate
    sims = (torch.sum(sims, dim = 1) - slates.shape[1]) / (slates.shape[1] * (slates.shape[1] - 1))
    return sims

def get_hit_rate(recommendation, candidates, labels):
    '''
    @input:
    - recommendation: (N * #recommendation)
    - candidates: (N * #candidates)
    - labels: 0-1 label matrix of shape (N * #candidates)
    @output:
    - hitRateAtK: [hitrate for k in range(#recommendation)]
    - hitRateAtK: (N * #recommendation)
    '''
    hitAtK = torch.zeros(recommendation.shape[0], recommendation.shape[1] + 1, device = recommendation.device).to(torch.uint8)
    for k in range(recommendation.shape[1]):
        found = (candidates.t() == recommendation[:,k]).to(torch.float)
        hit = torch.sum(found.t() * labels, dim = 1)
        hitAtK[:,k+1] = (hit > 0) | hitAtK[:,k]
    hitAtK = hitAtK[:,1:].to(torch.float)
    hitRateAtK = torch.mean(hitAtK,dim=0)
    return hitRateAtK, hitAtK

def get_recall(recommendation, candidates, labels):
    '''
    @input:
    - recommendation: (N * #recommendation)
    - candidates: (N * #candidates)
    - labels: 0-1 label matrix of shape (N * #candidates)
    @output:
    - recallAtK: [hitrate for k in range(#recommendation)]
    - recallAtK: (N * #recommendation)
    '''
    recallAtK = torch.zeros(recommendation.shape[0], recommendation.shape[1] + 1, device = recommendation.device)
    for k in range(recommendation.shape[1]):
        found = (candidates.t() == recommendation[:,k]).to(torch.float)
        recall = torch.sum(found.t() * labels, dim = 1)
        recallAtK[:,k+1] = recall + recallAtK[:,k]
    recallAtK = recallAtK[:,1:].to(torch.float)
    recallAtK = (recallAtK.t() / (torch.sum(labels, dim=1) + 1e-3)).t()
    return recallAtK

def get_rank_matrix(gt_score_matrix, candidate_score_matrix):
    '''
    @input:
    - gt_score_matrix: (N * slateSize * #ground truth items)
    - candidate_score_matrix: (N * slateSize * #candidate)
    '''
    rank = torch.zeros_like(gt_score_matrix, device = gt_score_matrix.device).to(torch.long)
    for i in range(gt_score_matrix.shape[-1]):
        for j in range(candidate_score_matrix.shape[-1]):
#             print(rank[:,:,i])
            rank[:,:,i] += (candidate_score_matrix[:,:,j] > gt_score_matrix[:,:,i]).to(torch.long)
#             print(rank[:,:,i])
#             input()
    return rank

def get_positional_recall(gt_score_matrix, candidate_score_matrix, labels):
    '''
    @input:
    - gt_score_matrix: (N * slateSize * #ground truth items)
    - candidate_score_matrix: (N * slateSize * #candidate)
    - labels: (N * #ground truth items)
    '''
    precall = torch.zeros_like(candidate_score_matrix, device = candidate_score_matrix.device).to(torch.float)
    # rank matrix: (N * slateSize * #ground truth items)
    rankMatrix = get_rank_matrix(gt_score_matrix, candidate_score_matrix)
#     print(rankMatrix == 0)
#     print(precal[:,:,0])
    precall[:,:,0] = torch.sum((rankMatrix == 0).transpose(0,1).to(torch.float) * labels.to(torch.float), dim = 2).t()
#     print(precal[:,:,0])
    for i in range(precall.shape[-1]-1):
        found = (rankMatrix == (i+1)).transpose(0,1).to(torch.float) * labels.to(torch.float)
        precall[:,:,i+1] = precall[:,:,i] + torch.sum(found, dim = 2).t()
    denom = 1.0 / (torch.sum(labels, dim = 1) + 1e-3)
    precall = (precall.transpose(0,2) * denom).transpose(0,2)
    return precall, torch.mean(precall, dim=0)
    
def get_positional_mrr(gt_score_matrix, candidate_score_matrix, labels):
    '''
    @input:
    - gt_score_matrix: (N * slateSize * #ground truth items)
    - candidate_score_matrix: (N * slateSize * #candidate)
    - labels: (N * #ground truth items)
    '''
    pmrr = torch.zeros_like(candidate_score_matrix, device = candidate_score_matrix.device).to(torch.float)
    # rank matrix: (N * slateSize * #ground truth items)
    rankMatrix = get_rank_matrix(gt_score_matrix, candidate_score_matrix)
    precall[:,:,0] = torch.sum((rankMatrix == 0).transpose(0,1).to(torch.float) * labels.to(torch.float), dim = 2).t()
    for i in range(precall.shape[-1]-1):
        found = (rankMatrix == (i+1)).transpose(0,1).to(torch.float) * labels.to(torch.float)
        precall[:,:,i+1] = precall[:,:,i] + torch.sum(found, dim = 2).t()
    denom = 1.0 / (torch.sum(labels, dim = 1) + 1e-3)
    precall = (precall.transpose(0,2) * denom).transpose(0,2)
    return precall, torch.mean(precall, dim=0)
    