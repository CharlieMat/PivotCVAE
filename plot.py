import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

from settings import PLOT_COLOR, PLOT_NAME, BETA_LIST, DEFAULT_BETA_TICKS
from my_utils import build_path

'''
Plot for data observation:
* plot_slate_response_distribution

Core plot function for evaluations:
* plot_standard_test_result
* plot_likelihood_test_result
* plot_latent
'''


# color templates of 6 types of responses
colors = ['#D98880', '#AF7AC5', '#5499C7', '#48C9B0', '#F4D03F', '#99A3A4']

# D = pickle.load(open("results/itemExchange/change_first_item.pkl", "rb"))
def plot_item_change(preds, targets):
    '''
    @input:
    - preds: L * nChange
    '''
    fig, axs = plt.subplots(6, preds.shape[1], sharex='col', sharey='row', figsize = (2 * preds.shape[1], 6),
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    if preds.shape[1] == 1:
        for i in range(6):
            selectedPreds = preds[targets==i,0]
            ax = axs[i]
            ax.hist(selectedPreds, color = colors[i], bins = 100, label = i)
            ax.set_yticklabels([])
        axs[5].set_xlabel("Change no item", fontsize=18)
    else:
        for i in range(6):
            selectedPreds = preds[targets==i]
            for j in range(preds.shape[1]):
                ax = axs[i,j]
                ax.hist(selectedPreds[:,j], color = colors[i], bins = 100, label = i)
                ax.set_yticklabels([])
        axs[5,0].set_xlabel("Original", fontsize=18)
        axs[5,5].set_xlabel("Perturb All", fontsize=18)
        for j in range(preds.shape[1]-2):
            axs[5,j+1].set_xlabel("Perturb " + str(j+1), fontsize=18)
    plt.show()
    fig.savefig("results/plots/change_first_item.pdf", bbox_inches='tight')

def plot_slate_response_distribution(responses, shape = (10,4), fig_name = "results/dist.pdf"): 
    respDist = np.zeros((responses.shape[0], 32))
    for i in range(responses.shape[0]):
        R = responses[i]
        totalResp = 0
        for v in R:
            totalResp *= 2
            totalResp += int(v)
        respDist[i,totalResp] += 1
    data = np.sum(respDist,0)
    # plot
    plt.figure(figsize = shape)
    plt.bar(np.arange(len(data)), data)
    plt.xticks(np.arange(len(data)), np.arange(respDist.shape[1]))
    build_path(fig_name)
    plt.savefig(fig_name)
    plt.show()

def set_box_color(bp, color):
    """
    box plot set color
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

'''
Plot latent distribution
'''

def plot_latent_box(reports, data_key = "", plot_key = "z"):
    '''
    @input:
    - reports: {model_key: {beta: L * dim}}
    - data_key: string for image saving
    - plot_key: one of {"z": zList, "mu": muList, "logvar": logvarList, "prior_mu": pMuList, "prior_logvar": pLogvarList}
    '''
    assert plot_key == "z" or plot_key == "mu" or plot_key == "prior_mu"
    nCol = 6
    nRow = 6
    modelKeys = list(reports.keys())
    for j in range(len(modelKeys)):
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]
        betaList = sorted(list(modelReport.keys()))
        # each model save a separate figure
        fig, axs = plt.subplots(6,6, figsize = (15,15))
        gs1 = gridspec.GridSpec(6,6)
        gs1.update(hspace=0.0)
        
        L = modelReport[betaList[0]][plot_key].shape[0]
        # sample from results
        filteredIndex = np.random.choice(L, min(max(500, int(0.005 * L)), L), replace = False)
        for i in tqdm(range(len(betaList))):
            beta = betaList[i]
            data = modelReport[beta][plot_key]
            resp = modelReport[beta]["resp"]
            # TSNE with 2 components
            tsne_embedded = TSNE(n_components=2).fit_transform(data[filteredIndex,:])
            # plot
            ax = axs[int(i/6),i%6]
            ax.scatter(tsne_embedded[:,0], tsne_embedded[:,1], c = resp[filteredIndex], s = 2.0)
            ax.set_title("Beta = %.5f"% beta)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
#         axs[5,5].legend(fontsize = 14)
#         plt.show()
        print("save figure \'results/plots/latent_" + plot_key + "_" + data_key + "_" + model_key + ".pdf\'")
        savePath = "results/plots/latent_" + plot_key + "_" + data_key + "_" + model_key + ".pdf"
        build_path(savePath)
        fig.savefig(savePath, bbox_inches='tight')
        plt.close()
            
def plot_latent_tsne(reports, data_key = "", plot_key = "z"):
    '''
    @input:
    - reports: {model_key: {beta: L * dim}}
    - data_key: string for image saving
    - plot_key: one of {"z": zList, "mu": muList, "logvar": logvarList, "prior_mu": pMuList, "prior_logvar": pLogvarList}
    '''
    assert plot_key == "z" or plot_key == "mu" or plot_key == "prior_mu"
    
    nCol = 6
    nRow = 6
    modelKeys = list(reports.keys())
    for j in range(len(modelKeys)):
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]
        betaList = sorted(list(modelReport.keys()))
        # each model save a separate figure
        fig, axs = plt.subplots(6,6, figsize = (15,15))
        gs1 = gridspec.GridSpec(6,6)
        gs1.update(hspace=0.0)
        
        L = modelReport[betaList[0]][plot_key].shape[0]
        # sample from results
        filteredIndex = np.random.choice(L, min(max(1000, int(0.005 * L)), L), replace = False)
        for i in tqdm(range(len(betaList))):
            beta = betaList[i]
            data = modelReport[beta][plot_key]
            resp = modelReport[beta]["resp"]
            # TSNE with 2 components
            tsne_embedded = TSNE(n_components=2).fit_transform(data[filteredIndex,:])
            # plot
            ax = axs[int(i/6),i%6]
#             print(resp[filteredIndex])
#             input()
            ax.scatter(tsne_embedded[:,0], tsne_embedded[:,1], c = [colors[int(k)] for k in resp[filteredIndex]], s = 2.0)
            ax.set_title("Beta = %.5f"% beta)
            ax.set_xticks([],[])
            ax.set_yticks([],[])
#         axs[5,5].legend(fontsize = 14)
#         plt.show()
        bll = len(betaList)
        ax = axs[int(bll/6), bll%6]
#         scatter = ax.scatter([0 for i in range(6)], [0 for i in range(6)], c = [float(i) for i in range(6)], s = 0.1)
        scl = [ax.scatter([i],[i], c = [colors[i]], s = 2.0, label = str(i)) for i in range(6)]
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.legend()
        print("save figure \'results/plots/latent_" + plot_key + "_" + data_key + "_" + model_key + ".pdf\'")
        savePath = "results/plots/latent_" + plot_key + "_" + data_key + "_" + model_key + ".pdf"
        build_path(savePath)
        fig.savefig(savePath, bbox_inches='tight')
        plt.close()

def plot_model(D, L, offset, color, ax, beta_list, detail, model_key = ""):
    if len(model_key) == 0:
        if detail:
            bp = ax.boxplot(D, positions=[i * 2.0 + 0.4 + 1.7 * offset / L for i in range(len(beta_list))], \
                            widths=1.5/L, showfliers=False)
            set_box_color(bp, color)
        else:
            ax.plot(np.arange(len(beta_list)) * 2.0 + 1.0, D, c = color)
    else:
        if detail:
            a = 1.0 * offset / L
            bp = ax.boxplot(D, positions=[i * 2.0 + 0.4 + 1.7 * offset / L for i in range(len(beta_list))], \
                            widths=1.5/L, showfliers=False)
            ax.legend([bp["boxes"][0]], [model_key])
            set_box_color(bp, color)
        else:
            ax.plot(np.arange(len(beta_list)) * 2.0 + 1.0, D, label = model_key, c = color)

            
def plot_likelihood_test_result(reports, detail = False, data_key = ""):
    if detail:
        fig, axs = plt.subplots(3,1, figsize = (16,6), sharex='col')
    else:
        fig, axs = plt.subplots(3,1, figsize = (8,6), sharex='col')
        
    gs1 = gridspec.GridSpec(3,1)
    gs1.update(hspace=0.0)
    bpPlots = []
    modelKeys = list(reports.keys())
    for j in range(len(modelKeys)):
        # plot for each model
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]
        betaList = sorted(list(modelReport.keys()))
        recs = []
        klds = []
        vaes = []
        for i in range(len(betaList)):
            beta = betaList[i]
            recs.append(modelReport[beta]["rec"])
            klds.append(np.mean(np.log(modelReport[beta]["kld"])))
            vaes.append(modelReport[beta]["vae"])
        # box plot of ENC and diversity
        plot_model(recs, len(modelKeys), j, model_color, axs[0], betaList, detail)
        # plot of coverage
        plot_model(klds, len(modelKeys), j, model_color, axs[1], betaList, detail = False, model_key = model_key)
        # plot of ILD
        plot_model(vaes, len(modelKeys), j, model_color, axs[2], betaList, detail)
        
    axs[0].set_ylabel("Reconstruction Loss", fontsize = 12 + len(modelKeys))
    axs[1].set_ylabel("Log KLD", fontsize = 12 + len(modelKeys))
    axs[1].legend(fontsize = 12 + len(modelKeys))
    axs[2].set_ylabel("VAE Loss", fontsize = 12 + len(modelKeys))
    
    plt.xticks(np.arange(len(betaList)) * 2.0 + 1.0, DEFAULT_BETA_TICKS)
    plt.xlim(0, len(betaList) * 2.0 + 1.0)
    plt.show()
    savePath = "results/plots/likelihood_test_" + data_key + ".pdf"
    build_path(savePath)
    fig.savefig(savePath, bbox_inches='tight')

def plot_standard_test_result(reports, detail = False, data_key = ""):
    modelKeys = list(reports.keys())
    betaList = sorted(list(reports[modelKeys[0]].keys()))
    if detail:
        fig, axs = plt.subplots(3,1, figsize = (int(5*len(modelKeys)*len(betaList)/35),6), sharex='col')
    else:
        fig, axs = plt.subplots(3,1, figsize = (int(3*len(modelKeys)*len(betaList)/35),6), sharex='col')
        
    gs1 = gridspec.GridSpec(3,1)
    gs1.update(hspace=0.0)
    bpPlots = []
    for j in range(len(modelKeys)):
        # plot for each model
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]

        coverages = []
        encs = []
        diversities = []
        for i in range(len(betaList)):
            beta = betaList[i]
            encs.append(modelReport[beta]["enc"][-1])
            diversities.append(modelReport[beta]["diversity"][-1])
            coverages.append(modelReport[beta]["coverage"][-1])
        # box plot of ENC and diversity
        plot_model(encs, len(modelKeys), j, model_color, axs[0], betaList, detail)
        # plot of coverage
        plot_model(coverages, len(modelKeys), j, model_color, axs[1], betaList, detail = False, model_key = model_key)
        # plot of ILD
        plot_model(diversities, len(modelKeys), j, model_color, axs[2], betaList, detail)
#             # box plot of ENC and diversity
#             ax = axs[0]
#             bp = ax.boxplot(enc, positions=[i * 1.0 * len(modelKeys) + 0.4 + 0.8 * j], \
#                             widths=0.6, showfliers=False)
#             set_box_color(bp, model_color)
#             # plot of ILD
#             ax = axs[2]
#             bp = ax.boxplot(diversity, positions=[i * 1.0 * len(modelKeys) + 0.4 + 0.8 * j], \
#                             widths=0.6, showfliers=False)
#             set_box_color(bp, model_color)
# #                 sp = ax.plot(np.arange(len(beta_list)) * 2.0 + 1.0, diversities, label = legends[j], c = colors[j])
#             # plot of coverage
#             ax = axs[1]
#             cp = ax.plot(np.arange(len(betaList)) * 2.0 + 1.0, coverages, label = model_key, c = model_color)
        
        
    axs[0].set_ylabel("ENC", fontsize = 12 + len(modelKeys))
#     axs[0].legend([bpPlots[0]["boxes"][0], bpPlots[1]["boxes"][0]], legends)
    axs[1].set_ylabel("Coverage", fontsize = 12 + len(modelKeys))
    axs[1].legend(fontsize = 12 + len(modelKeys))
    axs[2].set_ylabel("Diversity", fontsize = 12 + len(modelKeys))
#     axs[2].legend()
    
    plt.xticks(np.arange(len(betaList)) * 2.0 + 1.0, DEFAULT_BETA_TICKS, fontsize = 10 + len(modelKeys))
    plt.xlim(0, len(betaList) * 2.0 + 1.0)
    plt.show()
    savePath = "results/plots/standard_test_" + data_key + ".pdf"
    build_path(savePath)
    fig.savefig(savePath, bbox_inches='tight')
    
    
def plot_ranking_test_result(reports, data_key, plot_key = "hit_rate", slate_size = 5):
    '''
    @input:
    - plot_key: ["hit_rate", "recall", "p_hit_rate"], "p_hit_rate" means positional hit rate
    '''
    modelKeys = list(reports.keys())
    betaList = sorted(list(reports[modelKeys[0]].keys()))
    fig, axs = plt.subplots(slate_size, 1, figsize = (int(5*len(modelKeys)*len(betaList)/35),6), sharex='col')

    gs1 = gridspec.GridSpec(slate_size, 1)
    gs1.update(hspace=0.0)

    for j in range(len(modelKeys)):
        # plot for each model
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]
        # each model have a beta test
        betaList = sorted(list(modelReport.keys()))
        rs = np.zeros((len(betaList),slate_size))
        for i in range(len(betaList)):
            beta = betaList[i]
            rs[i] = modelReport[beta][plot_key]
        # box plot of ENC and diversity
        for i in range(slate_size):
            if i == 1:
                axs[i].plot(np.arange(len(betaList)) * 2.0 + 1.0, rs[:,i], c = model_color, label = PLOT_NAME[model_key])
            else:
                axs[i].plot(np.arange(len(betaList)) * 2.0 + 1.0, rs[:,i], c = model_color)
    for i in range(slate_size):
        axs[i].set_ylabel("@" + str(i+1), fontsize = 12 + len(modelKeys))
#     axs[0].legend([bpPlots[0]["boxes"][0], bpPlots[1]["boxes"][0]], legends)
#     axs[1].set_ylabel("Coverage", fontsize = 14)
    axs[1].legend(fontsize = 14)
#     axs[2].set_ylabel("Diversity", fontsize = 14)
#     axs[2].legend()
    
    plt.xticks(np.arange(len(betaList)) * 2.0 + 1.0, DEFAULT_BETA_TICKS, fontsize = 12 + len(modelKeys))
    plt.xlim(0, len(betaList) * 2.0 + 1.0)
    plt.show()
    savePath = "results/plots/ranking_test_" + data_key + "_" + plot_key + ".pdf"
    build_path(savePath)
    fig.savefig(savePath, bbox_inches='tight')
    
def plot_positional_ranking_test_result(reports, data_key, plot_key = "p_recall", slate_size = 5):
    assert plot_key == "p_recall"
    nCol = 6
    nRow = 6
    modelKeys = list(reports.keys())
    for j in range(len(modelKeys)):
        model_key = modelKeys[j]
        modelReport = reports[model_key]
        model_color = PLOT_COLOR[model_key]

        fig, axs = plt.subplots(6,6, figsize = (15,15), sharex='col', sharey='row')
        # each model save a separate figure
        gs1 = gridspec.GridSpec(6,6)
        gs1.update(hspace=0.0)
        
        betaList = sorted(list(modelReport.keys()))
        for i in tqdm(range(len(betaList))):
            beta = betaList[i]
            rs = modelReport[beta][plot_key]
            ax = axs[int(i/6),i%6]
            for j in range(rs.shape[0]):
                ax.plot(rs[j], color = colors[j])
            ax.set_title("Beta = %.5f"% beta)
        print("save figure \'results/plots/ranking_" + data_key + "_" + plot_key + "_" + model_key + ".pdf\'")
        savePath = "results/plots/ranking_" + data_key + "_" + plot_key + "_" + model_key + ".pdf"
        buildPath(savePath)
        fig.savefig(savePath, bbox_inches='tight')
        plt.close()
            
            
    