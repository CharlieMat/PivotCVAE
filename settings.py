DATA_ROOT = "../data/"
SIMDATA_ROOT = "../sim_data/"


BETA_LIST = [0.00001, 0.0001, 0.001] +\
            [0.002 + 0.001 * i for i in range(9)] + \
            [0.1, 1.0, 10.0]
DEFAULT_BETA_TICKS = ["0.00001", "", "0.001"] + \
                     ["" for i in range(8)] + \
                     ["0.01", "0.1", "1.0", "10.0"]

# BETA_LIST = [0.00001, 0.00003, 0.0001, 0.0003] + \
#             [0.0005 + 0.0001 * i for i in range(5)] + \
#             [0.001 + 0.001 * i for i in range(10)] + \
#             [0.012 + 0.002 * i for i in range(10)] + \
#             [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# DEFAULT_BETA_TICKS = ["0.00001", "", "0.0001", "", "0.0005"] + \
#                     ["" for i in range(4)] + ["0.001"] + \
#                     ["" for i in range(8)] + ["0.01"] + \
#                     ["" for i in range(10)] + \
#                     ["0.1", "", "1.0", "", "", "30.0"]

PLOT_COLOR = {\
    "listcvaewithprior": '#D7191C',\
    "ng_listcvaewithprior": '#D7898C',\
#     "pivotcvae_gt_pi": ("Pivot CVAE (gt_pi)", '#2C5BB6'),\
#     "pivotcvae_pt_pi": ("Pivot CVAE (pt_pi)", '#3CCB46'),\
#     "pivotcvae_spt_pi": ("Pivot CVAE (spt_pi)", '#B933B9'),\
    "pivotcvae_sgt_pi": '#B9B933',\
    "pivotcvae_gt_spi": '#33B9B9',\
#     "pivotcvae_pt_spi": ("Pivot CVAE (pt_spi)", '#FF3333'),\
#     "pivotcvae_spt_spi": ("Pivot CVAE (spt_spi)", '#3333FF'),\
    "pivotcvae_sgt_spi": '#33FF33',\
}

PLOT_NAME = {\
    "listcvaewithprior": "List CVAE", \
    "pivotcvae_sgt_pi": "Pivot CVAE (SGT-PI)", \
    "pivotcvae_gt_spi": "Pivot CVAE (GT-SPI)", \
    "pivotcvae_sgt_spi": "Pivot CVAE (SGT-SPI)", \
    "ng_listcvaewithprior": "Non-greedy List CVAE", \
}

PLOT_DEFAULT = {\
    "listcvaewithprior": ("List CVAE", '#D7191C', False),\
#     "pivotcvae_gt_pi": ("Pivot CVAE (gt_pi)", '#2C5BB6'),\
#     "pivotcvae_pt_pi": ("Pivot CVAE (pt_pi)", '#3CCB46'),\
#     "pivotcvae_spt_pi": ("Pivot CVAE (spt_pi)", '#B93399'),\
    "pivotcvae_sgt_pi": ("Pivot CVAE (sgt_pi)", '#B9B933', False),\
    "pivotcvae_gt_spi": ("Pivot CVAE (gt_spi)", '#33B9B9', False),\
#     "pivotcvae_pt_spi": ("Pivot CVAE (pt_spi)", '#FF3333'),\
#     "pivotcvae_spt_spi": ("Pivot CVAE (spt_spi)", '#3333FF'),\
    "pivotcvae_sgt_spi": ("Pivot CVAE (sgt_spi)", '#33FF33', False),\
}
# colors = [, , , '#793A9D', '#A38321']

NON_GREEDY_DEFAULT = {
    "listcvaewithprior": ("List CVAE", '#D7191C', False),\
    "pivotcvae_sgt_pi": ("Pivot CVAE (sgt_pi)", '#B9B933', False),\
    "pivotcvae_gt_spi": ("Pivot CVAE (gt_spi)", '#33B9B9', False),\
    "pivotcvae_sgt_spi": ("Pivot CVAE (sgt_spi)", '#33FF33', False),\
    "ng_listcvaewithprior": ("Non-greedy List CVAE", '#D7898C', True),\
}


PLOT_RANKING_DEFAULT = {\
    "diverse_mf": ("DiverseMF", '#A38321', False),\
#     "mf": ("MF", '#D7191C', False),\
#     "neumf": ("NeuMF", '#33B9B9', False),\
}
# colors = [, , , '#793A9D', '#A38321']

NON_GREEDY_RANKING_DEFAULT = {
    "mf": ("MF", '#D7191C', False),\
    "ng_mf": ("Non-Greedy MF", '#B9B933', True),\
    "neumf": ("NeuMF", '#33B9B9', False),\
    "ng_neumf": ("Non-Greedy NeuMF", '#33FF33', True),\
    "diverse_mf": ("DiverseMF", '#A38321', False),\
}
