
# which GPU
DEFAULT_GPU = "cuda:1"

# model variations and the model class it belongs to
modelType = {"list_cvae_with_prior": "list_cvae_with_prior", \
              "slate_cvae": "slate_cvae", \
              "slate_cvae_offset":"slate_cvae", \
              "slate_cvae_pre": "slate_cvae"}

# default settings
DEFAULT_PARAMS = {
    "doc_emb_size": 8,
    "gpu": True,
    "resp_model": {
        "doc_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 50,
        "batch_size": 64,
        "mlp_struct": [40, 1024, 5]
    },
    "user_resp_model": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.03,
        "weight_decay": 0.0001,
        "epoch_limit": 30,
        "batch_size": 64,
        "mlp_struct": [48, 256, 256, 5]
    },
    "mf_model": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "epoch_limit": 30,
        "batch_size": 256
    },
    "user_pointwise_resp_model": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 30,
        "batch_size": 64,
        "interaction_mlp_struct": [48, 512, 8],
        "mlp_struct": [24, 512, 1]
    },
    "list_cvae": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 40,
        "batch_size": 64,
        "latent_size": 16,
        "slate_size": 5,
        "condition_size": 6,
        "encoder_struct": [46, 128],
        "decoder_struct": [22, 128, 40],
        "cond_struct": [5, 6],
        "beta": 0.0003
    },
    "list_ae": {
        "doc_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 100,
        "batch_size": 64,
        "latent_size": 16,
        "slate_size": 5,
        "encoder_struct": [40, 1024, 16],
        "decoder_struct": [16, 1024, 40]
    },
    "list_vae": {
        "doc_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 40,
        "batch_size": 64,
        "latent_size": 16,
        "slate_size": 5,
        "encoder_struct": [40, 1024, 16],
        "decoder_struct": [16, 1024, 40],
        "beta": 0.000001
    },
    "list_cvae_with_prior": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 20,
        "batch_size": 64,
        "latent_size": 16,
        "slate_size": 5,
        "condition_size": 6,
#         "encoder_struct": [46, 128, 128, 128],
#         "decoder_struct": [22, 128, 128, 128, 40],
#         "cond_struct": [5, 6],
#         "prior_struct": [6, 128, 128],
        "encoder_struct": [46, 256, 64],
        "decoder_struct": [22, 256, 256, 40],
        "cond_struct": [5, 6],
        "prior_struct": [6, 256],
        "beta": 0.002
    },
    "slate_cvae": {
        "doc_emb_size": 8,
        "user_emb_size": 8,
        "gpu": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epoch_limit": 20,
        "batch_size": 64,
        "latent_size": 16,
        "slate_size": 5,
        "condition_size": 6,
        "encoder_struct": [46, 256, 64],
        "psm_struct": [22, 256, 256, 8],
        "scm_struct": [30, 256, 256, 32],
        "cond_struct": [5, 6],
        "prior_struct": [6, 256],
#         "encoder_struct": [46, 128, 128],
#         "psm_struct": [22, 128, 128, 8],
#         "scm_struct": [30, 128, 128, 32],
#         "cond_struct": [5, 6],
#         "prior_struct": [6, 128],
        "beta": 0.1
    },
    "urm": {
        "latent_size": 8,
        "slate_size": 5,
        "gpu": False,
        "device": "cuda:1"
    },
    "urm_p": {
        "latent_size": 8,
        "slate_size": 5,
        "p_bias_factor": 1.0,
        "p_bias_max": 0.3,
        "p_bias_min": -0.2,
        "gpu": False,
        "device": "cuda:1"
    },
    "urm_p_br": {
        "latent_size": 8,
        "slate_size": 5,
        "p_bias_factor": 1.0,
        "p_bias_max": 0.3,
        "p_bias_min": -0.2,
        "br_bias_factor": 1.0,
        "gpu": False,
        "device": "cuda:1"
    },
    "urm_p_mr": {
        "latent_size": 8,
        "slate_size": 5,
        "p_bias_factor": 1.0,
        "p_bias_max": 0.3,
        "p_bias_min": -0.2,
        "mr_bias_factor": 0.2,
        "gpu": False,
        "device": "cuda:1"
    },
    "urm_p_mr_bigrho": {
        "latent_size": 8,
        "slate_size": 5,
        "p_bias_factor": 1.0,
        "p_bias_max": 0.3,
        "p_bias_min": -0.2,
        "mr_bias_factor": 5.0,
        "gpu": False,
        "device": "cuda:1"
    }
}