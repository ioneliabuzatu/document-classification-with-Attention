import experiment_buddy


lr = 1e-4
which_optimizer = 'adam'
attention = "dot"
batch_size = 32
hidden_size = 512
document_length = -1
epochs = 50
weight_decay = 0.0

experiment_buddy.register(locals())
writer = experiment_buddy.deploy(
    host="mila", disabled=False, wandb_kwargs={'project': "nlp"},
    sweep_definition="sweep.yaml",
    proc_num=100
)
