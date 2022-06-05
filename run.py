import experiment_buddy
import os
import random
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFKC, BertNormalizer, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer
from torch import nn, Tensor, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import getpass
from classification_baseline_attention_model import ClassificationAttentionModel
from classification_baseline_attention_model import get_tokenizer
from classification_baseline_attention_model import get_initial_embedding
from classification_baseline_attention_model import get_loaders
from classification_baseline_attention_model import train_and_evaluate

# from classification_baseline_attention_model import
# from logger import Logger
# from tensor_utils import generate_padding_mask
# from training_watcher import TrainingWatcher


experiment_buddy.register_defaults({'task': 'document classification with attention'})
writer = experiment_buddy.deploy(
    host="", disabled=False, wandb_kwargs={'project': "nlp"},
    sweep_definition="sweep.yaml"
)

whoami = getpass.getuser()
home = str(Path.home())
print("home:", home)
if whoami == 'ionelia':
    root_data = "/home/ionelia/pycharm-projects/NLPwDL/assignment#2/data/nlpwdl2021_data/thedeep"
    embedding_path = f"{home}/pycharm-projects/NLPwDL/assignment#2/local/nlp_experiments/experiments2/glove/glove.840B" \
                     f".300d.txt"
    embedding_save_path = f"{home}/pycharm-projects/NLPwDL/assignment#2/local/nlp_experiments/experiments2" \
                          f"/initial_embedding.npy"
    tokenizer_save_path = 'tokenizer.json'
elif whoami == 'ionelia.buzatu':
    from pathlib import Path

    root_data = f"{home}/nlp/data/nlpwdl2021_data/thedeep"
    embedding_path = f"{home}/nlp/glove/glove.840B.300d.txt"
    embedding_save_path = f"{home}/nlp/initial_embedding.npy"
    tokenizer_save_path = f"{home}/nlp/tokenizer.json"


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
os.system('nvidia-smi')
name_dataset_size = "small"
optimizers = {'adam': Adam, 'sgd': torch.optim.SGD}


batch_size = 32
document_length = -1
hidden_size = 512
weight_decay = 0.01
lr = 0.001
momentum = 0.95
optim = 'adam'

tokenizer = get_tokenizer(data_path=f'{root_data}.{name_dataset_size}.train.txt', save_path=tokenizer_save_path)
embedding = get_initial_embedding(path=embedding_path, tokenizer=tokenizer, save_path=embedding_save_path)

model = ClassificationAttentionModel(
    embedding.weight.clone(),
    attention_type='dot',
    hidden_size=hidden_size,
    batch_size=batch_size
)
model = model.to(device)

train_loader, val_loader, test_loader = get_loaders(
    root_data, name_dataset_size, batch_size, tokenizer, document_length=document_length)

if optimizers['adam']:
    optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if optimizers['sgd']:
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

test_acc = train_and_evaluate(model, optim, train_loader, val_loader, test_loader, writer)

print(f'Test accuracy: {test_acc:.4f}')
