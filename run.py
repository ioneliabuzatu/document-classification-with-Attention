import getpass
import os
import random
from pathlib import Path

import config
import numpy as np
import torch
from torch.optim import Adam

from classification_baseline_attention_model import ClassificationAttentionModel
from classification_baseline_attention_model import get_initial_embedding
from classification_baseline_attention_model import get_loaders
from classification_baseline_attention_model import get_tokenizer
from classification_baseline_attention_model import train_and_evaluate


whoami = getpass.getuser()
home = str(Path.home())
print("home:", home)
if whoami == 'ionelia':
    root_data = "/home/ionelia/pycharm-projects/NLPwDL/assignment#2/data/nlpwdl2021_data/thedeep"
    glove_path = f"{home}/pycharm-projects/NLPwDL/assignment#2/local/nlp_experiments/experiments2/glove/glove.840B" \
                     f".300d.txt"
    embedding_save_path = f"{home}/pycharm-projects/NLPwDL/assignment#2/local/nlp_experiments/experiments2" \
                          f"/initial_embedding.npy"
    tokenizer_save_path = 'tokenizer.json'
    checkpoint_path = ""
elif whoami == 'ionelia.buzatu':
    from pathlib import Path

    root_data = f"{home}/nlp/data/nlpwdl2021_data/thedeep"
    glove_path = f"{home}/nlp/glove/glove.840B.300d.txt"
    embedding_save_path = f"{home}/nlp/initial_embedding.npy"
    tokenizer_save_path = f"{home}/nlp/tokenizer.json"
    checkpoint_path = f"{home}/nlp/"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
os.system('nvidia-smi')
name_dataset_size = "small"


tokenizer = get_tokenizer(data_path=f'{root_data}.{name_dataset_size}.train.txt', save_path=tokenizer_save_path)
embedding = get_initial_embedding(path=glove_path, tokenizer=tokenizer, save_path=embedding_save_path)

model = ClassificationAttentionModel(
    embedding.weight.clone(),
    attention_type=config.attention,
    hidden_size=config.hidden_size,
    batch_size=config.batch_size,
    checkpoint_path=checkpoint_path,
    epochs=config.epochs
).to(device)

train_loader, val_loader, test_loader = get_loaders(
    root_data, name_dataset_size, config.batch_size, tokenizer, document_length=config.document_length)

if config.which_optimizer == 'adam':
    weight_decay = config.weight_decay
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=weight_decay)
if config.which_optimizer == 'sgd':
    momentum = 0.95
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=momentum)

test_acc = train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, config.writer)

print(f'Test accuracy: {test_acc:.4f}')
