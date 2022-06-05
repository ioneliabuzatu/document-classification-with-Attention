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
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

UNK_TOKEN = '[UNK]'
PAD_TOKEN = '[PAD]'


class DocumentDataset(Dataset):
    def __init__(self, path, tokenizer, document_length=300):
        super().__init__()

        self.document_length = document_length
        self.documents = []
        self.labels = []
        self.path = path
        self.tokenizer = tokenizer

        with open(self.path, 'r', encoding='UTF8') as f:
            lines = f.readlines()

        for line in lines:
            text, label = self.__preprocess_line(line)
            self.documents.append(text)
            self.labels.append(label)

        self.classes = set(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.document_length > 0:
            token_ids = torch.zeros(self.document_length) + self._tokenize_ids(PAD_TOKEN)[0]
            document = self._tokenize_ids(self.documents[index])

            if document.shape[0] > self.document_length:
                document = document[:self.document_length]

            token_ids[:document.shape[0]] = torch.from_numpy(document)
        else:
            token_ids = torch.from_numpy(self._tokenize_ids(self.documents[index]))

        return token_ids, torch.from_numpy(np.array(self.labels[index]))

    @staticmethod
    def __preprocess_line(line: str) -> Tuple[str, int]:
        line = line.split(',', 1)[1]
        text, label = line.rsplit(',', 1)
        return text.strip('"'), int(label)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _tokenize_ids(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text).ids)

    def find_examples(self, model, k: int = 2, seed: int = 42) -> Tuple[List[int], List[int]]:
        correct = []
        misclassified = []

        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)

        for i in indices:
            token_ids, label = self[i]
            prediction = model.predict((token_ids.unsqueeze(0), Tensor([token_ids.shape[-1]]))).argmax()

            if len(correct) < k and prediction.item() == label:
                correct.append(i)
            elif len(misclassified) < k:
                misclassified.append(i)

            if len(misclassified) == k and len(correct) == k:
                break

        return correct, misclassified


def get_tokenizer(data=None, data_path=None, save_path=None, vocab_size=25_000) -> Tokenizer:
    if save_path is not None and os.path.exists(save_path):
        tokenizer = Tokenizer.from_file(save_path)
        print(f'Loaded tokenizer. Vocabulary size: {tokenizer.get_vocab_size()}.')
        return tokenizer

    if data is None:
        data = []

        with open(data_path, 'r', encoding='UTF8') as f:
            for line in f.readlines():
                line = line.split(',', 1)[1]
                text = line.rsplit(',', 1)[0].strip('"')
                data.append(text)

    tokenizer = Tokenizer(WordPiece(vocab={UNK_TOKEN: 1}, unk_token=UNK_TOKEN))

    tokenizer.normalizer = Sequence([NFKC(), BertNormalizer()])
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = WordPieceDecoder()

    trainer = WordPieceTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=[UNK_TOKEN, PAD_TOKEN])
    tokenizer.train_from_iterator(data, trainer=trainer)

    if save_path is not None:
        tokenizer.save(save_path)

    print(f'Trained tokenizer. Vocabulary size: {tokenizer.get_vocab_size()}.')

    return tokenizer


def pad_collate(padding_value: int):
    def collate_fn(batch: List[Tensor]):
        inputs, targets = zip(*batch)
        input_lengths = Tensor([len(x) for x in inputs])

        inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        targets = torch.stack(targets)

        return inputs, targets, input_lengths

    return collate_fn


def get_initial_embedding(path, tokenizer: Tokenizer, save_path: Union[str, Path, None] = None) -> nn.Embedding:
    if save_path is not None and os.path.exists(save_path):
        weight = np.load(save_path)
        embedding = nn.Embedding(weight.shape[0], weight.shape[1])
        embedding.weight = nn.Parameter(torch.from_numpy(weight))
        return embedding

    pre_trained = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
    embedding_size = pre_trained.shape[1]
    embedding_weigths = torch.zeros(tokenizer.get_vocab_size(), embedding_size)

    not_contained = 0
    for word, index in tqdm(tokenizer.get_vocab().items()):
        if word in pre_trained.index:
            embedding_weigths[index] = torch.from_numpy(np.array(pre_trained.loc[word]))
        else:
            not_contained += 1
            embedding_weigths[index] = torch.rand(embedding_size)

    if save_path is not None:
        np.save(save_path, embedding_weigths.detach().numpy())

    print(f'Initialized {not_contained} token(s) randomly as they are not part of the pre-trained embeddings.')

    embedding = nn.Embedding(embedding_weigths.shape[0], embedding_weigths.shape[1])
    embedding.weight = nn.Parameter(embedding_weigths)

    return embedding


def accuracy():
    def metric(y_hat: np.ndarray, y: np.ndarray) -> float:
        return (y_hat.argmax(-1) == y).sum() / y_hat.shape[0]

    return metric


def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, logger):

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        logger=logger,
        validation_metrics={
            'accuracy': accuracy()
        }
    )

    # model.resume_from_checkpoint(logger.best_checkpoint_path)

    _, metrics = model.evaluate(
        loader=test_loader,
        metrics={
            'accuracy': accuracy()
        }
    )

    return metrics['accuracy']


class ClassificationAttentionModel(nn.Module):
    def __init__(
            self,
            embedding_weights: Tensor,
            batch_size: 32,
            num_classes=12,
            hidden_size: int = 512,
            num_layers: int = 2,
            dropout: float = .5,
            freeze_embeddings: bool = False,
            bidirectional: bool = True,
            attention_type: str = 'dot',
            epochs=10,
            patience=5,
            checkpoint_path=None
    ):
        """
        :attention_type: can be either 'dot' or 'additive'
        """
        super().__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.best_accuracy_so_far = None
        self.delta = 0.0
        self.counter = 0
        self.attention_type = attention_type
        self.checkpoint_path = checkpoint_path

        self.vocab_size, self.embedding_size = embedding_weights.shape
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=not freeze_embeddings)

        if hidden_size > 0:
            self.backbone = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

            if bidirectional:
                self.hidden_size = hidden_size * 2
        else:
            self.backbone = None
            self.hidden_size = self.embedding_size  # or hidden_size

        single_head_attention_methods = {
            'dot': self.dot_product_attention,
            'additive': self.additive_attention,
            'multiplicative': self.multiplicative_attention
        }
        self.calculate_attention_weights = single_head_attention_methods[self.attention_type]
        self.query_dim = self.hidden_size

        self.query = nn.Parameter(torch.randn(size=(1, 1, self.query_dim)))
        self.scale = 1.0 / np.sqrt(self.query_dim)

        if attention_type == 'additive':
            self.W_queries = nn.Linear(self.query_dim, self.query_dim, bias=False)
            self.W_values = nn.Linear(self.query_dim, self.query_dim, bias=False)
            self.u = nn.Parameter(torch.randn(1, self.query_dim, 1))

        elif attention_type == 'multiplicative':
            self.W = nn.Parameter(torch.randn(self.query_dim, self.query_dim))

        self.linear = nn.Linear(self.hidden_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, input_lengths = inputs
        batch_size, sequence_length = inputs.shape

        embedded_input = self.embedding(inputs.long().to(self.device))
        assert embedded_input.shape == (batch_size, sequence_length, self.embedding_size)

        output = self.run_backbone(embedded_input, input_lengths)
        assert output.shape == (batch_size, sequence_length, self.hidden_size)

        padding_mask = self.generate_padding_mask(batch_size, sequence_length, input_lengths).to(self.device)
        assert padding_mask.shape == (batch_size, sequence_length)

        queries = self.query.repeat(batch_size, 1, 1)
        assert queries.shape == (batch_size, 1, self.query_dim)
        attention_weights = self.calculate_attention_weights(queries, output) * self.scale
        attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
        normed_attention_weights = F.softmax(attention_weights, dim=-1)
        assert normed_attention_weights.shape == (batch_size, 1, sequence_length)
        output = (normed_attention_weights @ output).squeeze(1)

        return self.linear(output.squeeze(1)), normed_attention_weights

    def run_backbone(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        if self.backbone is not None:
            embedded = pack_padded_sequence(input=inputs, lengths=input_lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.backbone(embedded)
            output, _ = pad_packed_sequence(output, batch_first=True)
            assert output.shape == (*inputs.shape[:2], self.hidden_size)
        else:
            output = inputs

        return output

    def generate_padding_mask(self, batch_size: int, sequence_length: int, input_lengths: Tensor) -> Tensor:
        mask = torch.zeros(batch_size, sequence_length).bool()
        for i, input_length in enumerate(input_lengths):
            mask[i, int(input_length):] = True
        return mask

    def dot_product_attention(self, queries, values) -> Tensor:
        attention_weights = queries @ values.permute(0, 2, 1)
        return attention_weights

    def multiplicative_attention(self, queries, values) -> Tensor:
        return queries @ self.W @ values.permute(0, 2, 1)

    def additive_attention(self, queries, values) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        queries = queries.repeat(1, sequence_length, 1)
        assert queries.shape == (batch_size, sequence_length, self.hidden_size)
        output = (torch.tanh(self.W_queries(queries) + self.W_values(values)) @ self.u).permute(0, 2, 1)
        assert output.shape == (batch_size, 1, sequence_length)
        return output

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def fit(self, train_loader, val_loader, optimizer, logger, validation_metrics: Union[dict, None] = None):
        if validation_metrics is None:
            validation_metrics = {}

        initial_loss, initial_accuracy = self.evaluate(loader=val_loader, metrics=validation_metrics)
        logger.add_scalar("validation_loss", initial_loss, 0)
        logger.add_scalar("validation_accuracy", initial_accuracy['accuracy'], 0)

        for epoch in tqdm(range(1, self.epochs)):
            average_loss_epoch = self.__step(train_loader, optimizer)
            validation_loss, val_accuracy = self.evaluate(loader=val_loader, metrics=validation_metrics)

            logger.add_scalar("training_epochs_loss", average_loss_epoch, epoch - 1)
            logger.add_scalar("validation_loss", validation_loss, epoch)
            logger.add_scalar("validation_accuracy", val_accuracy['accuracy'], epoch)
            print(f"done epoch {epoch} accuracy | {val_accuracy['accuracy']}.")

            if self.early_stopping(validation_loss):
                break

    def __step(self, data_loader, optimizer) -> float:
        self.train()
        average_loss = 0.

        for i, batch in enumerate(data_loader):
            inputs, targets = self.__unpack_batch(batch)
            optimizer.zero_grad()
            loss, _ = self._calculate_loss(inputs, targets)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            average_loss += (loss - average_loss) / (i + 1)

        return average_loss

    @torch.no_grad()
    def evaluate(self, loader, metrics: dict) -> Tuple[float, dict]:
        self.eval()

        average_loss = 0.
        predictions = None
        targets = None

        for i, batch in enumerate(loader):
            x, y = self.__unpack_batch(batch)

            loss, y_hat = self._calculate_loss(x, y)
            average_loss += (loss.item() - average_loss) / (i + 1)

            if predictions is None:
                predictions = y_hat.cpu().numpy()
                targets = y.numpy()
            else:
                predictions = np.concatenate((predictions, y_hat.cpu().numpy()))
                targets = np.concatenate((targets, y.numpy()))

        metric_values = {name: metric(predictions, targets) for name, metric in metrics.items()}

        return average_loss, metric_values

    @staticmethod
    def __unpack_batch(batch: List) -> Tuple[Union[Tuple, Tensor], Tensor]:
        if len(batch) == 3:  # padded sequence
            inputs, targets, input_lengths = batch
            inputs = (inputs, input_lengths)
        else:
            inputs, targets = batch

        return inputs, targets

    def _calculate_loss(self, inputs: Tuple[Tensor, Tensor], targets: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        logits, _ = self(inputs)
        return F.cross_entropy(logits, targets.long().to(self.device)), logits

    def predict(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        logits, _ = self(x)
        return torch.softmax(logits, dim=-1)

    def resume_from_checkpoint(self, path: Union[Path, str]):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def early_stopping(self, curr_validation_accuracy):
        """
        Stops the training if validation accuracy doesn't improve after a given patience.
        """

        if self.best_accuracy_so_far is None:
            self.counter = 0
            self.best_accuracy_so_far = curr_validation_accuracy
            torch.save(self.state_dict(), f"{self.attention_type}.{self.checkpoint_path}.best")

        elif curr_validation_accuracy < self.best_accuracy_so_far + self.delta:
            self.counter += 1
            print(f'EarlyStopper counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print("early stopping triggered. training stopped.")
                return True
        else:
            print(f'Val acc increased ({self.best_accuracy_so_far:.6f} --> {curr_validation_accuracy:.6f}),model saved')
            self.best_accuracy_so_far = curr_validation_accuracy
            torch.save(self.state_dict(), self.path)
            self.counter = 0

        return False


def get_loaders(root_data, name_dataset_size, batch_size, tokenizer, document_length=-1):
    train_set = DocumentDataset(
        path=f'{root_data}.{name_dataset_size}.train.txt',
        tokenizer=tokenizer,
        document_length=document_length,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              collate_fn=pad_collate(tokenizer.token_to_id(PAD_TOKEN)))
    val_set = DocumentDataset(
        path=f'{root_data}.{name_dataset_size}.validation.txt',
        tokenizer=tokenizer,
        document_length=document_length
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=pad_collate(tokenizer.token_to_id(PAD_TOKEN)))
    test_set = DocumentDataset(
        path=f'{root_data}.{name_dataset_size}.test.txt',
        tokenizer=tokenizer,
        document_length=document_length
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pad_collate(tokenizer.token_to_id(PAD_TOKEN)))

    return train_loader, val_loader, test_loader
