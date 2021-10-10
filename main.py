import time
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer,  get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from dataset.chaii_dataset import ChaiiDataset
from accelerate import Accelerator
from functools import partial

from ops.folds import create_folds, convert_answers
from ops.utils import prepare_train_features
from ops.optimizer import make_optimizer
from networks.model import Model
from train import train, evaluate

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--train_df", required=True, type=str,
    help="path to train dataframe"
)

parser.add_argument(
    "--test_df", required=True, type=str,
    help="path to test dataframe"
)

parser.add_argument(
    "--external_data_1", type=str,
    help="path to external dataframe",
)

parser.add_argument(
    "--external_data_2", type=str,
    help="path to external dataframe"
)

parser.add_argument(
    "--nfolds", type=int, default=5,
    help="number of folds"
)

parser.add_argument(
    '--model_name', type=str,
    help='name of the model'
)

parser.add_argument(
    '--max_length', type=int
)

parser.add_argument(
    '--doc_stride', type=int
)

parser.add_argument(
    '--max_answer_length', type=int
)

parser.add_argument(
    '--batch_size', type=int
)

parser.add_argument(
    '--num_workers', type=int
)

parser.add_argument(
    '--learning_rate', type=float,
)

parser.add_argument(
    '--wd', type=float
)

parser.add_argument(
    '--epochs', type=int
)

parser.add_argument(
    '--weight_decay', type=float
)

parser.add_argument(
    '--epsilon', type=float
)

parser.add_argument(
    '--seed', type=int, default=1000,
    help='kfold seed'
)

args = parser.parse_args()

train_data = pd.read_csv(args.train_df)
test_data = pd.read_csv(args.test_df)
external_data_1 = pd.read_csv(args.external_data_1)
external_data_2 = pd.read_csv(args.external_data_2)

train_data = pd.concat([train_data, external_data_1, external_data_2]).reset_index(drop=True)

train_data = create_folds(train_data, args.nfolds, args.seed)

train_data['answers'] = train_data[['answer_start', 'answer_text']].apply(convert_answers, axis=1)

accelerator = Accelerator()
print(f"{accelerator.device} is used")

x_train, x_valid = train_data.query(f"Fold != {args.nfolds}"), train_data.query(f"Fold == {args.nfolds}")

model = Model(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
pad_on_right = tokenizer.padding_side == 'right'

train_dataset = Dataset.from_pandas(x_train)
train_features = train_dataset.map(
    partial(
        prepare_train_features,
        tokenizer=tokenizer,
        pad_on_right=pad_on_right,
        max_length=args.max_length,
        doc_stride=args.doc_stride
    ),
    batched=True,
    remove_columns=train_dataset.column_names)

train_ds = ChaiiDataset(train_features)
train_dl = DataLoader(train_ds,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=True,
                      pin_memory=True,
                      drop_last=True)

valid_dataset = Dataset.from_pandas(x_valid)
valid_features = valid_dataset.map(
    partial(
        prepare_train_features,
        tokenizer=tokenizer,
        pad_on_right=pad_on_right,
        max_length=args.max_length,
        doc_stride=args.doc_stride
    ),
    batched=True,
    remove_columns=train_dataset.column_names)

valid_ds = ChaiiDataset(valid_features)
valid_dl = DataLoader(valid_ds,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=False,
                      pin_memory=True,
                      drop_last=False)

optimizer = make_optimizer(args, model)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=args.epochs * len(train_dl))

model, train_dl, valid_dl, optimizer, lr_scheduler = accelerator.prepare(model, train_dl, valid_dl, optimizer,
                                                                         lr_scheduler)

for fold in range(args.nfolds):
    print(f'Fold: {fold}')
    best_loss = 9999
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train(train_dl, model, optimizer)
        valid_loss = evaluate(model, valid_dl)

        if valid_loss <= best_loss:
            print(f"Epoch:{epoch} |Train Loss:{train_loss}|Valid Loss:{valid_loss}")
            print(f"Loss Decreased from {best_loss} to {valid_loss}")

            best_loss = valid_loss
            torch.save(model.state_dict(), f'./model{fold}/model{fold}.bin')
            tokenizer.save_pretrained(f'./model{fold}')

        end_time = time.time()
        print(f"Time taken by epoch {epoch} is {end_time - start_time:.2f}s")
        start_time = end_time