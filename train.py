import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ICVLPDataset
from model import ICVLPR
from train_loss import ctc_loss


class Trainer:
    def __init__(self):
        self.args = None
        self.ds_train = None
        self.dl_train = None
        self.ds_val = None
        self.dl_val = None
        self.model = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.parse_args()
        self.init_dataset()
        self.init_model()
        self.train()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Training Setting
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=250000)
        parser.add_argument('--learning-rate-scheduler-step', type=int, default=100000)
        # Checkpoint
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
        parser.add_argument('--checkpoint-save-interval', type=int, default=1000)

        self.args = parser.parse_args()
        self.log_args()

    def log_args(self):
        self.log('-' * 20)
        for key, value in vars(self.args).items():
            self.log(f'{key:<25}: {value}')
        self.log('-' * 20)

    def init_dataset(self):
        self.log('Initializing dataset...')

        def collate_fn(batch):
            """Collate function for the dataloader.

            Automatically adds padding to the target of each batch.
            """
            # Extract samples and targets from the batch
            samples, targets = zip(*batch)

            # Pad the target sequences to the same length
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

            # Return padded samples and targets
            return torch.stack(samples), padded_targets

        self.ds_train = ICVLPDataset('data', subset='train')
        self.dl_train = DataLoader(self.ds_train, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)
        self.log(f'Train Dataset Length: {len(self.ds_train)}')

        self.ds_val = ICVLPDataset('data', subset='val')
        self.dl_val = DataLoader(self.ds_val, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn)
        self.log('Datasets initialized.')

    def init_model(self):
        self.log('Initializing model...')
        self.model = ICVLPR()
        self.model.to(self.device)

        if self.args.checkpoint:
            self.log(f'Restoring model from {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=self.device)
            ))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.args.learning_rate_scheduler_step,
                                                            gamma=0.1)
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=False, reduction="mean")
        self.log('Model initialized.')

    def train(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch + 1
            self.step = self.epoch * len(self.dl_train)

            self.val_avg_loss = 0.0
            self.validate()

            self.train_loss = 0.0

            for batch in (pbar := tqdm(self.dl_train,
                                       desc=f'Epoch {self.epoch}',
                                       unit='step')):
                data, targets = batch

                self.train_model(data, targets)

                pbar.set_postfix(loss=self.train_loss / len(self.dl_train), val_loss=self.val_avg_loss)

            self.lr_scheduler.step()

            if self.epoch % self.args.checkpoint_save_interval == 0:
                self.save()

    def train_model(self, data, targets):
        data = data.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = ctc_loss(self.loss_fn, logits, targets)
        loss.backward()
        self.optimizer.step()
        self.train_loss += loss.item()

    def validate(self):
        total_loss, total_count = 0, 0
        self.model.eval()
        with torch.no_grad():
            for batch in (pbar := tqdm(self.dl_val,
                                       desc=f'Validation {self.epoch}',
                                       leave=False)):
                data, targets = batch
                data, targets = data.to(self.device), targets.to(self.device)
                logits = self.model(data)
                loss = ctc_loss(self.loss_fn, logits, targets)
                total_loss += loss.item()
                total_count += logits.size(0)
                self.val_avg_loss = total_loss / total_count
                pbar.set_postfix(val_loss=self.val_avg_loss)
        self.model.train()

    def save(self):
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'epoch_{self.epoch}.pth')
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        self.log(f'Checkpoint saved to {checkpoint_path}')

    @staticmethod
    def log(message):
        print(f'{message}')


if __name__ == '__main__':
    Trainer()
