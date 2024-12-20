import argparse
import os

import torch

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchmetrics.text import CharErrorRate
from tqdm import tqdm

from dataset import ICVLPDataset
from decoder import GreedyCTCDecoder
from metrics import LetterNumberRecognitionRate
from model import LPRNet, SpatialTransformerLayer, LocNet
from utils import CHARS_DICT, LABELS_DICT, TColor, pad_target_sequence


class Trainer:
    def __init__(self):
        self.args = None
        self.logger = None

        self.device = None

        self.ds_train = None
        self.dl_train = None
        self.ds_val = None
        self.dl_val = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None

        self.decoder = None
        self.rln = None
        self.cer = None

        self.lr_scheduler = None

        self.epoch = 0

        self.avg_loss = 0.0
        self.avg_acc = 0.0
        self.avg_rln = 0.0
        self.avg_cer = 0.0
        self.val_avg_loss = 0.0
        self.val_avg_acc = 0.0
        self.val_avg_rln = 0.0
        self.val_avg_cer = 0.0

        self.parse_args()
        self.resolve_device()
        self.log_args()

        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_loss()
        self.init_metrics()

        self.init_logger()

        self.run()

        self.cleanup()

    @staticmethod
    def log(message, level="info"):
        if level == "info":
            message = f"{TColor.OKBLUE}info{TColor.ENDC}: {message}"
        elif level == "error":
            message = f"{TColor.FAIL}error{TColor.ENDC}: {message}"
        print(f"{message}")

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Train ICVLPR model on Indonesian Commercial Vehicle License Plate dataset"
        )

        # Training Setting
        parser.add_argument(
            "--device", type=str, default=None, help="Device to use for training"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.001,
            help="Initial learning rate. Default: 0.001",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for training in each iteration. Default: 32",
        )
        parser.add_argument(
            "--epoch-start", type=int, default=0, help="Start epoch number"
        )
        parser.add_argument(
            "--epoch-end", type=int, default=1500, help="End epoch number"
        )
        parser.add_argument(
            "--learning-rate-scheduler-step",
            type=int,
            default=700,
            help="Learning rate scheduler step",
        )
        parser.add_argument(
            "--save-last",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save last checkpoint of the run",
        )
        parser.add_argument(
            "--concat-dataset",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Concatenate dataset for final training",
        )

        # Checkpoint
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="If set, will restore the model from the checkpoint path and continue training",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default="checkpoints",
            help="Directory to save checkpoints",
        )
        parser.add_argument(
            "--checkpoint-save-interval",
            type=int,
            default=100,
            help="Save checkpoint interval",
        )
        parser.add_argument(
            "--checkpoint-prefix",
            type=str,
            default="epoch",
            help="Checkpoint prefix when saving",
        )

        # Logging wandb
        parser.add_argument(
            "--wandb",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable wandb logging",
        )
        parser.add_argument("--run-id", type=str, default=None, help="Run ID for wandb")

        # Spatial Transformer Network
        parser.add_argument(
            "--stn-enable-at", type=int, default=300, help="Enable STN at epoch"
        )

        self.args = parser.parse_args()

    def resolve_device(self):
        if self.args.device is None:
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(self.args.device)

    def log_args(self):
        print("-" * 20)
        for key, value in vars(self.args).items():
            print(f"{TColor.GREEN}{key:<35}: {TColor.BOLD}{value}{TColor.ENDC}")
        print("-" * 20)

    def init_dataset(self):
        self.log("Initializing dataset...")

        img_transforms = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.07, 0.05),
                    scale=(0.7, 1),
                    shear=(-10, 10),
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.3,
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Resize((24, 94)),
            ]
        )

        self.ds_train = ICVLPDataset(
            "data",
            subset="train",
            transform=img_transforms,
            download=True,
        )
        self.ds_val = ICVLPDataset(
            "data",
            subset="val",
            transform=img_transforms,
        )

        if self.args.concat_dataset:
            self.ds_train = ConcatDataset([self.ds_train, self.ds_val])
            self.ds_val = ICVLPDataset(
                "data",
                subset="test",
                transform=img_transforms,
            )

        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=pad_target_sequence,
        )
        self.dl_val = DataLoader(
            self.ds_val,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=pad_target_sequence,
        )
        self.log(f"Train Dataset Length: {len(self.ds_train)}")
        self.log(f"Val Dataset Length: {len(self.ds_val)}")
        self.log(f"Datasets initialized: {self.ds_train.__class__.__name__}")

    def init_model(self):
        self.log("Initializing model...")

        loc = LocNet()
        stn = SpatialTransformerLayer(localization=loc, align_corners=True)

        num_classes = len(CHARS_DICT)

        self.model = LPRNet(num_classes=num_classes, stn=stn).to(self.device)
        self.model.use_stn(False)

        if self.args.checkpoint:
            self.log(f"Restoring model from checkpoint: {self.args.checkpoint}")
            self.log(
                self.model.load_state_dict(
                    torch.load(self.args.checkpoint, map_location=self.device)
                )
            )

        self.log("Model initialized.")

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.learning_rate_scheduler_step,
            gamma=0.1,
        )
        self.log(f"Optimizer initialized: {self.optimizer.__class__.__name__}")

    def init_loss(self):
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=False, reduction="mean")
        self.log(f"Loss function initialized: {self.loss_fn.__class__.__name__}")

    def init_metrics(self):
        self.decoder = GreedyCTCDecoder(blank=0)
        self.rln = LetterNumberRecognitionRate(blank=0)
        self.cer = CharErrorRate()
        self.log("Metrics initialized.")

    def init_logger(self):
        if not self.args.wandb:
            return

        import wandb

        config = {
            "learning-rate": self.args.learning_rate,
            "batch-size": self.args.batch_size,
            "epoch-start": self.args.epoch_start,
            "epoch-end": self.args.epoch_end,
            "epoch": abs(self.args.epoch_end - self.args.epoch_start),
            "dataset": self.ds_train.__class__.__name__,
            "dataset-concatenated": self.args.concat_dataset,
            "loss": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
        }

        self.logger = wandb.init(
            project="icvlpr",
            config=config,
            id=self.args.run_id,
            resume="allow",
        )

    def log_epoch(self):
        if self.logger:
            self.logger.log(
                {
                    "loss/train": self.avg_loss,
                    "loss/val": self.val_avg_loss,
                    "acc/train": self.avg_acc,
                    "acc/val": self.val_avg_acc,
                    "rln/train": self.avg_rln,
                    "rln/val": self.val_avg_rln,
                    "cer/train": self.avg_cer,
                    "cer/val": self.val_avg_cer,
                },
                step=self.epoch,
            )

    def calculate_loss(self, logits, targets):
        target_lengths = targets.ne(0).sum(dim=1)
        input_lengths = torch.full(
            size=(logits.size(0),), fill_value=logits.size(2), dtype=torch.long
        )
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        return self.loss_fn(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

    def activate_stn(self):
        if not self.model.using_stn and self.epoch > self.args.stn_enable_at:
            self.model.use_stn(True)
            self.log("Spatial Transformer Network activated.")

    def save_model(self):
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir, f"{self.args.checkpoint_prefix}_{self.epoch}.pth"
        )
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        self.log(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def save(self):
        if self.epoch % self.args.checkpoint_save_interval == 0:
            checkpoint_path = self.save_model()
            self.log_model(checkpoint_path)

    def log_model(self, path):
        if self.logger:
            self.logger.log_model(path=path, name=os.path.basename(path))

    def log_lr_scheduler(self):
        if self.epoch % self.args.learning_rate_scheduler_step == 0:
            self.log(f"Learning rate updated to {self.lr_scheduler.get_last_lr()}")

    def cleanup(self):
        if (
            self.args.save_last
            and self.args.epoch_end == self.epoch
            and self.epoch % self.args.checkpoint_save_interval != 0
        ):
            checkpoint_path = self.save_model()
            self.log_model(checkpoint_path)

        if self.logger:
            self.logger.finish()

    def _compare_texts(self, preds, labels):
        acc = 0
        for pred, label in zip(preds, labels):
            pred = "".join([LABELS_DICT[p] for p in pred])
            label = "".join([LABELS_DICT[i] for i in label.numpy()])
            acc += int(pred == label)
        return acc / len(preds)

    def run(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch + 1

            # Training
            self.activate_stn()
            self.model.train()
            self.train()

            # Validation
            self.model.eval()
            self.eval()

            # Log and save
            self.log_epoch()
            self.save()

            # Learning Rate Scheduler Step
            self.lr_scheduler.step()
            self.log_lr_scheduler()

    def train(self):
        running_loss = 0.0
        running_acc = 0.0
        running_rln = 0.0
        running_cer = 0.0

        for i, (images, targets, labels) in enumerate(
            pbar := tqdm(
                self.dl_train,
                desc=f"Epoch {self.epoch}/{self.args.epoch_end}",
                unit="step",
                leave=False,
                position=0,
            )
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.calculate_loss(logits, targets)
            loss.backward()
            self.optimizer.step()

            preds = self.decoder(logits)

            running_loss += loss.item()
            running_acc += self._compare_texts(preds, labels)
            running_rln += self.rln(preds, targets)
            running_cer += self.cer(preds, targets)

            pbar.set_postfix(
                loss=f"{running_loss / (i + 1):.4f}",
                acc=f"{running_acc / (i + 1):.4f}",
                rln=f"{running_rln / (i + 1):.4f}",
                cer=f"{running_cer/(i + 1):.4f}",
            )

        self.avg_loss = running_loss / len(self.dl_train)
        self.avg_acc = running_acc / len(self.dl_train)
        self.avg_rln = running_rln / len(self.dl_train)
        self.avg_cer = running_cer / len(self.dl_train)

    @torch.inference_mode()
    def eval(self):
        running_loss = 0.0
        running_acc = 0.0
        running_rln = 0.0
        running_cer = 0.0

        for i, (images, targets, labels) in enumerate(
            pbar := tqdm(
                self.dl_val,
                desc=f"Epoch {self.epoch}/{self.args.epoch_end}",
                unit="step",
                leave=True,
                position=0,
            )
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(images)
            loss = self.calculate_loss(logits, targets)

            preds = self.decoder(logits)

            running_loss += loss.item()
            running_acc += self._compare_texts(preds, labels)
            running_rln += self.rln(preds, targets)
            running_cer += self.cer(preds, targets)

            if self.args.concat_dataset:
                pbar.set_postfix(
                    loss=f"{self.avg_loss:.4f}",
                    acc=f"{self.avg_acc:.4f}",
                    rln=f"{self.avg_rln:.4f}",
                    cer=f"{self.avg_cer:.4f}",
                    test_loss=f"{running_loss / (i + 1):.4f}",
                    test_acc=f"{running_acc / (i + 1):.4f}",
                    test_rln=f"{running_rln / (i + 1):.4f}",
                    test_cer=f"{running_cer / (i + 1):.4f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{self.avg_loss:.4f}",
                    acc=f"{self.avg_acc:.4f}",
                    rln=f"{self.avg_rln:.4f}",
                    cer=f"{self.avg_cer:.4f}",
                    val_loss=f"{running_loss / (i + 1):.4f}",
                    val_acc=f"{running_acc / (i + 1):.4f}",
                    val_rln=f"{running_rln / (i + 1):.4f}",
                    val_cer=f"{running_cer / (i + 1):.4f}",
                )

        self.val_avg_loss = running_loss / len(self.dl_val)
        self.val_avg_acc = running_acc / len(self.dl_val)
        self.val_avg_rln = running_rln / len(self.dl_val)
        self.val_avg_cer = running_cer / len(self.dl_val)


if __name__ == "__main__":
    Trainer()
