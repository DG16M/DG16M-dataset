import os
import time
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import wandb

from se3dif.utils import makedirs, dict_to_device
from loguru import logger
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level="INFO"
)

class LossTracker:
    def __init__(self, use_wandb=False, wandb_logger=None, prefix=""):
        self.use_wandb = use_wandb
        self.logger = wandb_logger
        self.prefix = prefix   # use "" for train, "val/" for validation
        self.reset()

    def update(self, losses):
        for k, v in losses.items():
            if torch.is_tensor(v):
                v = v.detach().item()
            self.values[k].append(v)

    def mean(self, key):
        vals = self.values.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def summary(self):
        return {k: self.mean(k) for k in self.values}

    def log_to_wandb(self):
        if not self.use_wandb or self.logger is None:
            return
        summary = self.summary()
        summary_prefixed = {self.prefix + k: v for k, v in summary.items()}
        self.logger.log(summary_prefixed)

    def reset(self):
        from collections import defaultdict
        self.values = defaultdict(list)


def train(model, train_dataloader, epochs, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
        val_dataloader=None, clip_grad=False, val_loss_fn=None, optimizers=None, rank=0, device='cpu',
        args=None, logger_wandb=None):

    use_wandb = args.get("use_wandb", False)

    val_epoch_interval = 10
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizers[0], milestones=[40, 100, 180, 300], gamma=0.75
    ) if optimizers else None

    if val_dataloader:
        assert val_loss_fn is not None

    makedirs(model_dir)

    if rank == 0:
        summaries_dir = os.path.join(model_dir, 'summaries')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        makedirs(summaries_dir)
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(os.path.join(summaries_dir, exp_name))

    total_steps = -1
    val_steps = -1
    model.train()


    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(epochs):

            if (epoch + 1) % epochs_til_checkpoint == 0 and epoch and rank == 0:
                ckpt_path = os.path.join(
                    checkpoints_dir,
                    f'model_epoch_{epoch:04d}_iter_{total_steps:06d}.pth'
                )
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

            logger.info(f"Running epoch {epoch}")

            tracker = LossTracker(
                use_wandb=use_wandb,
                wandb_logger=logger_wandb,
                prefix="train/"
            )

            for step, (model_input, gt) in enumerate(train_dataloader):

                for optim in optimizers:
                    optim.zero_grad()

                total_steps += 1
                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)

                losses, iter_info = loss_fn(model, model_input, gt)

                total_loss = 0.0
                for name, loss in losses.items():
                    if "Accuracy" in name:
                        tracker.update({name: loss})
                        continue

                    loss_val = loss.mean()
                    total_loss += loss_val
                    tracker.update({name: loss_val.item()})

                    if rank == 0:
                        writer.add_scalar(name, loss_val, total_steps)

                if rank == 0:
                    writer.add_scalar("total_train_loss", total_loss, total_steps)

                if (total_steps + 1) % steps_til_summary == 0 and rank == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoints_dir, 'model_current.pth')
                    )


                    summary_dict = tracker.summary()
                    logger.info(
                        f"Epoch {epoch}, Step {total_steps} | " +
                        ", ".join([f"{k}: {v:.6f}" for k, v in summary_dict.items()])
                    )

                    tracker.log_to_wandb()
                    tracker.reset()

                total_loss.backward()

                if clip_grad:
                    norm = 1. if isinstance(clip_grad, bool) else clip_grad
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=norm)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

            # ---------------- Validation ----------------
            if (epoch + 1) % val_epoch_interval == 0:
                if val_dataloader is None:
                    logger.warning("No validation set passed, skipping")
                else:
                    logger.info("Running validation set...")
                    model.eval()

                    val_tracker = LossTracker(
                        use_wandb=use_wandb,
                        wandb_logger=logger_wandb,
                        prefix="val/"
                    )

                    for val_step, (model_input, gt) in enumerate(val_dataloader):
                        val_steps += 1
                        model_input = dict_to_device(model_input, device)
                        gt = dict_to_device(gt, device)

                        with torch.no_grad():
                            losses, _ = loss_fn(model, model_input, gt)

                        for name, loss in losses.items():
                            if "Accuracy" in name:
                                val_tracker.update({name: loss})
                                continue
                            val_tracker.update({name: loss.mean().item()})

                        if (val_step + 1) % 20 == 0:
                            summary_dict = val_tracker.summary()
                            logger.info(
                                f"Epoch {epoch}, Val Step {val_step} | " +
                                ", ".join([f"{k}: {v:.6f}" for k, v in summary_dict.items()])
                            )

                            val_tracker.log_to_wandb()
                            val_tracker.reset()

                    logger.info("Validation set finished")
                    model.train()

            if lr_scheduler:
                lr_scheduler.step()

            logger.info(f"Epoch {epoch} finished")
