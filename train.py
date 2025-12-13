import os
import json
import sys
import yaml
import torch
import wandb
import pandas as pd
from torch.nn.modules.batchnorm import SyncBatchNorm 
from utils.parse_import import run_imports
from utils.load_embeddings import load_pretrained_embeddings
from utils.schedulers import get_inverse_sqrt_scheduler, exponential_ramp_up_scheduler
from utils.augmentations import get_augmentation_transforms
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
from datasets.collate_functions import pad_captions
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_setup():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not "MASTER_ADDR" in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend = "nccl" if torch.cuda.is_available() else "gloo",
        rank = rank,
        world_size = world_size,
        device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
    )

    dist.barrier()
    return rank, local_rank, world_size


def run_training(config, rank, local_rank, world_size):
    EXPERIMENT_NAME = config.get("EXPERIMENT_NAME")
    EXPERIMENT_PATH = f"saved_models/{EXPERIMENT_NAME}"
    if not os.path.exists(EXPERIMENT_PATH) and rank==0:
        os.makedirs(EXPERIMENT_PATH)

    if not config.get("LOG_WANDB", False) or rank!=0:
        wandb_run = None
    else:
        wandb.login()
        wandb_run = wandb.init(project="image_captioning", name=EXPERIMENT_NAME, config=config)

    training_classes = run_imports(config)
    DatasetClass = training_classes["DatasetClass"]
    TokenizerClass = training_classes["TokenizerClass"]
    EncoderClass = training_classes["EncoderClass"]
    DecoderClass = training_classes["DecoderClass"]
    ModelClass = training_classes["ModelClass"]

    training_config = config.get("TRAINING", {})

    EPOCHS = training_config.get("EPOCHS", 1)
    BATCH_SIZE = training_config.get("BATCH_SIZE", 32)
    ENCODER_LR = training_config.get("ENCODER_LR", 1e-4)
    DECODER_LR = training_config.get("DECODER_LR", 1e-3)
    WARMUP_STEPS = training_config.get("WARMUP_STEPS", 200)
    DEVICE = (torch.device(f"cuda:{local_rank}")) if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = training_config.get("NUM_WORKERS", 1)

    aug_config = training_config.get("AUGMENTATION", {})
    ENABLE_AUGMENTATION = aug_config.get("FLAG", False)
    AUGMENTATION_START_EPOCH = aug_config.get("START_EPOCH", 0)

    CHECKPOINT_NAME = config.get("CHECKPOINT_NAME", None)

    print(f"Using device: {DEVICE}")

    tokenizer = TokenizerClass(**config["TOKENIZER"].get("PARAMS", {}))
    tokenizer.build_vocab(**config["TOKENIZER"].get("BUILD_VOCAB_PARAMS", {}))
    pad_token_id = tokenizer.pad_token_id

    embeddings = None
    if config["DECODER"].get("PRETRAINED_EMBEDDINGS_PATH", None) is not None:
        embeddings = load_pretrained_embeddings(
            config["DECODER"]["PRETRAINED_EMBEDDINGS_PATH"],
            tokenizer.vocab,
            config["DECODER"]["PARAMS"]["embed_dim"]
        )        

    encoder = EncoderClass(**config["ENCODER"].get("PARAMS", {}))
    decoder = DecoderClass(
        vocab_size=tokenizer.vocab_size,
        pretrained_embeddings=embeddings,
        **config["DECODER"].get("PARAMS", {})
    )

    model = ModelClass(encoder, decoder)
    if CHECKPOINT_NAME is not None:
        checkpoint_wts = torch.load(f"{EXPERIMENT_PATH}/checkpoints/{CHECKPOINT_NAME}", weights_only=True, map_location="cpu")
        checkpoint_wts = {k.replace("module.", ""): v for k, v in checkpoint_wts.items()}
        model.load_state_dict(checkpoint_wts)

    if not (wandb_run is None):
        wandb_run.watch(model, log="all", log_freq=1000)

    if rank == 0:
        print(f"Total encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}")
        print(f"Total decoder parameters: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}")

    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(DEVICE), device_ids=[local_rank] if torch.cuda.is_available() else None)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.module.decoder.parameters(), "lr": DECODER_LR}
        ],
        weight_decay=0.05
    )
    scheduler = get_inverse_sqrt_scheduler(optimizer, warmup_steps=WARMUP_STEPS)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        if rank == 0:
            print(f"Running Epoch {epoch+1}/{EPOCHS}")

        if ENABLE_AUGMENTATION:
            aug_level = exponential_ramp_up_scheduler(epoch+1, AUGMENTATION_START_EPOCH, EPOCHS, curvature=1)
            aug_transforms = get_augmentation_transforms(aug_level)
        else:
            aug_transforms = None

        train_dataset = DatasetClass(
            tokenizer=tokenizer, img_transform=encoder.transforms,
            wandb_run=wandb_run, augmentations=aug_transforms,
            rank=rank, world_size=world_size,
            **config["DATASET"]["TRAIN"].get("PARAMS", {})
        )

        val_dataset = DatasetClass(
            tokenizer=tokenizer, img_transform=encoder.transforms,
            rank=rank, world_size=world_size,
            **config["DATASET"]["VAL"].get("PARAMS", {})
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, collate_fn=lambda x: pad_captions(x, pad_token_id),
            prefetch_factor=2, pin_memory=False, shuffle=(True if not isinstance(train_dataset, IterableDataset) else None)
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, collate_fn=lambda x: pad_captions(x, pad_token_id),
            prefetch_factor=2, pin_memory=False
        )

        train_loss = model.module.train_model(train_dataloader, loss_fn, optimizer, scheduler, DEVICE, wandb_run, end_token_id=tokenizer.vocab["<end>"], min_len=15)
        val_loss = model.module.eval_model(val_dataloader, loss_fn, DEVICE, wandb_run)
        model.module.log_sample_captions(train_dataloader, train_dataset, tokenizer, DEVICE, wandb_run, epoch, num_samples=10)
        if rank == 0:
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{EXPERIMENT_PATH}/best_model.pth")
                print("Best model saved.")
    if rank == 0:
        torch.save(model.state_dict(), f"{EXPERIMENT_PATH}/model.pth")
        tokenizer.save(f"{EXPERIMENT_PATH}/tokenizer.json")
        if not (wandb_run is None):
            wandb.finish()
        print("Model saved successfully.")
    return


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python train.py <config.yaml>")
        return
    config_path = args[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    rank, local_rank, world_size = ddp_setup()
    print(rank, local_rank, world_size)
    run_training(config,rank, local_rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()
    return


if __name__ == "__main__":
    main()