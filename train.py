import os
import json
import sys
import yaml
import torch
import wandb
import pandas as pd
from utils.parse_import import run_imports
from utils.load_embeddings import load_pretrained_embeddings
from utils.schedulers import get_inverse_sqrt_scheduler, exponential_ramp_up_scheduler
from utils.augmentations import get_augmentation_transforms
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
from datasets.collate_functions import pad_captions


def run_training(config):
    EXPERIMENT_NAME = config.get("EXPERIMENT_NAME")
    EXPERIMENT_PATH = f"saved_models/{EXPERIMENT_NAME}"
    if not os.path.exists(EXPERIMENT_PATH):
        os.makedirs(EXPERIMENT_PATH)

    if config.get("LOG_WANDB", False):
        wandb.login()
        wandb_run = wandb.init(project="image_captioning", name=EXPERIMENT_NAME, config=config)
    else:
        wandb_run = None

    training_classes = run_imports(config)
    DatasetClass = training_classes["DatasetClass"]
    TokenizerClass = training_classes["TokenizerClass"]
    EncoderClass = training_classes["EncoderClass"]
    DecoderClass = training_classes["DecoderClass"]
    ModelClass = training_classes["ModelClass"]

    training_config = config.get("TRAINING", {})

    EPOCHS = training_config.get("EPOCHS", 1)
    BATCH_SIZE = training_config.get("BATCH_SIZE", 32)
    LR = training_config.get("LR", 1e-3)
    WARMUP_STEPS = training_config.get("WARMUP_STEPS", 200)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = training_config.get("NUM_WORKERS", 1)

    aug_config = training_config.get("AUGMENTATION", {})
    ENABLE_AUGMENTATION = aug_config.get("FLAG", False)
    AUGMENTATION_START_EPOCH = aug_config.get("START_EPOCH", {})

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
    if not (wandb_run is None):
        wandb_run.watch(model, log="all", log_freq=1000)

    print(f"Total encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}")
    print(f"Total decoder parameters: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = get_inverse_sqrt_scheduler(optimizer, warmup_steps=WARMUP_STEPS)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"Running Epoch {epoch+1}/{EPOCHS}")

        aug_level = exponential_ramp_up_scheduler(epoch+1, AUGMENTATION_START_EPOCH, EPOCHS, curvature=1)
        aug_transforms = get_augmentation_transforms(aug_level)

        train_dataset = DatasetClass(
            tokenizer=tokenizer, img_transform=encoder.transforms,
            wandb_run=wandb_run, augmentations=aug_transforms,
            **config["DATASET"]["TRAIN"].get("PARAMS", {})
        )

        val_dataset = DatasetClass(
            tokenizer=tokenizer, img_transform=encoder.transforms,
            wandb_run=wandb_run,
            **config["DATASET"]["VAL"].get("PARAMS", {})
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, collate_fn=lambda x: pad_captions(x, pad_token_id),
            prefetch_factor=4, pin_memory=True, shuffle=(True if not isinstance(train_dataset, IterableDataset) else None)
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, collate_fn=lambda x: pad_captions(x, pad_token_id),
            prefetch_factor=4, pin_memory=True
        )

        train_loss = model.train_model(train_dataloader, loss_fn, optimizer, scheduler, DEVICE, wandb_run)
        val_loss = model.eval_model(val_dataloader, loss_fn, DEVICE, wandb_run)
        model.log_sample_captions(train_dataloader, train_dataset, tokenizer, DEVICE, wandb_run, epoch, num_samples=10)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{EXPERIMENT_PATH}/best_model.pth")
            print("Best model saved.")

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
    run_training(config)
    return


if __name__ == "__main__":
    main()