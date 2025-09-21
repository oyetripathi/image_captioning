import os
import json
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.collate_functions import pad_captions
from tokenizers.word_tokenizer import WordTokenizer
from datasets.caption_dataset import FlickrDataset
from models.encoders import ResNet101Encoder
from models.decoders import DecoderLSTM
from models.vanilla_model import VanillaCaptioningModel

EXPERIMENT_NAME = "test"
EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 512
MODEL_SAVEPATH = f"saved_models/{EXPERIMENT_NAME}"

df = pd.read_csv("data_files/flickr8k/captions_split.txt")

tokenizer = WordTokenizer()
tokenizer.build_vocab(df, save_path="data_files/flickr8k/vocab.json")
pad_token_id = tokenizer.vocab.get("<pad>")

im_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = FlickrDataset(df[df["type"]=="train"], img_dir="data_files/flickr8k/Images", tokenizer=tokenizer, img_transform=im_transforms)
val_dataset = FlickrDataset(df[df["type"]=="val"], img_dir="data_files/flickr8k/Images", tokenizer=tokenizer, img_transform=im_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: pad_captions(x, pad_token_id))
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_captions(x, pad_token_id))

encoder = ResNet101Encoder()
decoder = DecoderLSTM(feature_dim=2048, vocab_size=len(tokenizer.vocab), embed_dim=EMBED_DIM, pretrained_embeddings=None)

model = VanillaCaptioningModel(encoder, decoder)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    print(f"Running Epoch {epoch+1}/{EPOCHS}")
    train_loss = model.train_model(train_dataloader, loss_fn, optimizer, DEVICE)
    val_loss = model.eval_model(val_dataloader, loss_fn, DEVICE)
    print(f"Training Loss: {train_loss[-1]:.4f}, Validation Loss: {val_loss:.4f}")

if not os.path.exists(MODEL_SAVEPATH):
    os.makedirs(MODEL_SAVEPATH)

torch.save(model.state_dict(), f"{MODEL_SAVEPATH}/model.pth")
tokenizer.save(f"{MODEL_SAVEPATH}/tokenizer.json")

config = {
    "experiment_name": EXPERIMENT_NAME,
    "tokenizer": {
        "module": "tokenizers.word_tokenizer",
        "class": "WordTokenizer",
    },
    "encoder": {
        "module": "models.encoders",
        "class": "ResNet101Encoder",
        "params": {}
    },
    "decoders": {
        "module": "models.decoders",
        "class": "DecoderLSTM",
        "params": {
            "feature_dim": 2048,
            "vocab_size": len(tokenizer.vocab),
            "embed_dim": EMBED_DIM,
            "pretrained_embeddings": None
        }
    },
    "model": {
        "module": "models.vanilla_model",
        "class": "VanillaCaptioningModel",
    }
}
with open(f"{MODEL_SAVEPATH}/config.json", "w") as f:
    json.dump(config, f)

print("Model saved successfully.")
