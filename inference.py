import torch
import pandas as pd
from datasets.caption_dataset import FlickrDataset
from datasets.collate_functions import pad_captions
from importlib import import_module

EXPERIMENT_NAME = "test"
MODEL_SAVEPATH = f"saved_models/{EXPERIMENT_NAME}"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(f"{MODEL_SAVEPATH}/config.json", "r") as f:
    config = json.load(f)

df = pd.read_csv("data_files/flickr8k/captions_split.txt")

import_module(config["encoder"]["module"])
import_module(config["decoder"]["module"])
import_module(config["model"]["module"])

tokenizer_class = getattr(import_module(config["tokenizer"]["module"]), config["tokenizer"]["class"])
tokenizer = tokenizer_class()
tokenizer.load(f"{MODEL_SAVEPATH}/tokenizer.json")
pad_token_id = tokenizer.vocab.get("<pad>")

im_transforms = transforms.Compose([
    transforms.ToTensor()
])
dataset = FlickrDataset(df, img_dir="data_files/flickr8k/Images", tokenizer=tokenizer, img_transform=None)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_captions(x, pad_token_id))

encoder_class = getattr(import_module(config["encoder"]["module"]), config["encoder"]["class"])
encoder = encoder_class()
decoder_class = getattr(import_module(config["decoder"]["module"]), config["decoder"]["class"])
decoder = decoder_class(**config["decoder"]["params"])

model_class = getattr(import_module(config["model"]["module"]), config["model"]["class"])
model = model_class(encoder, decoder)

model.load_state_dict(torch.load(f"{MODEL_SAVEPATH}/model.pth"))

captions = model.generate_caption(dataloader, tokenizer, DEVICE, max_len=20, beam_size=None)

# need an inference dataset class, just like FlickrDataset but without captions