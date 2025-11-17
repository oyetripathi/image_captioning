import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, annot_df_path, img_dir, tokenizer, img_transform=None, y_transform=None, wandb_run=None):
        self.annot_df = pd.read_csv(annot_df_path)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform
    
    def __len__(self):
        return self.annot_df.shape[0]
    
    def __getitem__(self, idx):
        row_id = self.annot_df.index[idx]
        img_filename = self.annot_df.iloc[idx, 0]
        img_path = f"{self.img_dir}/{img_filename}"
        image = Image.open(img_path).convert("RGB")
        
        label = self.annot_df.iloc[idx, 1]
        token_ids, attention_mask = self.tokenizer.encode(label)
        if self.img_transform:
            image = self.img_transform(image)

        return {    
            "id": row_id,
            "image": image,
            "caption_text": label,
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }