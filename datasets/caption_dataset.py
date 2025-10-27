import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, annot_df, img_dir, tokenizer, img_transform=None, y_transform=None, img_resize_shape=(224, 224)):
        self.annot_df = annot_df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.img_resize_shape = img_resize_shape
    
    def __len__(self):
        return self.annot_df.shape[0]
    
    def __getitem__(self, idx):
        row_id = self.annot_df.index[idx]
        img_filename = self.annot_df.iloc[idx, 0]
        img_path = f"{self.img_dir}/{img_filename}"
        image = Image.open(img_path).convert("RGB").resize(self.img_resize_shape)
        
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