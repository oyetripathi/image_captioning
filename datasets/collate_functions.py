import torch
from torch.nn.utils.rnn import pad_sequence

def pad_captions(batch, pad_token_id):
    images = [x["image"] for x in batch]
    captions = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        "images": images,
        "captions": captions,
        "masks": masks
    }
