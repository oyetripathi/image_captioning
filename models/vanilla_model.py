import torch
import torch.nn as nn
from tqdm import tqdm


class VanillaCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VanillaCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        img_encoding = self.encoder(images)
        outputs = self.decoder(captions, img_encoding)
        return outputs
    
    def train_model(self, train_loder, loss_fn, optimizer, device):
        loss_list = []
        num_batch = len(train_loder)
        self.to(device)
        self.train()
        for batch_idx, batch in enumerate(tqdm(train_loder)):
            optimizer.zero_grad()
            images, captions = batch["images"].to(device), batch["captions"].to(device)
            logits = self(images, captions[:, :-1])
            iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
            iter_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                loss_list.append(iter_loss.item())
        return loss_list
    
    def eval_model(self, val_loader, loss_fn, device):
        total_loss = 0
        num_batch = len(val_loader)
        self.to(device)
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                images, captions = batch["images"].to(device), batch["captions"].to(device)
                logits = self(images, captions[:, :-1])
                iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
                total_loss += iter_loss.item()
        return total_loss / num_batch

    def generate_caption(self, dataloader, tokenizer, device, max_len=20, beam_size=None):
        self.to(device)
        self.eval()
        captions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                images = batch["images"].to(device)
                img_encodings = self.encoder(images)
                inputs = torch.tensor([[tokenizer.vocab.get("<start>")]]*images.shape[0]).to(device)
                for _ in range(max_len):
                    logits = self.decoder(inputs, img_encodings)
                    next_token_ids = logits.argmax(dim=-1)[:, -1].unsqueeze(-1)
                    inputs = torch.cat([inputs, next_token_ids], dim=-1)
                captions.extend([tokenizer.decode(ids) for ids in inputs.cpu().numpy()])
        return captions

