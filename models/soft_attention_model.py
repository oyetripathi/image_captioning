import json
import torch
import torch.nn as nn
from tqdm import tqdm


class CaptioningModelWithSoftAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(CaptioningModelWithSoftAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        img_encoding = self.encoder(images)
        outputs, attn_wts = self.decoder(captions, img_encoding)
        return outputs, attn_wts
    
    def train_model(self, train_loder, loss_fn, optimizer, device, wandb_run=None):
        total_loss = 0
        num_batch = len(train_loder)
        self.to(device)
        self.train()
        for batch_idx, batch in enumerate(tqdm(train_loder)):
            optimizer.zero_grad()
            images, captions = batch["images"].to(device), batch["captions"].to(device)
            logits, _ = self(images, captions[:, :-1])
            iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
            iter_loss.backward()
            optimizer.step()
            total_loss += iter_loss.item()
            if wandb_run:
                wandb_run.log({"Batch Loss": iter_loss.item()})
        return total_loss / num_batch

    def eval_model(self, val_loader, loss_fn, device, wandb_run=None):
        total_loss = 0
        num_batch = len(val_loader)
        self.to(device)
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                images, captions = batch["images"].to(device), batch["captions"].to(device)
                logits, _ = self(images, captions[:, :-1])
                iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
                total_loss += iter_loss.item()
                if wandb_run:
                    wandb_run.log({"Validation Batch Loss": iter_loss.item()})
        return total_loss / num_batch
    
    @staticmethod
    def decode_caption(token_ids, attn_wts, tokenizer):
        eos_token = tokenizer.vocab.get("<end>")
        if eos_token in token_ids:
            end_idx = token_ids.index(eos_token)
            return tokenizer.decode(token_ids[:end_idx]), attn_wts[:end_idx, :]
        else:
            return tokenizer.decode(token_ids), attn_wts

    def generate_caption(self, dataloader, tokenizer, device, max_len=20, beam_size=None):
        self.to(device)
        self.eval()
        all_row_ids = []
        all_captions = []
        all_attn_wts = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                row_ids = batch["id"]
                batch_size = row_ids.shape[0]

                images = batch["images"].to(device)
                img_encodings = self.encoder(images)
                img_encodings = img_encodings.view(batch_size, self.decoder.feature_dim, -1).permute((0, 2, 1))

                inputs = torch.tensor([[tokenizer.vocab.get("<start>")]]*batch_size).to(device)
                seq_attn_wts = torch.zeros((batch_size, max_len, img_encodings.shape[1])).to(device)
                h,c = self.decoder.init_hidden_state(batch_size, self.decoder.hidden_dim, device)

                for t in range(max_len):
                    logits, h, c, attn_wts = self.decoder.decode_step(img_encodings, inputs[:, -1], h, c)
                    next_token_ids = logits.argmax(dim=-1).unsqueeze(-1)
                    inputs = torch.cat([inputs, next_token_ids], dim=-1)
                    seq_attn_wts[:, t, :] = attn_wts
                outputs = zip(inputs.cpu().numpy(), seq_attn_wts.cpu().numpy())

                for ids, attn_wts in outputs:
                    caption, attn_wt = self.decode_caption(ids.tolist(), attn_wts, tokenizer)
                    all_captions.append(caption)
                    all_attn_wts.append(json.dumps(attn_wt.tolist()))
                
                all_row_ids.extend(row_ids.cpu().numpy().tolist())
        return {
            "id": all_row_ids,
            "generated_caption": all_captions,
            "attention_weights": all_attn_wts
        }

