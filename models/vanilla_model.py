import torch
import torch.nn as nn
from metrics.BLEU import compute_bleu
import wandb
from tqdm import tqdm


class VanillaCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VanillaCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, masks):
        img_encoding = self.encoder(images)
        outputs = self.decoder(captions, img_encoding, masks)
        return outputs
    
    def train_model(self, train_loder, loss_fn, optimizer, scheduler, device, wandb_run=None):
        total_loss = 0
        num_batch = 0
        self.to(device)
        self.train()
        for batch_idx, batch in enumerate(tqdm(train_loder)):
            optimizer.zero_grad()
            images, captions, masks = batch["images"].to(device), batch["captions"].to(device), batch["masks"].to(device)
            logits = self(images, captions[:, :-1], masks[:, :-1])
            iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
            iter_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += iter_loss.item()
            num_batch += 1
            if wandb_run:
                wandb_run.log({"Batch Loss": iter_loss.item()})
                wandb_run.log({"Learning Rate": optimizer.param_groups[0]["lr"]})
        return total_loss / num_batch
        
    def eval_model(self, val_loader, loss_fn, device, wandb_run=None):
        total_loss = 0
        num_batch = 0
        self.to(device)
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                images, captions, masks = batch["images"].to(device), batch["captions"].to(device), batch["masks"].to(device)
                logits = self(images, captions[:, :-1], masks[:, :-1])
                iter_loss = loss_fn(logits.view(-1, logits.shape[-1]), captions[:, 1:].reshape(-1))
                total_loss += iter_loss.item()
                num_batch += 1
                if wandb_run:
                    wandb_run.log({"Validation Batch Loss": iter_loss.item()})
        return total_loss / num_batch

    def log_sample_captions(self, dataloader, dataset, tokenizer, device, wandb_run, meta="", num_samples=5):
        if wandb_run is None:
            return
        results = self.generate_caption(dataloader, tokenizer, device, beam_size=3, num_samples=num_samples)
        images = dataset.get_images_from_list_id(results["id"])
        my_table = wandb.Table(columns=["image", "generated_caption", "true_caption"])
        for i in range(len(results["id"])):
            my_table.add_data(
                wandb.Image(images[i]),
                results["generated_caption"][i],
                results["true_caption"][i],
            )
        wandb_run.log({f"Sample Captions {meta}": my_table})
        return
    
    @staticmethod
    def decode_caption(token_ids, tokenizer):
        eos_token = tokenizer.vocab.get("<end>")
        if eos_token in token_ids:
            return tokenizer.decode(token_ids[:token_ids.index(eos_token)])
        else:
            return tokenizer.decode(token_ids)
    
    def _beam_search_batch(self, img_encodings, tokenizer, device, beam_size=3, max_len=20, alpha=0.6):
        batch_size = img_encodings.shape[0]
        start_token = tokenizer.vocab.get("<start>")
        end_token = tokenizer.vocab.get("<end>")
        vocab_size = tokenizer.vocab_size

        sequences = torch.full((batch_size, beam_size, 1), start_token, dtype=torch.long).to(device)
        scores = torch.zeros(batch_size, beam_size).to(device)

        img_encodings = (
            img_encodings.
            unsqueeze(1)
            .repeat(1, beam_size, 1, 1)
            .view(-1, img_encodings.shape[1], img_encodings.shape[2])
        )

        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool).to(device)

        for _ in range(max_len):
            input_ids = sequences.view(-1, sequences.shape[-1])
            logits = self.decoder(input_ids, img_encodings)
            next_token_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            new_scores = scores.unsqueeze(-1) + log_probs.view(batch_size, beam_size, vocab_size)
            new_scores = new_scores.view(batch_size, -1)

            topk_scores, topk_indices = torch.topk(new_scores, beam_size, dim=-1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            sequences = torch.cat(
                [sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])), token_indices.unsqueeze(-1)],
                dim=-1
            )
            scores = topk_scores
            finished = finished.gather(1, beam_indices) | (token_indices == end_token)
            if finished.all():
                break
        end_mask = (sequences == end_token).to(torch.int64)
        first_end = torch.where(
            end_mask.any(dim=-1), end_mask.argmax(dim=-1)+1, torch.tensor(max_len, device=device)
        )
        final_scores = scores / (first_end.float() ** alpha)
        best_beam_idx = final_scores.argmax(dim=1)
        best_sequences = sequences[torch.arange(batch_size), best_beam_idx, :].cpu().tolist()
        return best_sequences

    def generate_caption(self, dataloader, tokenizer, device, beam_size=1, num_samples=None): 
        self.to(device)
        self.eval()
        true_captions = []
        captions = []
        row_ids = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                images = batch["images"].to(device)
                img_encodings = self.encoder(images)
                best_ids_list = self._beam_search_batch(img_encodings, tokenizer, device, beam_size=beam_size, max_len=tokenizer.max_len)
                captions.extend([self.decode_caption(ids, tokenizer) for ids in best_ids_list])
                true_captions.extend([self.decode_caption(ids, tokenizer) for ids in batch["captions"].cpu().numpy().tolist()])
                row_ids.extend(batch["id"].cpu().numpy().tolist())
                if num_samples is not None and len(captions) >= num_samples:
                    break
        
        ret = {
            "id": row_ids[:num_samples] if num_samples is not None else row_ids,
            "generated_caption": captions[:num_samples] if num_samples is not None else captions,
            "true_caption": true_captions[:num_samples] if num_samples is not None else true_captions,
        }
        return ret
