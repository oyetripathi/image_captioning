import os
import random
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from PIL import Image
import wandb
from io import BytesIO
from collections import defaultdict
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class AsyncManager:
    def __init__(self, concurrency, timeout, max_retries):
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries

    async def download_image(self, session, url, log_fn=None):
        start_time = time.time()
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if log_fn:
                            log_fn(
                                {
                                    "loader/download_time": time.time() - start_time,
                                },
                                hist=True
                            )
                        return Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                await asyncio.sleep(0.5)
        if log_fn:
            log_fn(
                {
                    "loader/download_time": time.time() - start_time,
                },
                hist=True
            )
        return None

    async def download_multiple_images(self,urls, log_fn=None):
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            sem = asyncio.Semaphore(value=self.concurrency)
            async def bound_fetch(url):
                async with sem:
                    return await self.download_image(session, url, log_fn=log_fn)

            results = await asyncio.gather(*[bound_fetch(url) for url in urls])
    
            return results

class WBLogger:
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run
        self.hist_buffers = defaultdict(list)

    def log(self, data, hist=False):
        if not self.wandb_run:
            return
        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else 0
        for k, v in data.items():
            if not hist:
                    self.wandb_run.log({
                        f"{k}/worker_{wid}": v
                    }, commit=False)
            else:
                self.hist_buffers[k].append(v)
                if len(self.hist_buffers[k]) >= 50:
                    self.wandb_run.log({
                        f"{k}/worker_{wid}": wandb.Histogram(self.hist_buffers[k])
                    })
                    self.hist_buffers[k] = []
        return

class LAIONPOPDataset(IterableDataset):
    def __init__(self, pq_path, tokenizer, urls_per_batch=16, samples_per_worker=5000, concurrency=10, img_transform=None, y_transform=None, augmentations=None, wandb_run=None):
        self.pq_path = pq_path
        self.all_pq_files = [filename for filename in os.listdir(self.pq_path) if filename.endswith(".parquet")]
        self.df = None
        self.tokenizer = tokenizer
        self.urls_per_batch = urls_per_batch
        self.samples_per_worker = samples_per_worker
        self.img_transform = img_transform
        self.y_transform = y_transform
        self.augmentations = augmentations
        self.async_manager = AsyncManager(concurrency=concurrency, timeout=4, max_retries=2)
        self.wblogger = WBLogger(wandb_run)
        self.worker_num = None
        self.choose_pq()
    
    @staticmethod
    def validate_batch(image, caption):
        if image is None:
            return False
        if (image.size[0] < 150) or (image.size[1] < 150):
            return False
        if len(caption.split()) < 3:
            return False
        return True

    def choose_pq(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.worker_num = -1
            worker_files = self.all_pq_files
        else:
            k = len(self.all_pq_files) // worker_info.num_workers
            self.worker_num = worker_info.id
            worker_files = self.all_pq_files[(worker_num)*k:(worker_num+1)*k]
        
        chosen_pq  = random.choice(worker_files)
        self.df = pd.read_parquet(
            f"{self.pq_path}/{chosen_pq}"
        ).rename(
            columns={"llava_caption": "caption", "key": "id"}
        )[["id", "url", "caption"]]
        if len(self.df) > self.samples_per_worker:
            self.df = self.df.sample(n=self.samples_per_worker).reset_index(drop=True)
        return
    
    def get_images_from_list_id(self, indices):
        assert (not self.df is None)
        urls_to_fetch = [self.df[self.df["id"].astype(int)==idx]["url"].to_list()[0] for idx in indices]
        images = asyncio.run(self.async_manager.download_multiple_images(urls_to_fetch, log_fn=None))
        return images


    def __iter__(self):
        processed = 0
        skipped = 0
        start_time = time.time()
        last_hearbeat = time.time()

        for i in self.df.index[::self.urls_per_batch]:
            cur_time = time.time()
            if cur_time - last_hearbeat > 5:
                self.wblogger.log({"dataset/worker_heartbeat": self.worker_num, "dataset/worker_time_elapsed": cur_time - start_time})
                last_hearbeat = cur_time
            ret = []
            batch = self.df[i:i+self.urls_per_batch]
            urls = batch["url"].to_list()
            captions = batch["caption"].to_list()
            ids = batch["id"].to_list()
            images = asyncio.run(self.async_manager.download_multiple_images(urls, log_fn=self.wblogger.log))
            for row_id, image, caption in zip(ids, images, captions):
                if not self.validate_batch(image, caption):
                    skipped += 1
                    self.wblogger.log({"dataset/invalid_samples_skipped": skipped})
                    continue
                self.wblogger.log(
                    {
                        "dataset/image_size/w_hist": image.size[0],
                        "dataset/image_size/h_hist": image.size[1],
                        "dataset/caption_length_hist": len(caption.split())
                    },
                    hist=True
                )
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    self.wblogger.log(
                        {
                            "dataset/processed_samples": processed,
                            "dataset/processing_rate_hist": processed / elapsed,
                        }
                    )
                if self.augmentations:
                    image = Image.fromarray(self.augmentations(image=np.array(image))["image"])
                if self.img_transform:
                    image = self.img_transform(image)
                token_ids, attention_mask = self.tokenizer.encode(caption)
                yield {
                    "id": int(row_id),
                    "image": image,
                    "input_ids": token_ids,
                    "attention_mask": attention_mask
                }