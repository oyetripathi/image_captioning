import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class AsyncManager:
    def __init__(self, concurrency, timeout, max_retries):
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries

    async def download_image(self, session, url):
        for _ in range(self.max_retries):
            try:
                async with session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        return Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                await asyncio.sleep(0.5)
        return None

    async def download_multiple_images(self,urls):
        connector = aiohttp.TCPConnector(limit_per_host=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            sem = asyncio.Semaphore(value=self.concurrency)
            async def bound_fetch(url):
                async with sem:
                    return await self.download_image(session, url)
        
            results = await asyncio.gather(*[bound_fetch(url) for url in urls])
    
            return results

class ConceptualCaptionsDataset(IterableDataset):
    def __init__(self, df, tokenizer, samples_per_worker=5000, urls_per_batch=16, concurrency=10, img_transform=None, y_transform=None):
        self.df = df.sample(n=samples_per_worker).reset_index(drop=True)
        self.urls_per_batch = urls_per_batch
        self.img_transform = img_transform
        self.y_transform = y_transform
        self.async_manager = AsyncManager(concurrency=concurrency, timeout=4, max_retries=2)
    
    @staticmethod
    def validate_batch(image, caption):
        if image is None:
            return False
        if (image.size[0] < 150) or (image.size[1] < 150):
            return False
        if len(caption.split()) < 3:
            return False
        return True

    def __iter__(self):
        i = 0
        while(i < len(self.df)):
            ret = []
            for j in range(100):
                batch = self.df[i:i+self.urls_per_batch]
                urls = batch["url"].to_list()
                captions = batch["caption"].to_list()
                images = asyncio.run(self.async_manager.download_multiple_images(urls))
                ret.extend([(image,caption) for (image ,caption) in zip(images, captions) if self.validate_batch(image, caption)])
                i += self.urls_per_batch

                if (len(ret) >= self.urls_per_batch) or (i >= len(self.df)):
                    break
            
            for img, cap in ret:
                if self.img_transform:
                    img = self.img_transform(img)
                yield {
                    "id": 0,
                    "image": img,
                    "input_ids": [0,1],
                    "attention_mask": [1,1]
                }