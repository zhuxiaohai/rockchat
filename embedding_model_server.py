import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sentence_transformers import SentenceTransformer
from typing import List
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.status import HTTP_504_GATEWAY_TIMEOUT
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor

batch_size = 32 # gpu batch_size in order of your available vram

max_request = 10  # max request for future improvements on api calls / gpu batches (for now is pretty basic)
request_flush_timeout = .1  # flush time out for future improvements on api calls / gpu batches (for now is pretty basic)
request_time_out = 30  # Timeout threshold
gpu_time_out = 5  # gpu processing timeout threshold
port = 8501


class BAAIWrapper:
    def __init__(self, embedding_name: str):
        self.embedding_model = SentenceTransformer(embedding_name)

    def embed(self, sentences: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=batch_size
        )
        embeddings = embeddings.tolist()
        return embeddings


class EmbedRequest(BaseModel):
    sentences: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class RequestProcessor:
    def __init__(self, model: BAAIWrapper, max_request_to_flush: int, accumulation_timeout: float):
        self.model = model
        self.max_batch_size = max_request_to_flush
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue()
        self.response_futures = {}
        self.processing_loop_task = None
        self.processing_loop_started = False  # Processing pool flag lazy init state
        self.executor = ThreadPoolExecutor()  # Thread pool
        self.gpu_lock = asyncio.Semaphore(1)  # Sem for gpu sync usage

    async def ensure_processing_loop_started(self):
        if not self.processing_loop_started:
            print('starting processing_loop')
            self.processing_loop_task = asyncio.create_task(self.processing_loop())
            self.processing_loop_started = True

    async def processing_loop(self):
        while True:
            requests, request_types, request_ids = [], [], []
            start_time = asyncio.get_event_loop().time()

            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (asyncio.get_event_loop().time() - start_time)
                if timeout <= 0:
                    break

                try:
                    req_data, req_type, req_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    requests.append(req_data)
                    request_types.append(req_type)
                    request_ids.append(req_id)
                except asyncio.TimeoutError:
                    break

            if requests:
                await self.process_requests_by_type(requests, request_types, request_ids)

    async def process_requests_by_type(self, requests, request_types, request_ids):
        tasks = []
        for request_data, request_type, request_id in zip(requests, request_types, request_ids):
            if request_type == 'embed':
                task = asyncio.create_task(
                    self.run_with_semaphore(self.model.embed, request_data.sentences, request_id))
            else:  # 'rerank'
                task = asyncio.create_task(
                    self.run_with_semaphore(self.model.rerank, request_data.sentence_pairs, request_id))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def run_with_semaphore(self, func, data, request_id):
        async with self.gpu_lock:  # Wait for sem
            future = self.executor.submit(func, data)
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=gpu_time_out)
                self.response_futures[request_id].set_result(result)
            except asyncio.TimeoutError:
                self.response_futures[request_id].set_exception(TimeoutError("GPU processing timeout"))
            except Exception as e:
                self.response_futures[request_id].set_exception(e)

    async def process_request(self, request_data: EmbedRequest, request_type: str):
        try:
            await self.ensure_processing_loop_started()
            request_id = str(uuid4())
            self.response_futures[request_id] = asyncio.Future()
            await self.queue.put((request_data, request_type, request_id))
            return await self.response_futures[request_id]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error {e}")


app = FastAPI()

# Initialize the model and request processor
model = BAAIWrapper("/workspace/data/private/zhuxiaohai/models/bge_finetune_emb")
processor = RequestProcessor(model, accumulation_timeout=request_flush_timeout, max_request_to_flush=max_request)


# Adding a middleware returning a 504 error if the request processing time is above a certain threshold
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        return await asyncio.wait_for(call_next(request), timeout=request_time_out)

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        return JSONResponse({'detail': 'Request processing time excedeed limit',
                             'processing_time': process_time},
                            status_code=HTTP_504_GATEWAY_TIMEOUT)


@app.post("/embeddings/", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    embeddings = await processor.process_request(request, 'embed')
    return EmbedResponse(embeddings=embeddings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
