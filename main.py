from typing import List, Tuple, Union, cast
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.status import HTTP_504_GATEWAY_TIMEOUT
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline import QAPineline
from search_engine import QASearchEngine, VectorSim
from find_keywords import LabellerByRules
from recall import RecallBySearchEngine
from merge import QAMerge
from rank import QAScorer
from rerank import QAReranker


max_request = 10  # max request for future improvements on api calls / gpu batches (for now is pretty basic)
request_flush_timeout = .1  # flush time out for future improvements on api calls / gpu batches (for now is pretty basic)
request_time_out = 60  # Timeout threshold
gpu_time_out = 20  # gpu processing timeout threshold
port = 8501


class PipelineWrapper:
    def __init__(self, pipeline_config):
        self.qa_pipeline = QAPineline(pipeline_config)

    def get_qa(self, query: str) -> List[dict]:
        qa_results = self.qa_pipeline.run(
            query
        )
        return qa_results


class PipelineRequest(BaseModel):
    query: str


class PipelineResponse(BaseModel):
    qa_results: List[dict]


class RequestProcessor:
    def __init__(self, model: PipelineWrapper, max_request_to_flush: int, accumulation_timeout: float):
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
            task = asyncio.create_task(
                self.run_with_semaphore(self.model.get_qa, request_data.query, request_id))
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

    async def process_request(self, request_data: PipelineRequest, request_type: str):
        try:
            await self.ensure_processing_loop_started()
            request_id = str(uuid4())
            self.response_futures[request_id] = asyncio.Future()
            await self.queue.put((request_data, request_type, request_id))
            return await self.response_futures[request_id]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error {e}")


app = FastAPI()

keywords_config = {
    "class": LabellerByRules,
    "config": {
        "dim_df_path": "data/dim_df20240315.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_name", "cat"),
        "error_col": ("error", "error"),
    }
}

recall_config = {
    "vector_search": {
        "class": RecallBySearchEngine,
        "config": {
            "search_engine": {
                "class": QASearchEngine,
                "database_path": "data/database20240506.csv",
                "id_col": "qa_id",
                "index_columns": [("model_list", "model"), ("cat_name", "cat"), ("error_list", "error")],
                "score_model": {
                    "type": "vector",
                    "class": VectorSim,
                    "embedding_col": "question",
                    "embedding_model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_emb"
                },
            },
        },
        "top_n": 10,
    }
}

merge_config = {
    "class": QAMerge,
    "config": {
        "vector_search": 1,
    }
}

rank_config = {
    "class": QAScorer,
    "config": {
        "model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_reranker_question_top20",
        "query_key": "query_cleaned",
        "item_key": "question",
        "database_path": "data/database20240506.csv",
    }
}

rerank_config = {
    "class": QAReranker,
    "config": {
        "rank_key": [("rank", False)],
        "show_cols": ["question", "answer"],
        "reranking_scheme": {
            "recall_ranking_score_threshold": 0.75,
            "recall_ranking_top_n": 2,
        },
        "database_path": "data/database20240506.csv",
    }
}

pipeline_config = {
    "router": keywords_config,
    "recall": recall_config,
    "merger": merge_config,
    "ranker": rank_config,
    "reranker": rerank_config,
}

model = PipelineWrapper(pipeline_config)

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


@app.post("/qa/", response_model=PipelineResponse)
async def get_results(request: PipelineRequest):
    qa_results = await processor.process_request(request, 'get_qa')
    return PipelineResponse(qa_results=qa_results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
