from fastapi import FastAPI
from vllm import AsyncLLMEngine, AsyncLLMEngineClient
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import uvicorn
import os

# 配置
MODEL_PATH = "/home/featurize/data/banking77_merged"
HOST = "0.0.0.0"
PORT = 8000

# 定义FastAPI
app = FastAPI(title="Banking Intent Model", description="基于VLLM部署的银行意图分类模型")

# 设置vLLM引擎
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    tensor_parallel_size=1,          # 多GPU并行数，此处设为1
    dtype="bfloat16",                # 量化类型
    max_num_seqs=256,                # 最大并发序列数
    max_model_len=1024,              # 最大序列长度（与训练时保持一致）
    disable_log_stats=False,         # 启用性能统计
    worker_use_ray=False,            # 是否使用Ray分布式
    engine_use_ray=False,
)

# 设置异步引擎
engine = AsyncLLMEngine.from_engine_args(engine_args)

# 定义预测API
@app.post("/predict")
async def pred_intent(text: str):
    # 处理单条文本的意图预测结果
    sampling_params = SamplingParams(
        temperature=0.01,    # 低温度确保确定性输出
        top_p=0.95,          # 保留概率高于0.95的token
        max_tokens=10,       # 意图分类不需要长文本生成，所以设置为10
        n=1                  # 只生成1个结果
    )
    
    # 推理
    request_id = os.urandom(16).hex()  # 得到唯一请求ID
    results_generator = engine.generate(
        prompt=f"以下是一段用户请求：'{text}'。\n请直接返回该请求的意图类别：",
        sampling_params=sampling_params,
        request_id=request_id
    )
    
    # 获取结果
    res = None
    async for request_output in results_generator:
        res = request_output
    
    # 解析结果
    label = res.outputs[0].text.strip()
    return {"text": text, "label": label}

# 启动服务
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="log",
        timeout_keep_alive=30  # 保持连接超时时间
    )