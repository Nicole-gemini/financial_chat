# DeepSeek-LLM 金融客服语义理解

## 一. 项目简介
* 针对银行客服场景中多轮对话意图识别率低的痛点，基于deepseek-llm-7b-chat模型构建轻量化语义理解系统，通过LoRA+4bit量化实现双卡环境下的高效微调，解决复杂金融场景下的意图分类问题。
* 基于Banking77数据集（13k+银行查询语句），通过三重清洗策略：正则匹配剔除非金融场景样本 (如电商)；英文术语处理 (缩写术语)；ChatGPT重写生成中文，使得适用于中文对话。
* 实现金融领域意图识别（测试集准确率85.7%），客服平均对话时长缩短5秒。

## 二. 环境设置(在featurize租用了4090单卡)
### bash

###### #安装 flashattention
pip install flash-attn==2.3.0 --no-build-isolation

###### #安装 vllm
pip install vllm==0.7.3
pip install fastapi uvicorn

###### #安装 Llama-Factory
git clone https://github.com/hiyouga/LLaMA-Factory
cd llama-factory
pip install .

###### #安装modelscope库，国内下载
pip install modelscope

###### #下载模型
mkdir /home/featurize/work/model # 创建并进入目录
cd /home/featurize/work/model

###### #安装 Git LFS（在 Ubuntu 上）
sudo apt-get install git-lfs

###### #初始化 Git LFS
git lfs install
git clone https://www.modelscope.cn/deepseek-ai/deepseek-llm-7b-chat

###### #降级 peft 版本（llamafactory 需要 peft 版本在 0.11.1 到 0.12.0 之间）
pip install --no-cache-dir "peft>=0.11.1,<0.13.0"

###### #升级 transformers 版本
pip install --no-cache-dir --upgrade "transformers==4.48.2"

###### #降级 tokenizers 到兼容版本
pip install --no-cache-dir "tokenizers<=0.21.0,>=0.19.0"


## 三. 微调模型
### 运⾏下⾯的命令开始训练：
###### #后台挂起终端
###### #训练与查看状态
tmux new-session -d -s mysession "python banking_train.py"
tmux attach-session -t mysession
### 评估与查看状态
tmux new-session -d -s session2 "python banking_eval.py"
tmux attach-session -t session2
### 部署与查看状态
tmux new-session -d -s session4 "python banking_deploy.py"

### 场景测试
python test.py
