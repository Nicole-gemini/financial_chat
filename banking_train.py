# finetune_lora.py
# 用于训练Banking77数据集的LoRA微调脚本
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, DatasetDict
import torch
import numpy as np

# 配置信息

class BankingTrainer:
    def __init__(self, config):
        self.config = config
        self.device_use = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setting(self):
        # 加载数据集
        self._load_dataset()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"],
            padding_side="right",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 准备模型
        self._load_model()
        
    def _load_dataset(self):
        dataset = load_from_disk(self.config["data_path"])
        train_val = dataset["train"].train_test_split(
                                        train_size=0.8, # 80% 作为新训练集
                                        test_size=0.2,  # 20% 作为验证集
                                        shuffle=True,   # 随机打乱数据
                                        seed=42         # 固定随机种子以保证可复现性
                                    )
        self.train_dataset = train_val["train"] # 新训练集
        self.val_dataset = train_val["test"] # 验证集           
        print(f"数据集:\n"
              f"训练集共有: {len(self.train_dataset)} 个样本\n"
              f"验证集集共有: {len(self.val_dataset)} 个样本")

    def _load_model(self):
        # 加载并配置4bit量化模型
        
        # 4bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )


        # 加载分类模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["base_model"],
            num_labels=77,  # Banking77的类别数
            quantization_config=bnb_config,
            device_map=self.device_use,
            trust_remote_code=True
        )
        
        # 添加LoRA适配器
        self._lora_adapter()
        
    def _lora_adapter(self):
        # 添加LoRA微调模块
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            changed_modules=self.config["changed_modules"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def tokenize_batch(self, sample):
        # 批量分词处理
        return self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    def train(self):
        # 训练
        # 数据集预处理
        train_data = self.train_dataset.map(
            self.tokenize_batch,
            batched=True,
            batch_size=512,
            num_proc=4,
            remove_columns=["text"]  # 移除原始文本字段
        )
        
        val_data = self.val_dataset.map(
            self.tokenize_batch,
            batched=True,
            batch_size=512,
            num_proc=4,
            remove_columns=["text"]
        )
        # 数据转换为PyTorch格式
        train_data.set_format("torch") 
        val_data.set_format("torch")
        # 训练参数
        train_params = TrainingArguments(
            output_dir="/home/featurize/data/banking77_lora",
            per_device_train_batch_size=8,  # 4bit量化允许更大batch
            per_device_eval_batch_size=16,
            learning_rate=1e-4,
            num_train_epochs=5,  # 根据清洗后数据质量增加轮次
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            logging_steps=50,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )
        
        # 自定义评估指标
        def eval_metrics(eval_pred):
            pred, labels = eval_pred
            pred = np.argmax(pred, axis=1)
            return {"accuracy": (pred == labels).mean()}
        
        # 初始化Trainer
        mytrainer = Trainer(
            model=self.model,
            args=train_params,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=eval_metrics,
            tokenizer=self.tokenizer
        )
        
        # 执行训练
        print("训练开始：")
        mytrainer.train()
        
        # 保存最佳模型
        mytrainer.save_model("/home/featurize/data/banking77_lora")
        print("训练完成！")

if __name__ == "__main__":
    # 训练配置
    config = {
        "base_model": "/home/featurize/work/model/deepseek-llm-7b-chat",
        "data_path": "/home/featurize/data/processed_banking77",
        "changed_modules": ["q_proj", "v_proj"],
        "lora_rank": 8,
        "precision": "fp16",
        "max_seq_length": 512
    }
    
    # 初始化并运行
    bank_trainer = BankingTrainer(config)
    bank_trainer.setting()
    bank_trainer.train()

    # 训练命令
    """
    # 单卡训练
    python banking_train.py

    # 多卡训练
    torchrun --nproc_per_node=2 banking_train.py
    """
    # 监控
    """
    tensorboard --logdir=/home/featurize/data/banking77_lora/runs
    """
