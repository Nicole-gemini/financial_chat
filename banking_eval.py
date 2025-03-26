# evaluate.py
"""
评估脚本
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_from_disk
import pandas as pd
import json

class BankingEvaluator:
    def __init__(self, config):
        # 初始化
        self.config = config
        self.device_use = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.test_dataset = None
        
    def setting(self):
        # 准备数据集
        self._load_dataset()
        
        # 准备分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"],
            padding_side="right",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 准备模型
        self._load_model()

    def _load_dataset(self):
        # 加载测试数据集
        dataset = load_from_disk(self.config["data_path"])
        self.test_dataset = dataset["test"]
        print(f"测试集共有: {len(self.test_dataset)} 个样本")
    
    def _load_model(self):
        # 加载基座模型
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["base_model"],
            quantization_config=bnb_config,
            device_use=self.device_use,
            trust_remote_code=True
        )
        # 加载LoRA适配器
        self._lora_adapter()

    def _lora_adapter(self):
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config["lora_path"],
            adapter_name="banking77_adapter",
            adapter_type="text_task",
        )
        self.model.eval()


    def evaluate(self):
        # 评估模型效果
        
        # 推理管道
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_use,
            batch_size=self.config["batch_size"],
            truncation=True
        )
        
        # 预测结果
        print("预测结果： ")
        texts = self.test_dataset["text"]
        true_labels = self.test_dataset["label"].tolist()
        
        predictions = []
        for i in tqdm(range(0, len(texts), self.config["batch_size"]),
                     desc="progress",
                     total=len(texts)//self.config["batch_size"]+1):
            batch = texts[i:i+self.config["batch_size"]]
            results = classifier(batch)
            predictions.extend([int(res["label"].split("_")[-1]) for res in results])
        
        # 评估指标
        self._metrics(true_labels, predictions)
        
    def _metrics(self, y_true, y_pred):
        # 实现多个评估指标
        print("\n评估指标:")
        
        # 准确率
        accuracy = accuracy_score(y_true, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=self._get_label_names()))
        
        # 混淆矩阵分析
        self._bank_cm(y_true, y_pred)
        
        # 保存结果
        self._save_results(y_true, y_pred, accuracy)
    
    def _get_label_names(self):
        # 获取类别标签名称，具体看Banking77.py中的中文类别标签名称
        return [f"class_{i}" for i in range(77)]  # Banking77有77个类别
    
    def _bank_cm(self, y_true, y_pred):
        # 混淆矩阵分析
        cm = confusion_matrix(y_true, y_pred)
            
        # 保存混淆矩阵
        pd.DataFrame(cm).to_csv("/home/featurize/data/Bank_confusion_matrix.csv", index=False)
    
    def _save_results(self, y_true, y_pred, accuracy):
        # 保存评估结果
        results = {
            "Accuracy": accuracy,
            "Predictions": y_pred,
            "True_labels": y_true
        }
        with open("/home/featurize/data/Bank_evaluation.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    config = {
        "base_model": "/home/featurize/model/deepseek-llm-7b-chat",
        "data_path": "/home/featurize/data/processed_banking77",
        "lora_path": "/home/featurize/data/banking77_lora",
        "batch_size": 16,
        "max_seq_length": 512
    }
    
    evaluator = BankingEvaluator(config)
    evaluator.setting()
    evaluator.evaluate()