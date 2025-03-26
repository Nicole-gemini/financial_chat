# data_processing.py
"""
数据处理模块
"""
import re
import json
import openai
from banking77 import Banking77
from tqdm import tqdm
from datasets import load_dataset,DatasetDict

global error
error = 0

class BankingDataProcessor:
    def __init__(self, config):
        self.non_financial = config["non_financial_regex"]
        self.term_mapping = config["term_mapping"]
        self.gpt_ch = config["chatgpt_params"]
        
    def re_filter(self, sample):
        """正则匹配剔除非金融场景样本"""
        text = sample["text"]
        label = sample["label"]
        for pattern in self.non_financial:
            if re.search(pattern, text, re.IGNORECASE):
                return None
        return {"text": text, "label": label}

    def expand_terms(self, sample):
        """英文术语处理"""
        text = sample["text"]
        label = sample["label"]
        for term, replacement in self.term_mapping.items():
            text = re.sub(rf'\b{term}\b', replacement, text, flags=re.IGNORECASE)
        return {"text": text, "label": label}
    

    def chinese_version(self, sample):
        """翻译成中文并重写"""
        text = sample["text"]
        label = sample["label"]
        for _ in range(3):  # 最大重试次数
            try:
                response = openai.ChatCompletion.create(
                    **self.gpt_ch,
                    messages=[{
                        "role": "system",
                        "content": "Rewrite this banking query using chinese, keep the original meaning."
                    },{
                        "role": "user",
                        "content": text
                    }]
                )
                return {"text": response.choices[0].message['content'].strip(), "label": label}
            except Exception as e:
                error += 1
                print(f"翻译失败: {error} 次，已返回原文")
                continue
        return {"text": text, "label": label}  # 失败时返回原文

    def process_dataset(self, dataset):
        """三重清洗流程"""
        # 第一阶段：正则过滤
        filtered_data = dataset.filter(self.re_filter)
        
        # 第二阶段：术语处理
        mapped_data = filtered_data.map(self.expand_terms)
        
        # 第三阶段：重写生成中文
        tqdm.pandas(desc="ChatGPT Rewriting")
        rewritten_data = mapped_data.map(
            self.chinese_version,
            batched=False,
            load_from_cache_file=False
        )
        
        return rewritten_data

def data_generator(file_path):
    """自定义数据集生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    config = {
        "non_financial_regex": [
        "\\be-?commerce\\b","\\brefund\\b","online\\s?shop",
        "\\bdelivery\\b","product\\s?return","\\bamazon\\b",
        "\\borders?\\b", "\\bcart\\b","\\bcheckout\\b",
        "\\bshipping\\b","\\binvoice\\b","\\bcoupon\\b",
        "\\bdiscount\\b","\\bpromo\\s?code\\b","product\\s?exchange",
        "\\btracking\\s?number\\b"
        ],
        "term_mapping": {
            "T+0": "real-time transfer",
            "CD": "Certificate of Deposit",
            "APR": "Annual Percentage Rate",
            "ACH": "Automated Clearing House",
            "PIN": "Personal Identification Number",
            "ATM": "Automated Teller Machine",
            "SWIFT": "Society for Worldwide Interbank Financial Telecommunication",
            "KYC": "Know Your Customer",
            "FICO": "FICO Score",
            "IRA": "Individual Retirement Account",
            "FDIC": "Federal Deposit Insurance Corporation",
            "401(k)": "Employer-Sponsored Retirement Plan"
        },
        "chatgpt_params": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.2,
            "max_tokens": 1024
        }
    }

    
    # 加载原始数据集
    origin_dataset = load_dataset("/home/featurize/work/finacial_chat/banking77.py", name="banking77")

    processor = BankingDataProcessor(config)
    
    # 处理训练集和测试集
    processed_lists = []
    for data in origin_dataset:
        processed = processor.process_dataset(data)
        processed_lists.append(processed)
    
    # 保存处理后的数据集
    final_dataset = DatasetDict({
        "train": processed_lists[0],
        "test": processed_lists[1]
    })
    final_dataset.save_to_disk("/home/featurize/data/processed_banking77")