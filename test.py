import requests
import json

bank_api = "http://localhost:8000/predict"

def bank_test(text):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        bank_api,
        data=json.dumps(payload),
        headers=headers
    )
    return response.json()

# 测试
if __name__ == "__main__":
    samples = [
        "如何将资金转到另一个账户？",
        "我的信用卡年费是多少？",
        "最近的ATM在哪里？",
        "如何通过手机银行向境外账户转账？需要提供哪些信息？",
        "我的信用卡昨天被盗刷了，现在需要挂失并补办新卡",
        "请问你们支行周六营业吗？",
        "我想了解当前三年期大额存单的利率，最低起存金额是多少？"
    ]
    
    for text in samples:
        res = bank_test(text)
        print(f"输入: {res['text']}")
        print(f"预测意图: {res['label']}\n")