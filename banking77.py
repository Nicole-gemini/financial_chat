# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BANKING77 dataset."""

import csv

import datasets


_CITATION = """\
@inproceedings{Casanueva2020,
    author      = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
    title       = {Efficient Intent Detection with Dual Sentence Encoders},
    year        = {2020},
    month       = {mar},
    note        = {Data available at https://github.com/PolyAI-LDN/task-specific-datasets},
    url         = {https://arxiv.org/abs/2003.04807},
    booktitle   = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020}
}
"""  # noqa: W605

_DESCRIPTION = """\
BANKING77 dataset provides a very fine-grained set of intents in a banking domain.
It comprises 13,083 customer service queries labeled with 77 intents.
It focuses on fine-grained single-domain intent detection.
"""

_HOMEPAGE = "https://github.com/PolyAI-LDN/task-specific-datasets"

_LICENSE = "Creative Commons Attribution 4.0 International"

_TRAIN_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


class Banking77(datasets.GeneratorBasedBuilder):
    """BANKING77 dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(
                    # 将英文标签替换为中文标签
                    names=[
                        "激活我的银行卡",
                        "年龄限制",
                        "第三方支付",
                        "ATM支持",
                        "自动充值",
                        "银行转账后余额未更新",
                        "支票或现金存入后余额未更新",
                        "收款方不被允许",
                        "取消转账",
                        "银行卡即将到期",
                        "银行卡受理范围",
                        "银行卡到达通知",
                        "银行卡派送预估时间",
                        "银行卡关联/绑定",
                        "银行卡无法使用",
                        "银行卡支付费用被扣",
                        "银行卡支付记录异常",
                        "银行卡支付汇率错误",
                        "银行卡被吞",
                        "现金取款手续费",
                        "现金取款记录异常",
                        "修改密码",
                        "银行卡信息泄露/被盗用",
                        "非接支付无法使用",
                        "国家/地区支持",
                        "银行卡支付被拒绝",
                        "取现被拒绝l",
                        "转账被拒绝",
                        "直接扣款记录异常",
                        "一次性虚拟卡限额",
                        "修改个人信息",
                        "汇兑费用",
                        "汇率",
                        "通过APP兑换货币",
                        "账单额外费用",
                        "转账失败",
                        "法定货币支持",
                        "申请一次性虚拟卡",
                        "申请实体卡",
                        "申请备用卡",
                        "申请虚拟卡",
                        "银行卡丢失或被盗",
                        "手机丢失或被盗",
                        "订购实体卡",
                        "忘记密码",
                        "银行卡支付待处理",
                        "取现待处理",
                        "充值待处理",
                        "转账待处理",
                        "密码被锁定",
                        "收款",
                        "退款未到账",
                        "请求退款",
                        "银行卡付款撤销",
                        "支持的卡种与货币",
                        "注销账户",
                        "银行转账充值费用",
                        "银行卡充值费用",
                        "现金或支票充值",
                        "充值失败",
                        "充值限额",
                        "充值已退回",
                        "使用银行卡充值",
                        "重复扣费",
                        "转账手续费被扣",
                        "转账至账户",
                        "收款人未收到转账",
                        "转账时间",
                        "无法验证身份",
                        "验证我的身份",
                        "验证资金来源",
                        "验证充值",
                        "虚拟卡无法使用",
                        "Visa或Mastercard",
                        "为什么要验证身份",
                        "收款金额错误",
                        "取现汇率错误",
                    ]
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            # call next to skip header
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                text, label = row
                yield id_, {"text": text, "label": label}
