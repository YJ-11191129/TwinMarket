# TwinMarket - A股市场模拟系统（1.0版本）

<p align="center">[ <a href="README.md">English</a> | <a href="README_zh.md">中文</a> ]</p>

<div align="center">
  <img src="assets/img/TwinMarket.png" alt="TwinMarket Overview" width="100%" style="max-width: 1000px; margin: 0 auto; display: block;">
</div>

## 📖 项目简介

TwinMarket 是一个创新的股票交易模拟系统，通过集成大语言模型（LLM）技术，模拟真实的股票市场交易环境。系统通过多智能体协作，实现了包括个性化交易策略、社交网络互动、新闻信息分析等在内的全方位市场模拟。

### 🎯 核心特性

- **🤖 智能交易代理**：基于 LLM 的个性化交易决策系统
- **🌐 社交网络模拟**：完整的论坛互动和用户关系网络
- **📊 多维度分析**：整合技术指标、新闻信息、市场情绪等多种因素
- **🎲 行为金融建模**：考虑处置效应、彩票偏好等行为金融因素
- **⚡ 高性能并发**：支持大规模用户并发交易模拟
- **📈 实时撮合引擎**：完整的订单撮合和交易执行系统

## 🚀 快速开始

```bash
# 自行配置 API 与 embedding 模型：
cp config/api_example.yaml config/api.yaml
cp config/embedding_example.yaml config/embedding.yaml

# 运行样例
bash script/run.sh
```

## 📝 开发指南

### 离线恐慌机制与反转实验

新增不依赖模型 API 的小规模实验：区分信息冲击、流动性压力、羊群传播和强制去杠杆，记录信念与被动交易，并通过配对对照评估反转因子的未来条件收益。它是未经真实数据校准的机制实验，不代表现实交易 Alpha。运行方式、日频/Tick 字段与 TwinMarket 适配边界见 [实验说明](experiments/panic_reversal/README.md)。

### 扩展交易策略

在 `trader/trading_agent.py` 中实现新的交易策略：

```python
def custom_strategy(self, market_data):
    """自定义交易策略"""
    # 实现你的策略逻辑
    pass
```

### 添加新的评估指标

在 `trader/utility.py` 中添加评估函数：

```python
def calculate_custom_metric(trades):
    """计算自定义指标"""
    # 实现指标计算
    pass
```

## 📚 使用 TwinMarket 的优秀论文

欢迎社区贡献。如果你的论文使用了 TwinMarket，欢迎提交 PR 添加到这里。

| 标题 | 代码 | 论文 |
| --- | --- | --- |
| Interpreting Emergent Extreme Events in Multi-Agent Systems | https://github.com/mjl0613ddm/IEEE | https://arxiv.org/abs/2601.20538 |

## 🧾 引用

```bibtex
@inproceedings{yang2025twinmarket,
      title={TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets},
      author={Yuzhe Yang and Yifei Zhang and Minghao Wu and Kaidi Zhang and
              Yunmiao Zhang and Honghai Yu and Yan Hu and Benyou Wang},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
      series={NeurIPS},
      volume={39},
      year={2025},
      url={https://arxiv.org/abs/2502.01506},
}
```

## 📄 许可证

本项目采用 MIT License，详见 `LICENSE`。

