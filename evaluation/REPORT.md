# SuperMew RAG 检索评估报告

## 1. 实验目的

验证不同检索策略对真实行业白皮书问题的证据召回效果，并为后续 RAG 优化提供依据。

## 2. 实验设置

- 数据：`H3_AP202505221677044543_1.pdf`，新茶饮行业白皮书。
- 评估集：30 条人工整理问题，覆盖定义、数字、时间、列表、供应链、门店运营、数字化和跨页问题。
- 返回数量：Top-5。
- 评估指标：Hit Rate@5、MRR、关键词覆盖率、平均延迟。

## 3. 实验结果

| 策略 | Hit@5 | MRR | 关键词覆盖率 | 平均延迟 |
|---|---:|---:|---:|---:|
| Dense | 1.000 | 0.906 | 0.847 | 216.6 ms |
| BM25 Sparse | 0.833 | 0.773 | 0.711 | 6.5 ms |
| Hybrid + RRF | 1.000 | 0.894 | 0.909 | 199.7 ms |
| Hybrid + Rerank | 未配置 | 未配置 | 未配置 | 未执行 |

## 4. 结论

1. Dense 和 Hybrid 在本次 30 条评估集上的 Top-5 页面命中率均为 1.000。
2. Dense 的 MRR 为 0.906，略高于 Hybrid 的 0.894；但 Hybrid 的关键词覆盖率更高，说明不同指标对排序和证据覆盖的侧重点不同。
3. BM25 延迟最低，但 Hit@5 和 MRR 低于语义检索，单独使用不适合作为当前主检索策略。
4. 当前 Hybrid 是准确性、排序质量和延迟之间的较好折中。
5. 本次运行未配置 Rerank 服务，因此没有把 Hybrid + Rerank 纳入有效对比；配置服务后再单独进行 Rerank 实验。

## 5. 复现实验

```powershell
docker compose up -d
.\.venv\Scripts\python.exe evaluation\evaluate_retrieval.py --top-k 5
```

原始逐题结果保存在 `evaluation/results.json`。

> 本报告当前指标来自 30 条评估问题。30 条样本适合工程方向判断，但仍不代表通用 benchmark 成绩。
