# GreenSpec RAG

面向建筑规范场景的可追溯混合检索与证据包服务。项目将规范条文、表格、公式及其页码定位统一为 canonical JSONL，通过 BM25、BGE 向量检索、RRF、去重/多样性约束与本地 reranker 生成受预算约束的 `EvidencePack`，供上层研究 Agent 引用与核验。

本仓库重点解决三个问题：

- 建筑规范中的条文、跨页表格和公式能否被稳定召回；
- 返回证据能否保留标准版本、条文号、表/公式 ID 和 PDF 页码；
- 检索、答案生成和上层报告中的声明能否分别评测，而不是用单一均值掩盖问题。

## 系统流程

```text
原始规范 PDF / Markdown
        ↓
canonical JSONL（条文、表格、公式、父子关系、来源定位）
        ↓
BM25 Top50 ─┐
             ├─ RRF(k=60) → 去重/多样性 → rerank Top30 → TopK
BGE Top50 ──┘
        ↓
EvidencePack（主证据 + 必需依赖 + 引用定位，默认预算 5000 tokens）
        ↓
greenspec / 其他下游 Agent
```

关键设计：

- canonical JSONL 是事实源，BM25、Qdrant collection 和缓存均可重建；
- dense embedding 使用本地 `bge-base-zh-v1.5`；
- reranker 使用本地 `bge-reranker-v2-m3`，默认重排 RRF/多样性后的 30 个候选；
- EvidencePack 默认最多选择 8 条主证据，并补充必需表格、公式和直接父级信息；
- EvidencePack 默认总预算为约 5000 tokens，超预算时优先保留 required evidence、续表和引用定位；
- 依赖不可用时默认 fail-closed，只有请求显式允许才返回降级结果。

## 当前评测结果

以下指标来自冻结的 candidate manifest 与评测数据集。完整规范语料、索引、逐条答案、API judge 日志和人工复核记录不提交到公开仓库，只公开聚合结果。

### 检索侧

确定性评测使用 92 条带 required evidence 的正例：

- Hybrid Hit@10：**89/92，96.74%**；
- 条文 Hit@10：**89/92，96.74%**；
- ID-based Context Recall：**95.65%**。

RAGAS 检索语义评测：

- Context Precision：**0.8711**，覆盖 **92/92**；
- Context Relevance：**0.9864**，覆盖 **92/92**；
- Context Recall：已评分样本均值 **0.9951**，覆盖 **81/92**。

Context Recall 仍有 11 条 judge 输出缺失，因此 `0.9951` 仅代表已评分的 81 条，不能作为完整 92 条的发布结论。

### EvidencePack 与引用侧

对 98 条预算后 EvidencePack 完成人工全量复核：

- Citation Accuracy：**98/98，100%**；
- Citation Completeness：**94/98，95.92%**；
- EvidencePack Supports Question：**86/98，87.76%**；
- 超出 5000-token 预算的 EvidencePack：**0**。

Citation Accuracy 检查引用是否指向正确标准、条文、表格/公式和页码；Citation Completeness 检查回答问题所需的关键引用是否齐全；EvidencePack Supports Question 检查预算后证据包整体是否足以回答问题。

### 答案侧

使用 94 条冻结答案和冻结 EvidencePack，以 `qwen-plus` 作为 RAGAS judge、本地 `bge-base-zh-v1.5` 作为 Answer Relevancy embedding：

- Faithfulness：**0.7612**，覆盖 **94/94**；
- Answer Relevancy：**0.8047**，覆盖 **94/94**；
- answer error：**0**；
- judge error：**0**。

Faithfulness 衡量答案声明能否由当前检索上下文推出；Answer Relevancy 衡量答案是否直接、完整地回应问题，不能替代事实正确性或引用完整性检查。

在 20 条高分歧/低分对抗性样本上进行人工校准：

- Faithfulness 严格一致：Qwen **9/20**，DeepSeek **3/20**；
- Answer Relevancy 严格一致：Qwen **18/20**，DeepSeek **17/20**。

该 20 条样本不是随机总体样本，以上数据只用于 judge 选型，不能解释为模型总体准确率。当前正式答案侧 judge 采用 `qwen-plus`。

### Agent 端到端验证

GreenSpec 作为外接知识库接入 `greenspec` 后，在一次真实北京办公建筑绿色建筑预审任务中：

- 端到端确定性检查 **10/10 PASS**；
- 返回本地规范证据 12 条、WebEvidenceLedger 联网证据 4 条；
- claim-level verification 为 **3/3 supported**；
- 项目关键参数不足时由 gate 阻止直接给出高风险达标结论。

该结果属于真实任务的 contract/gate 验证，不等同于对长篇最终报告计算 RAGAS Faithfulness。

## 评测口径与可追溯信息

- Candidate manifest：`hybrid_20260901T100941Z_21691b37`；
- Retrieval dataset：`candidate_ragas_v2_a20001c23c49_retrieval`；
- Retrieval manifest SHA-256：`d8288f25e57984934a6234965269337921b514840bc99c0d9879d54e7496b193`；
- Answer dataset：`candidate_ragas_v2_a20001c23c49_answer`；
- Answer manifest SHA-256：`755a6d19391395db86e0c0b502976f8f00c47b757e7f28b9986ad7cc6d608352`；
- RAGAS：`0.4.3`；
- 答案 judge：`qwen-plus`；
- Answer Relevancy embedding：本地 `bge-base-zh-v1.5`，CPU，strictness=3。

对比不同运行时必须保持数据集、manifest、EvidencePack、答案、judge 配置和 metric coverage 一致。缺失值不计入均值，并必须单独报告覆盖率。

## 仓库结构

```text
contracts/        EvidencePack、评测标注和 API schema
deploy/           构建、导出、评测、复核队列和发布脚本
greenspec_rag/    canonical 适配、检索、rerank、EvidencePack 与 API
prd/              产品与架构设计说明
tests/            协议、检索、证据包和评测适配器测试
compose.yaml      Qdrant、模型诊断、RAG API 与 RAGAS evaluator
```

`data/`、本地模型、索引、运行 trace 和评测明细由 `.gitignore` 排除。建筑规范原文可能受许可约束，本仓库不分发原始语料。

## 快速启动

环境要求：Python 3.10+、Docker Desktop；完整 hybrid/rerank 流程还需要 NVIDIA GPU 与可用的容器 GPU 运行时。

复制本地配置：

```powershell
Copy-Item .env.example .env
```

`.env` 用于本机运行路径和评测 API Key，不得提交。服务本身不依赖外部 LLM；只有一次性的 RAGAS evaluator 会读取 judge key。

下载并检查本地模型：

```powershell
docker compose --profile setup run --rm model-download
docker compose run --rm gpu-diagnostics python deploy/verify_embedding.py
docker compose run --rm gpu-diagnostics python deploy/verify_reranker.py
```

已有本地 canonical 和索引时，启动服务：

```powershell
docker compose up -d qdrant rag-api
Invoke-RestMethod http://127.0.0.1:8787/health
```

检索接口为 `POST /v1/retrieve`，请求、响应和错误结构见 `contracts/openapi.yaml`。每次 Docker 或 Qdrant 重启后可执行：

```powershell
python -m deploy.verify_baseline_startup
```

## 测试与评测入口

运行协议与检索测试：

```powershell
python -m unittest discover -s tests
```

安装可选 RAGAS 依赖：

```powershell
python -m pip install -e ".[ragas-eval]"
```

主要评测模块：

```text
deploy/evaluate_retrieval_modes.py
deploy/evaluate_required_evidence.py
deploy/prepare_candidate_ragas_datasets.py
deploy/run_ragas_retrieval_trial.py
deploy/run_ragas_answer_trial.py
deploy/build_candidate_evidence_pack_review.py
deploy/summarize_candidate_evidence_pack_review.py
```

## 数据与安全边界

- `.env`、原始规范、canonical 正文、索引和模型不进入 Git；
- 完整问题、答案、EvidencePack、judge diagnostics 和人工复核记录不公开；
- API Key 不写入源码、README 或命令参数；
- 对外结果只发布聚合指标，并同时注明数据集、manifest、judge 和覆盖率；
- 本仓库不包含运行完整系统所需的受许可规范语料。
