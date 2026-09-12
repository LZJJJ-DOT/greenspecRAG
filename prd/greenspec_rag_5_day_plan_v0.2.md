# GreenSpec RAG 5 天开发计划

> 版本：v0.2  
> 日期：2026-08-25  
> 对应 PRD：greenspec_rag_prd_v0.3.md  
> 目标：在资源前置满足的情况下跑通完整的 BM25 + Qdrant dense + RRF + reranker + Evidence Pack 闭环。原 v0.1 保留不变。

## 0. 前置资源门禁（Day 0）

进入五日实施前必须确认：

- Qdrant endpoint、collection 名称、持久化目录和可写权限。
- bge-base-zh-v1.5 与 bge-reranker-v2-m3 的本地权重路径、版本、设备和最大长度。
- 两份清洗 Markdown、对应原始 PDF、图片/表格/公式资源和允许的数据目录。
- Python 环境、SQLite FTS5、jieba 0.42.1、模型依赖和可写索引目录。

缺少上述资源时，只允许运行协议、解析和 mock 测试；不得把 BM25-only 或接口占位宣称为完整混合检索完成。

## Day 1：协议、样例与兼容适配

### 目标

冻结可实现的 canonical JSONL、Evidence Pack、版本登记和 API 契约。

### 任务

- 生成 clause.schema.json、standard.schema.json、clause_relation.schema.json、evidence_pack.schema.json。
- 固定 clause_id、parent_id、evidence_id、content_type、verification_status 和 document_status。
- 定义 POST /v1/retrieve、POST /v1/index/build、GET /v1/index/build/{build_id}、GET /health 的请求、响应、错误和降级结构。
- 为现有 clauses.jsonl 编写兼容适配器，保持 applicability_results、risk_register、evidence_verification 协议兼容。
- 准备至少 10 个评测问题和人工标注字段，建立 standard_registry 与 GB 50189 pending_manual_review 空登记。

### 交付与验收

Schema、样例和 API 契约测试通过；重复 ID、空来源和未知字段行为有失败用例；旧 LocalClauseRetriever 的字段适配测试通过。

## Day 2：Markdown 解析与质量门禁

### 目标

把两份格式不同的清洗 Markdown 转为 canonical JSONL，并能阻止不安全索引发布。

### 任务

- 解析 55015 的页区域继承注释。
- 解析 50378 的页级与节点级注释，节点级元数据优先，最近页区域补全。
- 统一条文、父节点、表格、公式、图和条文说明节点。
- 保留 PDF 物理页与原书印刷页，校验 source_sha256、asset 路径、表格续表、公式编号和 ID 唯一性。
- 检查 50378 的 appendices_detected 与实际附录扫描结果；不一致时阻断 manifest。
- 输出 canonical JSONL、审计报告、source manifest 和失败样例。

### 交付与验收

两份输入可由同一入口处理；30 条文、10 张表、全部公式和附录节点抽样可追溯；硬失败不生成可发布 manifest。

## Day 3：SQLite FTS5 与 BM25 基线 API

### 目标

完成可复现的中文 BM25 检索和基础 retrieve 接口。

### 任务

- 使用固定 jieba 版本构造 retrieval_text，保留标准号、条文号、单位和数值字段。
- 建立 SQLite FTS5 索引和索引 manifest，支持重建而不覆盖 canonical JSONL。
- 实现明确标准/条文/地区硬过滤，以及建筑类型/设计阶段/绿建目标软过滤。
- unknown 字段不得误杀候选；commentary 节点不得进入索引。
- 实现 /v1/retrieve 的 BM25-only 诊断模式、request_id、错误包和 trace。

### 交付与验收

条文号、表号、公式号和中文主题查询均有固定结果；BM25-only baseline 可重复运行；过滤、零结果、非法 top_k 和缺失日期有测试。

## Day 4：Embedding、Qdrant、RRF 与多样性

### 目标

把 dense 召回和 BM25 通过固定顺序合并成可解释的 hybrid 结果。

### 任务

- 校验本地 embedding 模型维度、归一化和设备，批量写入 Qdrant。
- 在 payload 中保留 clause_id、parent_id、standard_id、content_type、document_status、indexable、页码和 verification_status。
- BM25 与 dense 各召回 50 条，分别按 clause_id 精确去重。
- 按合并后的 1-based rank 计算 RRF(k=60)，记录原始 rank/score。
- 应用文本哈希、token Jaccard 和同父节点最多 3 条的多样性控制。
- 构建临时索引，成功后原子发布新 index_manifest_id，并记录去重和过滤日志。

### 交付与验收

BM25-only、dense-only、hybrid 三种结果可复现；相同 clause_id 只保留一个合并结果；同父节点的表格/公式不会因 parent_id 相同被误删；hybrid 不低于任一单路 baseline。

## Day 5：Reranker、Evidence Pack 与端到端验收

### 目标

完成完整 retrieve 闭环、引用完整性和评测报告。

### 任务

- 接入 bge-reranker-v2-m3，固定批大小、最大输入、超时和设备信息。
- reranker 超时默认失败关闭；只有 allow_degraded=true 才返回带 degraded_modes 的 RRF 结果。
- 补充父级上下文，构造 Evidence Pack 和 primary_normative/supporting_table_formula/version_relation 分组。
- 校验标准、版本、条文号、PDF 页码、印刷页码、表号/式号和 evidence_id。
- 运行 10 个评测案例，分别记录 Clause Hit@10、Citation Accuracy、Citation Completeness 和人工复核召回。
- 运行端到端 API、健康检查、构建状态和兼容回归测试，输出未解决问题清单。

### 交付与验收

完整 hybrid retrieve 可运行；Evidence Pack Schema 通过；Schema/追溯率 100%；Clause Hit@10 >= 0.85；Citation Accuracy >= 0.95；Citation Completeness >= 0.90；关键引用错误为 0。未满足指标时只能发布为实验 manifest，不能标记为基线版本。

## 每日通用要求

- 每天结束生成 run_id、变更摘要、测试结果和未解决问题。
- 任何数据、分词器、模型或过滤规则变化都生成新 manifest，不覆盖历史结果。
- 原始 PDF、Markdown、canonical JSONL、索引和日志分目录存放。
- 日志只记录 request/run/build/manifest、耗时、候选数和降级状态，不打印项目资料敏感正文。
- RAG 只输出 needs_manual_review、missing_facts 和 warnings；GreenSpec 负责人工任务、适用性、风险和报告流程。
