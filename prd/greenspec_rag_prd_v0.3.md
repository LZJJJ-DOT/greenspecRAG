# GreenSpec RAG 轻量侧车 PRD

> 版本：v0.3  
> 日期：2026-08-25  
> 基于：greenspec_rag_prd.md v0.2  
> 修订目的：补齐数据契约、检索接口、质量门禁、运行前提、版本治理和验收标准；保留原 v0.2 不变。

## 1. 项目定义与范围

greenspec-rag 是 GreenSpec 的本地规范证据检索侧车。它解析规范资料，建立 SQLite FTS5 BM25 与 Qdrant dense 索引，执行混合召回、RRF、去重和 reranker 精排，并返回可审计的 Evidence Pack。

它不负责最终合规结论、自动版本替代裁决、项目适用性最终判断、风险登记、人工复核流程状态或报告发布。RAG 只返回结构化的 needs_manual_review、missing_facts 和 warnings；GreenSpec 负责业务流程编排。

首批活动标准为 GB/T 50378-2019、GB 55015-2021；项目范围为北京普通办公楼/公共建筑的方案深化和初步设计，目标为二星级预审和三星级差距分析。GB 50189-2015 只做引用和版本治理登记，不抽取独立条文；关系未确认前固定为 pending_manual_review。

首版不做最终合规结论、全国地方政策自动适配、能耗/采光/声环境/碳排放计算、自动废止或替代裁决以及 UI 工作台。

## 2. 固定设计决策

- canonical JSONL 是唯一事实源，索引和缓存必须可重建。
- BM25 固定使用 SQLite FTS5；中文分词首版固定 jieba 0.42.1，版本写入 index manifest。
- 向量库固定使用 Qdrant；Embedding 使用本地 bge-base-zh-v1.5；reranker 使用本地 bge-reranker-v2-m3。
- BM25 与 dense 默认各召回 50 条；RRF 常数为 60，rank 从 1 开始。
- clause_id 是所有 canonical 节点的稳定主键，parent_id 指向父节点；不再使用未定义的 child_id。
- evidence_id 首版直接采用稳定的 clause_id；retrieval_run_id 标识一次返回。
- retrieve 默认返回规范原文 text、父级上下文和引用元数据。
- 无法确认 OCR、版本关系、适用条件或关键数值时，不静默猜测。

## 3. 模块边界

| 模块 | 负责 | 不负责 |
|---|---|---|
| ingestion/normalization | Markdown/PDF/资源读取、字段统一、来源哈希 | 修改原文、猜测 OCR |
| chunking/indexing | 父子节点、FTS5、Qdrant、manifest | 判断条文适用性 |
| retrieval | 查询解析、过滤、BM25、dense、RRF、去重、reranker | 最终结论 |
| governance | 标准登记、版本关系、状态和证据等级 | 用相似度宣布废止 |
| evidence | Evidence Pack、引用校验、人工复核标记 | 创建 GreenSpec 任务 |
| GreenSpec | 项目画像、适用性、风险、人工复核和报告 | 规范解析和检索排序 |

## 4. Canonical JSONL 数据契约

每行一个节点，UTF-8 编码，按 clause_id 确定性排序。必须提供 clause.schema.json，Schema 失败时索引构建中止。

节点必填或稳定字段：

- schema_version：当前 clause.v1。
- clause_id：全局唯一稳定主键；parent_id：父节点 ID，根节点为 null。
- document_id、standard_id、standard_name、standard_version、source_file、source_sha256。
- document_status：current、superseded、partially_superseded、pending_manual_review。
- content_type、clause_type、clause_no、clause_title、hierarchy。
- text：规范原文；retrieval_text：标题、编号、主题、适用条件和原文拼接文本。
- region、building_type、design_phase、green_target_scope。
- requires_project_facts、requires_calculation、requires_manual_review。
- pdf_page_start/end、printed_page_start/end；印刷页为空时不能用物理页冒充。
- source_level：T0、T0_CANDIDATE、T1、T2、T3、PROJECT。
- verification_status：verified、needs_manual_review、source_unverified。
- indexable、table_id、formula_id、asset_ids、supersession_ids、provenance。

content_type 枚举固定为：publication_info、reference_standards、toc、normative_front_matter、normative_body、normative_clause、normative_table、normative_formula、appendix、appendix_clause、commentary_front_matter、commentary、commentary_clause、commentary_table、commentary_formula、commentary_appendix、figure、non_content。commentary 节点首版保留归档但 indexable=false。

父节点为章、节、附录或条文说明章节；子节点为完整条文、款项、注、表格、公式或图节点。表格续表共享 table_id；公式必须保留公式、式中变量和所属条文。子节点不得跨越无关条文。

旧 clauses.jsonl 通过适配器转换：source_page 到 pdf_page_start/end，clause_text 到 text，clause_summary 仅作辅助字段，channel=local_clause 到 normative_clause。缺失父节点、印刷页或哈希保留 null 并标记人工复核。

## 5. 数据质量门禁和索引发布

每次构建必须生成 extraction_run_id、source_manifest_id、index_manifest_id。

硬失败条件包括：JSON Schema 失败、ID 重复、标准号/来源/哈希为空、物理页不连续、节点页码无法追溯、表格续表无唯一主表、公式编号重复、图片 asset/path 不存在、indexable 节点缺少 content_type 或有效来源，以及 50378 头部 appendices_detected=false 与实际附录节点不一致。

印刷页为空、OCR/来源未验证、版本关系待确认属于警告。警告候选可返回，但不能作为已验证权威证据支持确定性回答。

索引写入临时目录和新 manifest；全部校验、向量化和 BM25 完成后原子切换活动 manifest；失败不得覆盖当前索引。

## 6. 检索顺序和参数

原始查询 -> 保守解析 -> 明确条件硬过滤 -> FTS5 BM25 Top 50 -> Qdrant dense Top 50 -> 两路分别按 clause_id 精确去重 -> 合并并计算 RRF(k=60) -> 近重复/同父节点多样性控制 -> reranker -> 父级上下文 -> Evidence Pack。

硬过滤仅适用于 indexable=true、明确指定的标准号/条文号/日期/地区以及 include_commentary=false。建筑类型、设计阶段、绿建目标和未知适用条件为软过滤；未知值不得直接过滤候选。

近重复控制使用规范化文本哈希和 token Jaccard；同一父节点默认最多保留 3 个候选，但条文、表格、公式等不同 content_type 可各保留一个。trace 保留过滤原因、原始 rank/score 和最终 score。

Day 0 必须检查 Qdrant 地址/collection/持久化目录、两套模型权重、向量维度/归一化/最大长度/设备、批大小/超时以及 SQLite FTS5 和分词器版本。模型和 Qdrant 资源由部署环境提供，缺失时只能跑协议或 mock 测试，不能宣称完整混合检索完成。

## 7. API 契约

### POST /v1/retrieve

请求字段：query、project_profile、filters、top_k、allow_degraded。filters 包含 standard_ids、clause_nos、as_of_date、must_be_current、include_commentary。解析失败字段为 null；must_be_current=true 且没有 as_of_date 时返回参数错误。

响应必须包含 request_id、retrieval_run_id、index_manifest_id、items、filters_applied、warnings、degraded_modes。每个 item 至少包含 evidence_id、clause_id、parent_id、content_type、原文 text、父级上下文、标准/版本/条文号、PDF/印刷页码、source_file、source_sha256、verification_status、missing_facts、bm25_rank、dense_rank、rrf_score、rerank_score；缺失分数使用 null。

### POST /v1/index/build

请求包含 source_manifest_id、rebuild、dry_run，返回 202 和 build_id。新增 GET /v1/index/build/{build_id} 查询状态、失败原因和最终 index_manifest_id；仅完整成功的构建可成为活动索引。

### GET /health

返回服务版本、活动 manifest、SQLite、Qdrant、Embedding、reranker 状态；依赖未就绪返回 503，并区分 live、ready 和具体依赖错误。

错误统一为 error.code、message、request_id、retryable、details。Qdrant、Embedding 或 reranker 不可用时默认失败关闭，只有 allow_degraded=true 才返回带 degraded_modes 的 BM25/RRF 降级结果。

## 8. Evidence Pack、版本治理和人工复核

Evidence Pack 固定包含 evidence_pack_id、retrieval_run_id、index_manifest_id、原始 query、items、证据分组、citations、warnings、needs_manual_review、missing_facts。

证据分组为 primary_normative、supporting_table_formula、version_relation、project_facts、policy。条文说明只归档，不进入正式检索和规范性回答。每个规范性 claim 必须绑定 evidence_id；表格/公式引用必须带 table_id 或 formula_id；引用至少包含标准编号、名称、版本、条文号和 PDF 页码。

standard_registry 至少记录编号、名称、版本、发布日期、实施日期、状态、区域、权威等级、来源等级和关系核验状态。clause_relations 至少记录 relation_id、from/to 节点、关系类型、范围、状态、生效日期、证据引用和复核人；状态为 verified、pending_manual_review 或 rejected。GB 50189 关系初始为 pending_manual_review，不得自动过滤、宣布整本废止或生成确定性替代结论。

## 9. 评测、测试和验收

首批评测集至少 10 条，覆盖条文号、表格临界值、公式/单位、建筑类型/阶段过滤、正文与条文说明区分、跨页表格、版本日期边界、无证据、缺少项目事实和待人工确认关系。

首版门槛：Schema 通过率 100%；可索引节点标准/来源/页码可追溯率 100%；Clause Hit@10 >= 0.85；Citation Accuracy >= 0.95；Citation Completeness >= 0.90；关键引用错误为 0；hybrid 命中率不低于 BM25-only 和 dense-only 基线。

测试必须覆盖 Schema/解析、注释继承、页码、表格/公式、FTS5 分词、未知字段过滤、RRF 顺序、Qdrant payload、reranker 超时、引用完整性、版本关系和端到端 API。数据、分词器、模型或过滤规则变化都生成新 run_id。

## 10. 运行安全和兼容

API 不接受任意文件路径；原始资料、Markdown、JSONL 和索引分目录存放；日志记录 request/run/build/manifest、耗时、候选数和降级状态，不打印项目敏感正文；对 query、top_k、批大小、并发和模型超时设上限。

保留现有 applicability_results、risk_register、evidence_verification 协议。现有 LocalClauseRetriever 作为 KeywordRetriever 适配器保留，新的 ClauseRetriever 统一 BM25/dense/hybrid 返回结构；GreenSpec 继续拥有适用性、风险、人工复核和报告发布权。

## 11. Definition of Done

- 两份首批 Markdown 通过同一套 canonical Schema 和质量门禁。
- 任意返回 item 可追溯到标准、版本、条文号、PDF 页码、原文和来源哈希。
- 表格、公式、父级上下文完整；条文说明不进入正式检索。
- BM25-only、dense-only、hybrid 和 reranker 可复现，排名和过滤原因可解释。
- Qdrant/BGE 缺失时不会伪装成完整混合检索成功。
- 无证据、版本未知或项目事实缺失时返回结构化人工复核信息。
- 评测集、manifest、模型配置和指标报告可重复运行。
- 现有 GreenSpec 三类协议及兼容测试继续通过。
