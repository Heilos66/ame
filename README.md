# AME · 联想记忆引擎（Associative Memory Engine）

> 一个以事件为核心、以"语义-实体-空间-时间"为锚点、由"向量层+图层"双骨架支撑、
> 通过"软→硬→软"循环实现联想检索与深度思考的**嵌入式记忆系统**（C++17，零第三方依赖）。

**特性清单**

- 双骨架：向量层（模糊语义）+ 图层（可解释关联），"软入口→硬扩散→软重排"三层管道；
- 实体档案：三级消歧、属性槽位 + SUPERSEDES 双时态知识更新、时间线视图；
- 写入即生长：共现/共同经历/NEAR 地理/NEXT 叙事链/组成结晶 自动建边；
- 深度思考：M8 能量扩散多轮 + IRCoT 式 LLM 引导迭代检索 双模式；
- 中文就绪：AC 自动机词典匹配（70 万词 µs 级）、词林/反义词库/高频蒸馏词库 68 万词条；
- 基准可评：LoCoMo / LongMemEval / BEAM 离线评测管线 + DeepSeek LLM 管线（可插拔）。

设计文档：[docs/design/联想记忆引擎设计文档.md](docs/design/联想记忆引擎设计文档.md)（设计冻结版，含架构图）、
[docs/engineering/联想记忆引擎_工程实现与优化文档.md](docs/engineering/联想记忆引擎_工程实现与优化文档.md)（实现与迭代记录）。
文档索引见 [docs/README.md](docs/README.md)。

独立 HTTP 知识检索服务见 [RAGFlow 兼容接口与 ECS 部署](services/ragflow/README.md)。
通过 `AME_BUILD_RAGFLOW_SERVICE=ON` 构建；支持 Bearer 鉴权、`/api/v1/retrieval`、
`/health`，接受并忽略不适用的 RAGFlow KNN、重排和相似度阈值参数。

---

## 一、架构总览

```
                        core（公共类型 / uid / 极简JSON / 日志 / RelType 系数）
                          │
                       storage (M9)
                       ┌──┴───┐
                    time(M10) vector(M13)
                       │      │
        keyword(M3) ───┘      └─── entity(M2)
           │                            │
        extractor(M1) ← time      graph_writer(M4) ← keyword/time/vector/extractor
           │                            │
           └──────→ seed(M5) ────────────┘
                        │
                    diffusion(M6) ★
                        │
                     rerank(M7) ← time/vector
                        │
                     engine(M12)  ← 聚合以上全部模块（六动词外观）
                        │
                  bench（基准适配）/ llm（DeepSeek 接入）
```

12 个模块 + core 公共层（3.0 已裁 M8 thinker / M11 feedback / M14 learning，
裁剪依据见文末"3.0 简化版"），每模块一个子目录、一个静态库，依赖用
`target_link_libraries` 显式声明（无环）。

---

## 二、完整设计要点

**双骨架与桥**（设计文档 §2）。图层是拓扑：节点、带类型的边、可解释路径；向量层是几何：
点、距离、方向。语义间有两种基本关系——构成关系（向量层内部方向分量）与关联关系
（图层带类型边）；两种关系通过"桥"互化：边权是向量几何在图上的快照（向量→图），
成分反复提及结晶为组成边、关联高频使用熔化进向量（图→向量）。运行时桥即"软→硬→软"：
软入口 ANN 接住任意表达，硬扩散沿类型边精确推理，软重排做语境校准。

**四类锚点与指称层**（§2.3、§3）。实体（Entity）、空间（Place/Region）、语义（Keyword）、
时间（Phase/时间链）挂在双骨架上而非第三层；Mention/Alias 是"语言世界→语义世界"的
唯一入口边，uid 与名字彻底分离。

**三层时间投影**（§4）。唯一事实源是 Event.timestamp（UTC）；地方时/饭点/打烊后由
Place.timezone/open_hours 投影，人生阶段由 Entity.LIFE_PHASE 投影，叙事顺序由 NEXT
时间链投影。推导时间永不存储（防时区/夏令时坑）。

**写入即生长**（§3.4）。每写一条事件，自动执行五项：同事项关键词两两共现+1、
同事项实体两两共同经历+1、新 Place 挂 IN（geohash 前缀）与 NEAR（≤500m）边、
推导标签（地方时/时段/人生阶段）、成分提及≥3 次结晶为组成边；长期未命中边衰减剪枝。

**软→硬→软 Recall 管道**（§6.2）。① 软入口（M5 种子生成：硬种子 uid/强标识/精确词
跳过向量通道，软种子 embed→ANN topK，词法 IDF 种子补语义弱项，坐标种子地理定位；
词典未加载端（Android）M1 提不出中文词时，CJK 通道退回查询文本的字符二元组
子串匹配兜底，≥2 个命中才收录）；② 硬扩散（M6 加权扩散，唯一的新信息产生环节）；③ 软重排（M7 融合打分：
`final = α·图扩散分 + β·语境余弦 + γ·时间衰减×importance`）。

---

## 三、主要算法思路

**加权扩散（M6）**。从种子出发做加权 BFS：边传导 = 类型系数（因果 0.9/推理 0.9/
组成 0.8/相似 0.7/共现 0.6/对立 0.5）× 边权（>5 高频边 log 软化），能量按 1/(hop+1)
逐跳衰减，深度硬限 2~3 跳，budget 限扩展节点数，每节点保留 top3 传播路径（可解释）。
PageRank 式出度归一防枢纽节点（主角实体）均匀喷灌淹没直接证据；休眠关键词不参与。

**融合打分（M7）**。`final = α·图扩散分 + β·语境余弦相似度 + γ·时间衰减×importance`
（默认 0.5/0.3/0.2，固定权重；3.0 已裁 M14 离线调优，bench 可用
`--final-alpha/--final-beta` 做对比实验）。图扩散分以事件节点能量做 max 归一（关键词枢纽
能量是多路径累加的，若用全局 max 会把直接证据压到 0）；filters 支持时间范围/实体
类型/人生阶段/场所时段；explain 生成中文路径说明。

~~Think 编排（M8，双模式）~~ **[3.0 已裁]**：能量模式与 IRCoT 多轮取证增益≤噪声带，
thinker 模块整体移除（归档证据见文末裁剪表）；读取链路为 seed→diffusion→rerank 单遍管道。

**束搜索（引导式扩散原语）**。`diffuse_beam`：best-first，逐跳只扩能量
top-m 前沿（默认 m=50），diffuse 接口不变。原语保留；监督边门控 `M(edge)` 的
训练器随 M14 在 3.0 裁撤（gate_fn 接口仍在，未训练不回归）。

**实体解析三级决策树（M2）**。① uid 直连（校验存在性）；② 强标识符精确匹配
（VIN/手机号，无则建档）；③ 裸名字 + 上下文证据投票（共同经历+0.3、空间一致+0.2、
唯一候选+0.1、基础 0.4）：>0.9 直连、0.5~0.9 标记待确认、<0.5 新建并标记疑似重复；
merge/split 随时纠错（旧 uid 跳转表，Mention 中间层使改指成本极低）。

**实体档案槽位 + SUPERSEDES 双时态（M2/M4）**。槽位四元组
(slot, value, valid_from, valid_to, source_event)：同实体同槽位写新值 → 旧值标
valid_to 并连 SUPERSEDES 边；槽位词有 df≤5 守卫（防相邻轮互相取代）。查询侧"新旧都在
选新"：仅当取代者也在候选集才丢旧证据（新值未召回时旧证据兜底）。
`lookup_as_of` / `slot_as_of` 支持历史时刻双时态查询。

**关键词管理四规则与三分流词典（M3）**。别名归一（别名表 O(1)）、停用词黑名单、
低频休眠与唤醒（出现<3 且无边不参与扩散）、受控主题表。词典三分流：syn_strong
（真同义/词形→别名归并）、syn_near（近义→保留双节点 REL_SIMILAR 边，>0.6 且 ≤8）、
antonyms（反义→REL_OPPOSITE 边，系数 0.5 最低档，防反义被当强联想）；canonical
组内首词为标准词，内置标准词不可改写。

**弃答门控**。证据最高分 + 语境相似度双闸门（阈值走配置）：无证据或低置信 →
NO INFORMATION；LLM 答题器 prompt 同语义兜底。实测对抗题弃答准确率 0.78~0.90。

**AC 自动机 + 持久化**。词典匹配 Aho-Corasick：Unicode 码点转移（防中文按字节错配）、
排序小数组+二分转移边；70 万词构建 1.3s / 内存 ~58MB / 单文本扫描 ~17µs
（较逐词查找快三个数量级）。二进制持久化：构建+写 0.16s，加载 179ms（vs 重建 1.36s），
magic+版本+词典 (size,mtime) 校验，不一致自动重建。

**证据打包（NEXT±1）与两阶段聚合作答**。top-k 证据事件带同 session 前后各一轮上下文，
缓解片段信息不足（judge +16pp 实测）。聚合类问题（how many/total/所有）可走两阶段：
k_pool 候选池（~50 条）→ 本地预过滤（池内 BM25-lite，零 LLM，`--agg-prefilter-n`
默认 20）→ 批量提取（`--agg-batch` 默认 5 条/次，编号 [E#] 输入、逐条输出事实/NONE，
保留思考）→ 事实列表聚合作答（`--answer-mode aggregate`，默认关）。LME 14 道聚合题
实测 judge 3/14 vs 逐条基线 0/14（3 题 0→1、零退化），answer 侧调用 4.8 次/题 vs
逐条 16.2 次/题（约 1/3.4）；瓶颈仍在提取召回（平均 3.2 条事实/题），未达转正线。

---

## 四、基准评测

三个长期记忆基准的离线评测管线（数据获取见各 README，数据文件不入库）。
**完整评测报告与失败分析见 [docs/reports/](docs/reports/README.md)**
（[LoCoMo 报告](docs/reports/locomo_report.md) / [LongMemEval 报告](docs/reports/longmemeval_report.md)）：

| 基准 | 数据获取 | 指标口径 |
|---|---|---|
| LoCoMo | github.com/snap-research/locomo（`data/locomo10.json`） | token F1、evidence recall@k（dia_id 命中）、judge、abstain |
| LongMemEval | github.com/xiaowu0162/LongMemEval（`longmemeval_s.json`） | recall@k（answer_session_ids 命中）、judge、abstention |
| BEAM | github.com/mohammadtavakoli78/BEAM | nugget 词重叠代理分（IJudge 可换真实 LLM）、event_ordering Kendall tau-b |

**历史指标快照（2.0 前最优管线：rule 摄入 + 实体档案 + 蒸馏词典 + 查询扩展 +
证据内容词终排 + NEXT±1 + think + k_pool；其中蒸馏词典/查询扩展/think 已在 3.0
裁撤，现行口径见文末"3.0 简化版"基线）**：

| 基准 | recall@10 | judge | 备注 |
|---|---|---|---|
| LoCoMo sample50（固定 50 题） | 0.500 | **0.500** | 起点 0.271/0.220；recall 0.542→0.500 微降、judge 0.440→0.500 |
| LoCoMo 全量 v3（1986 题） | 0.654（cat2） | 0.375 | **修复前口径**（未含日期解析/终排修复），待重跑刷新 |
| LongMemEval 全量 500 | 0.672 | 0.398 | **修复前口径**（未含偏好题修复），待重跑刷新 |
| LongMemEval 前 100 | **0.820** | **0.590** | single-session 0.757 / multi-session 0.200 |

**全量评测失败点修复（失败分析驱动，A/B 验证）**：

- **LoCoMo cat2 时序题（judge 0.072）**：双根因。① LoCoMo 会话日期格式
  "1:56 pm on 8 May, 2023" 未被 `parse_session_date` 识别（时分数字被误判为月/日），
  全部会话落入合成时间戳，LLM 拿到的证据日期全错；② think 证据并入选址带时近偏置，
  词面精确命中的金标被顶到队尾。修复：月名日期解析 + 证据内容词 BM25-lite 终排
  （recall 大候选池不提前截断，终排后才截 top-k）。A/B（12 题抽样）：judge 0/12 → 11/12。
- **LME single-session-preference（judge 0/30，recall 0.233）**：根因是实体通道第一人称
  映射按 INVOLVES 度数盲选主说话人——assistant（258）高于 user（250）导致错选助手实体，
  检索池错位遮蔽了标准管道的正确召回。修复：角色感知第一人称（优先 user 角色、排除
  assistant 角色）+ 通道弃权回退（弱通道结果不再遮蔽标准召回）+ 弃答门控对内容词
  精确命中豁免。A/B（30 题全量）：judge 0/30 → 8/30，recall 0.233 → 0.633，
  误弃答 23/30 → 14/30。

> ⚠️ **日期解析 bug 影响面（显著标注）**：该 bug 使 LoCoMo 全部会话时间戳落入合成值
> （2023-11-14 起按会话序递增），修复前所有涉及时序的口径均被污染——cat2 时序题
> （证据日期标签全错）、SUPERSEDES 新旧过滤、rerank 时间衰减项、证据日期前缀。
> 上表 v3 / LME 500 全量数字均为修复前口径；cat2（0/12→11/12）与 LME 偏好题
>（0/30→8/30）为小样本 A/B 口径，全量重跑后刷新。

### 已否证的假设

- **证据摘要整合（digest）**：scratchpad 式 LLM 摘要（纯摘要替代 / 摘要+原文双通道 /
  摘要仅供定位三版）在 LME multi-session 30 题上全部判负（judge 0.200 → 0.100 / 0.167）。
  根因：摘要改写丢失精确细节，并在计数聚合题诱发"精确但错误"的答案。功能已干净移除。
- **IRCoT 对聚合题**：子查询确能补证据（16/19 证据变化、2 题弃答转回答），但聚合类
  答案完整性无改善（judge 4/19 = 4/19 零翻转）——瓶颈在答题侧聚合，不在检索侧，
  IRCoT 保持默认关。
- **条件归一**：查询词无直接命中才词典扩展——sample50 上 0.521/0.380 不优于
  全局扩展+蒸馏词典（0.542/0.440），WordNet 在门控下零新增命中。保持默认关。
- **WordNet/gap 大词典默认灌库**：宽泛同义归一稀释精度（judge -6pp），保留文件待用。
- **两阶段聚合作答**：逐条版 1/5 vs 0/5 微正；预过滤(N=20)+批量(5条/次)降本组合
  3/14 vs 0/14（3 题 0→1、零退化，answer 侧调用降至 ~1/3.4，缓存重放第二遍 0 新调用），
  仍未达转正线（阶段一提取召回不足，平均仅 3.2 条事实/题），默认关。

---

## 五、编译说明

零第三方依赖（无 SQLite/RocksDB/hnswlib/任何第三方库）：持久化用自写极简 JSON 快照；
向量检索用暴力余弦 topK（封装在 ANN 接口后，可换 hnswlib）；LLM 调用经 curl 子进程
（可注入替换 libcurl）。

```bash
# WSL (Ubuntu-22.04) / Linux：
sudo apt install g++ cmake ninja-build   # g++ ≥ 11
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Windows 侧源码挂载在 `/mnt/d/agent/memory/ame`（WSL 访问 /mnt 盘构建即可，勿用
Windows MinGW——工程使用 POSIX 接口 gmtime_r/popen 等）。

---

## 六、使用说明

```bash
./build/demo/ame_demo                 # 中文演示：remember→lookup→recall(explain)→快照
cd build && ctest --output-on-failure # 模块单测 + e2e + bench + llm

# 基准评测（以 LoCoMo 为例，数据自备）
./build/bench/ame_bench --dataset locomo --data bench/data/locomo10.json \
  --answer llm --judge llm --concurrency 16 --cache-dir bench/cache \
  --rerank cheap --cheap-rerank bench/models/cheap_model.json \
  --out report.json --dump results.ndjson
# 常用参数：--limit N --k 10 --abstain-threshold 0.15
#   --extractor rule|llm --answer baseline|llm --judge f1|llm
#   --rerank bm25|cheap|llm（默认 bm25；cheap 零 LLM 特征重排，llm 全池语义打分）
#   --rerank-guard 0.2 --entity-pass-threshold 0.5 --final-alpha 0.5 --final-beta 0.3
#   --save-graph P / --load-graph P（摄入快照，零提取重放）
#   --question-ids ids.json（固定样本）--name-alias P（专名归一表）
#   --entity-profile on|off --conditional-normalize on|off --answer-mode aggregate
#   --agg-prefilter-n 20（聚合预过滤保留条数）--agg-batch 5（批量提取每次条数）
```

**DeepSeek 配置**（只此两种方式，key 不入库不入日志）：

- 环境变量 `AME_DEEPSEEK_API_KEY`（优先）；或
- `bench/config/deepseek.local.json`（`{"api_key": "..."}`，已被 .gitignore 排除）。
- 模型：`deepseek-v4-flash`（批量）/ `deepseek-v4-pro`；推理模型 max_tokens 含思考预算，
  提取类任务默认关思考（`thinking:{type:disabled}`），答题保留思考。

**词库工具**：见 [lexicon/README.md](lexicon/README.md)（资产总表/schema/许可/加载/蒸馏复跑命令）。

---

## 七、目录结构

```
core/ storage/ time/ vector/ keyword/ extractor/ entity/ graph_writer/
seed/ diffusion/ rerank/ engine/                       # 12 模块 + core（3.0 裁 M8/M11/M14）
bench/        # 基准适配层（loader/ingest/answer/metrics/CLI/fixtures/scripts）
llm/          # DeepSeek 接入（客户端/提取器/答题器/评审器/词典蒸馏）
lexicon/      # 词库资产（68万+ 词条，含 README 与 lexicon_src 整理中间层）
tests/        # 模块单测 + e2e + bench + llm（ctest 注册）
demo/         # 中文演示程序
docs/         # 文档（design/ 设计文档+架构图，engineering/ 工程实现与优化文档，reports/ 评测报告）
```

### 答题链路的模块联动（QA pipeline 全模块角色表）

| 模块 | 在答题链路的角色 |
|---|---|
| core | uid/类型/JSON/传导系数，全链路数据底座 |
| M9 storage | 邻接表/词法候选扫描/图快照（--save-graph/--load-graph） |
| M10 time | 会话日期解析、question_date 相对时间锚定、证据时间衰减、有效期过滤 |
| M13 vector | 哈希 embedding 软种子与语境分（弱语义兜底） |
| M1 extractor | 规则提取（摄入/查询侧关键词）/ LlmExtractor 会话批提取（实体+关键词+属性三元组） |
| M2 entity | 实体档案：消歧、槽位（知识更新）、时间线；查询侧人名锚定 |
| M3 keyword | 别名归一（--name-alias 专名归一表）、低频休眠（3.0 已裁词典灌库/RELATED 边召回通道） |
| M4 graph_writer | 写入即生长：共现/共历/NEAR/NEXT 链/SUPERSEDES |
| M5 seed | 硬/软/词法（IDF)/坐标种子，实体锚点硬种子 |
| M6 diffusion | 加权扩散（出度归一防枢纽淹没），recall 的推理原语 |
| M7 rerank | α/β/γ 融合（事件级归一，固定权重）、filters、explain；bench 终排三档 --rerank bm25|cheap|llm |
| M12 engine | 六动词外观 + remember_turn/get_entity_profile/组合快照，全模块装配 |
| bench/llm | 加载/摄入/答题/判分/词典蒸馏/缓存（接入层工具链） |

---

## 八、路线图与已知限制

**路线图**：① 聚合类问题两阶段答题（阶段一提取召回增强：k_pool 全量逐项）；
② 语料自训练词向量接入 IEmbedder；③ LongMemEval-S 全量 500 题（发布期全量评测）；
④ BEAM 真实数据适配。引导式扩散 RL 边门控与每用户 LoRA 属 M14 路线，已随 3.0 裁剪冻结。

**已知限制（相对设计文档）**

- **暴力 ANN**：`Storage::ann_search` 为全表余弦 topK，接口后可替换 hnswlib；
  `geo_lookup` 为暴力 haversine，可替换 GeoHash/R-tree。
- **哈希 embedding**：M13 用字符 n-gram feature hashing（确定性、支持中文 UTF-8），
  非语义模型——词面基线相似度使弃答门控对"词面相关但信息缺失"的对抗题不敏感；
  `IEmbedder` 可插拔换 BGE 等真模型。
- **规则提取器**：M1 为词典 + 启发式（非 LLM）；`IExtractor` 可插拔。
- **时区**：固定小时偏移（`Place.attrs["tz_offset"]`），无 tz 数据库/夏令时。
- **SUPERSEDES 槽位启发式**：同实体同槽位词（df≤5）即触发，语义偏宽（过度标记已收敛
  但非精确冲突检测）。
- 快照为单文件 JSON；批量导入的分批事务在内存图下退化为接口语义。
- AC 自动机结构未进图快照（重载词典文件或 AC cache 重建，179ms）。

---

## 3.0 简化版（dev-3.0，七刀裁剪）

在 2.0 的实验谱系收敛后，3.0 按 **docs/reports/ 已归档的证伪/停用清单** 裁剪，
只保留有实证红利的路径（保留项 bench 数字回归实测见下表；裁剪依据均可溯到报告）。

**保留**（实证资产）：1.x 图引擎六动词管道（记忆/联想/实体/纠错/档案/快照）、
`--rerank bm25|cheap|llm` 三档 + guard/entity-pass 横切、cheap 特征学习重排
（cheap_model_v2，11 特征）、LLM 全池语义打分 + 作答三指令（judge 0.63→0.73）、
name-alias 专名归一（names_zh 6/6）、AC 自动机中文通道、快照 save/load。

**裁剪**（七刀，均按归档证据）：

| 刀 | 内容 | 依据（docs/reports/） |
|---|---|---|
| 1 | 2.0 分层存储 M15–M19（memobs/memfact/memindex/memview/mem）+ lmem 后端 + JNI/Java 双后端 | lmem P0-P3 系列报告，2.0 管线全链未超 1.x graph |
| 2 | 词库召回系：graded 边表（WordEdgeTable）/同义近义反义灌库/extractor 大词表（12 词元贡献≈0；RELATED 边端到端≈0） | industry_baseline / 各轮消融 |
| 3 | M11 feedback + M14 learning（含 beam/gate 耦合） | 反馈闭环报告：增益在噪声带内 |
| 4 | thinker（M8 能量扩散 + IRCoT） | think 通道增益≤噪声带 |
| 5 | 实验重排档 formula/svec/xenc + 历史别名（--llm-rerank/--llm-rerank-all 等） | formula_rerank/svec_eval/xenc_spike/wmd_spike 四份证伪归档 |
| 6 | 实验脚本移 archive_retired/（17 个，不删） | 报告复跑性 |
| 7 | README 本节 + versionCode 30/versionName "3.0-dev" | — |

**裁剪后基线**（sample50 冻结配置 entity-pass 0.5 + guard 0.2）：
bm25 ≈0.46 / cheap ≈0.557 / llm ≈0.65-0.67（judge 0.73）——与裁剪前冻结证据逐位一致。
已裁脚本/模块的历史报告数字不受影响（产物固化于 bench/runs/，脚本留存 archive_retired/）。

**回归复跑记录**（sample50 同冻结配置，`--answer llm --judge llm` 全旗帜）：

| 日期 | 提交 | bm25（覆盖率/judge） | cheap | llm | ctest | 结论 |
|---|---|---|---|---|---|---|
| 2026-10-26 | 9a04b77（APK cheap 默认档+文档同步） | 0.521 / 0.68 | 0.604 / 0.74 | 0.750 / 0.72 | 14/15（test_bench=预存环境 UB，同基线） | 档位序 bm25<cheap<llm 保持，均不低于归档基线；绝对值上移为 uid/LLM 复跑噪声，本次改动不触 bench 路径 |

复跑命令：`--dataset locomo --data bench/data/locomo10.json --question-ids
bench/data/locomo_sample50.json --rerank <档> --entity-pass-threshold 0.5
--rerank-guard 0.2 --answer llm --judge llm --concurrency 16 --cache-dir bench/cache`
（cheap 档加 `--cheap-rerank bench/models/cheap_model.json`）。
APK 验证：assembleDebug 通过，`libame_jni.so` 导出 `nativeLoadCheapModel`、
APK 含 `assets/cheap_model.json`；真机 logcat 运行时确认待做。

**APK 重排档（3.0 起）**：默认 **cheap**——`assets/cheap_model.json` 建 session 时注入
JNI（`AmeEngine.loadCheapModel`），recall 在原生侧走 候选池60 → 11 特征逻辑重排 → topN，
零 LLM、毫秒级；加载失败自动回退引擎原序（bm25 行为）。手动"深度重排"开关不变，
开启时由 Java 侧 M3 全池语义打分（llm 档）。文档已同步：设计文档被裁节就地标注
"[3.0 已裁]"，2.0 分层架构稿标注裁撤归档，工程文档/目录树/CLI 参数均为现行口径。
