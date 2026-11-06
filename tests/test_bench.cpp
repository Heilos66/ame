/*
 * bench 基准适配层测试：夹具加载（三格式）、摄入后图规模、
 * 指标已知答案断言（token F1 / nugget 代理分 / Kendall tau-b）、
 * 弃答门控、知识更新（SUPERSEDES + current/as_of）、事件排序 order_events。
 */
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "ame/bench/bench.h"
#include "ame/bench/metrics.h"
#include "ame/bench/store.h"
#include "ame/engine/engine.h"
#include "ame/llm/llm.h"
#include "test_util.hpp"

using namespace ame;

static const char* kFixtureDir = AME_FIXTURE_DIR;  // CMake 注入

static void test_load_fixtures() {
  auto loco = load_locomo(std::string(kFixtureDir) + "/locomo_mini.json");
  CHECK(loco.size() == 1);
  CHECK(loco[0].sessions.size() == 3);
  CHECK(loco[0].sessions[0].turns.size() == 3);
  CHECK(loco[0].sessions[0].turns[0].dia_id == "D1:1");
  CHECK(loco[0].questions.size() == 5);
  CHECK(loco[0].questions[0].category == 1);  // 不透明整数原样保留
  CHECK(loco[0].questions[1].evidence_dia_ids.size() == 2);

  auto lme = load_longmemeval(std::string(kFixtureDir) + "/longmemeval_mini.json");
  CHECK(lme.size() == 3);
  CHECK(lme[0].sessions.size() == 2);
  CHECK(lme[0].sessions[0].id == "s1");
  CHECK(lme[0].questions[0].answer_session_ids.size() == 1);
  CHECK(lme[2].questions[0].unanswerable);  // abstention 题

  auto beam = load_beam(std::string(kFixtureDir) + "/beam_mini.json");
  CHECK(beam.size() == 1);
  CHECK(beam[0].sessions.size() == 2);
  CHECK(beam[0].questions.size() == 4);
  CHECK(beam[0].questions[0].nuggets.size() == 3);
  CHECK(beam[0].questions[3].unanswerable);
}

static void test_ingest_graph() {
  auto loco = load_locomo(std::string(kFixtureDir) + "/locomo_mini.json");
  Engine e;
  IngestStats st = ingest(e, loco[0]);
  CHECK(st.events == 9);                    // 9 轮 → 9 事件
  CHECK(st.entities >= 2);                  // Alice/Bob 至少两个实体
  // 同一 speaker 跨轮复用同一 uid
  const Node* alice = e.storage().find_by_name(NodeKind::Entity, "Alice");
  CHECK(alice != nullptr);
  int alice_events = 0;
  for (auto& [ed, n] : e.storage().neighbors(alice->uid, {RelType::INVOLVES}))
    if (n->kind == NodeKind::Event) ++alice_events;
  CHECK(alice_events == 6);                 // Alice 说了 6 轮
  // NEXT 链：9 事件 → 8 条 NEXT 边
  int next_edges = 0;
  for (auto* n : e.storage().nodes_of_kind(NodeKind::Event))
    next_edges += (int)e.storage().neighbors(n->uid, {RelType::NEXT}).size();
  CHECK(next_edges >= 8);
  // dia_id 落属性
  bool has_dia = false;
  for (auto* n : e.storage().nodes_of_kind(NodeKind::Event))
    if (n->attrs.count("dia_id") && n->attrs.at("dia_id") == "D2:3") has_dia = true;
  CHECK(has_dia);
}

static void test_metrics_known_values() {
  // token F1：pred {the,cat} gold {the,cat,sat} → 2*2/(2+3)=0.8
  CHECK_NEAR(token_f1("the cat", "the cat sat"), 0.8, 1e-9);
  CHECK_NEAR(token_f1("", "anything"), 0.0, 1e-9);
  // CJK 单字成词
  CHECK(tokenize("失眠").size() == 2);

  // nugget 代理分：nugget1 全命中(F1=0.6→1)，nugget2 零命中(→0)，均值 0.5
  double ns = nugget_proxy_score("booked a flight and reserved a hotel",
                                 {"booked a flight", "bought insurance"});
  CHECK_NEAR(ns, 0.5, 1e-9);

  // Kendall tau-b：参考序 [flight, hotel, insurance]，回答中 hotel 先于 flight
  double tau = order_tau_b("reserved a hotel then booked a flight and bought travel insurance",
                           {"booked a flight", "reserved a hotel", "bought travel insurance"});
  CHECK_NEAR(tau, 1.0 / 3.0, 1e-9);
  // 完全正序 → 1
  CHECK_NEAR(order_tau_b("booked a flight, reserved a hotel, bought travel insurance",
                         {"booked a flight", "reserved a hotel", "bought travel insurance"}),
             1.0, 1e-9);
  // 完全逆序 → -1
  CHECK_NEAR(order_tau_b("bought travel insurance, reserved a hotel, booked a flight",
                         {"booked a flight", "reserved a hotel", "bought travel insurance"}),
             -1.0, 1e-9);
}

static void test_abstention_gate() {
  Engine e;
  e.remember_turn("Alice", "I adopted a cat named Mochi.", 1700000000, "D1:1", "s1");
  BenchConfig cfg;
  cfg.abstain_threshold = 0.15;
  // 相关提问：证据分高 → 不弃答
  BenchQuestion q1;
  q1.question = "I adopted a cat named Mochi.";  // 与事件完全同文 → 软种子 sim≈1
  auto r1 = answer(e, q1, cfg);
  CHECK(!r1.abstained);
  CHECK(r1.score > cfg.abstain_threshold);
  CHECK(!r1.evidence_dia_ids.empty());
  CHECK(std::find(r1.evidence_dia_ids.begin(), r1.evidence_dia_ids.end(), "D1:1") !=
        r1.evidence_dia_ids.end());
  // 无关提问：相似度极低 → 证据分低于高门控阈值 → 弃答
  BenchConfig cfg2;
  cfg2.abstain_threshold = 0.9;
  BenchQuestion q2;
  q2.question = "zzqq xwvu";
  auto r2 = answer(e, q2, cfg2);
  CHECK(r2.abstained);
  CHECK(r2.answer_text == "unanswerable");
}

static void test_knowledge_update() {
  Engine e;
  auto r1 = e.remember_turn("Alice", "I moved to Seattle for work.", 1700000000, "D1:1", "s1");
  auto r2 = e.remember_turn("Alice", "I moved to Portland last month.", 1700000000 + 86400,
                            "D2:1", "s2");
  // 共享槽位词 "moved"(attr) + 同一实体 Alice → 旧事件被取代
  const Node* old_ev = e.storage().get_node(r1.event_id);
  CHECK(old_ev && old_ev->attrs.count("valid_to"));
  CHECK(e.storage().find_edge(r2.event_id, r1.event_id, RelType::SUPERSEDES) != nullptr);

  // 当前视角：只剩 Portland 事件
  LookupQuery lq;
  lq.keywords = {"moved"};
  lq.current_only = true;
  auto cur = e.lookup(lq);
  CHECK(cur.size() == 1);
  CHECK(cur[0]->uid == r2.event_id);

  // as_of 历史视角：t1 之后 t2 之前 → Seattle 事件
  auto hist = e.lookup_as_of(lq, 1700000000 + 3600);
  CHECK(hist.size() == 1);
  CHECK(hist[0]->uid == r1.event_id);
}

static void test_order_events() {
  Engine e;
  auto r1 = e.remember_turn("Alice", "First I booked a flight.", 1700000000, "", "s1");
  auto r2 = e.remember_turn("Alice", "Then I reserved a hotel.", 1700000060, "", "s1");
  auto r3 = e.remember_turn("Alice", "Finally I bought insurance.", 1700000120, "", "s1");
  // 打乱输入顺序，输出应为时间升序
  auto ordered = e.order_events({r3.event_id, r1.event_id, r2.event_id});
  CHECK(ordered.size() == 3);
  CHECK(ordered[0] == r1.event_id && ordered[1] == r2.event_id && ordered[2] == r3.event_id);
}

static void test_snapshot_with_slots() {
  // 组合快照：图 + 实体档案槽位（含 valid_to 历史）往返一致
  Engine e;
  auto r1 = e.remember_turn("Alice", "I moved to Seattle for work.", 1700000000, "D1:1", "s1");
  auto r2 = e.remember_turn("Alice", "I moved to Portland last month.", 1700000000 + 86400,
                            "D2:1", "s2");
  std::string alice = r1.resolved_entities[0].second;
  e.entities().put_slot(alice, "residence", "Seattle", 1700000000, r1.event_id);
  e.entities().put_slot(alice, "residence", "Portland", 1700000000 + 86400, r2.event_id);

  std::string path = "/tmp/ame_test_combined_snapshot.json";
  CHECK(e.save(path) == Err::Ok);
  Engine e2;
  CHECK(e2.load(path) == Err::Ok);
  CHECK(e2.storage().node_count() == e.storage().node_count());
  CHECK(e2.storage().edge_count() == e.storage().edge_count());
  auto p = e2.get_entity_profile(alice);
  CHECK(p.current_slot("residence") == "Portland");
  CHECK(p.slots.size() == 2);  // 历史值 Seattle 保留
  CHECK(e2.entities().slot_as_of(alice, "residence", 1700000000 + 3600) == "Seattle");
  // recall 一致性
  SeedQuery sq;
  sq.text = "moved";
  auto a = e.recall(sq, {}, 2, 10, false);
  e2.reranker().set_weights(e.reranker().weights());
  auto b = e2.recall(sq, {}, 2, 10, false);
  CHECK(a.events.size() == b.events.size());
  for (size_t i = 0; i < a.events.size(); ++i)
    CHECK(a.events[i].event->uid == b.events[i].event->uid);
}


// 会话日期解析：LoCoMo 月名格式（"1:56 pm on 8 May, 2023"）+ 数字格式回归
//（月名缺失时时分数字曾被误判为月/日 → 全部会话落入合成时间，cat2 时序题全灭的根因）。
static void test_parse_session_date() {
  auto expect = [](int y, int m, int d, int hh, int mm) {
    std::tm tm{};
    tm.tm_year = y - 1900;
    tm.tm_mon = m - 1;
    tm.tm_mday = d;
    tm.tm_hour = hh;
    tm.tm_min = mm;
    return (TimePoint)timegm(&tm);
  };
  // LoCoMo 月名格式：D Mon, Y + 12 小时制时刻
  CHECK(parse_session_date("1:56 pm on 8 May, 2023", 0) == expect(2023, 5, 8, 13, 56));
  CHECK(parse_session_date("1:14 pm on 25 May, 2023", 0) == expect(2023, 5, 25, 13, 14));
  CHECK(parse_session_date("10:02 am on 2 July, 2023", 0) == expect(2023, 7, 2, 10, 2));
  // Mon D, Y 变体（无时刻默认正午）
  CHECK(parse_session_date("May 8, 2023", 0) == expect(2023, 5, 8, 12, 0));
  // 数字格式回归：Y-M-D 与 M/D/Y
  CHECK(parse_session_date("2023/05/30 (Tue) 23:40", 0) == expect(2023, 5, 30, 23, 40));
  CHECK(parse_session_date("5/25/2023", 0) == expect(2023, 5, 25, 12, 0));
  // 无法解析 → fallback
  CHECK(parse_session_date("no date here", 42) == 42);
}

// 证据终排（时序题）：think 并入的近期客套证据不得盖过词面精确命中的旧金标事件
//（locomo_v3 cat2 实测失败模式："charity race" 金标被时近偏置顶到第 9 位）。
static void test_evidence_content_rerank() {
  Engine e;
  const TimePoint old_ts = 1700000000;
  const TimePoint new_ts = old_ts + 180 * 86400;  // 180 天后（时近偏置拉满）
  e.remember_turn("Melanie", "I ran a charity race for mental health last Saturday.",
                  old_ts, "D2:1", "s2");
  for (int i = 0; i < 5; ++i)  // 近期客套轮：同实体、零内容词重叠
    e.remember_turn("Melanie", "Thanks so much, your words really mean a lot to me.",
                    new_ts + i * 60, "D9:" + std::to_string(i + 1), "s9");
  BenchConfig cfg;  // 默认 entity-profile / think 均开（时序题触发 think 并入）
  BenchQuestion q;
  q.question = "When did Melanie run a charity race?";
  auto r = answer(e, q, cfg);
  CHECK(!r.evidence_uids.empty());
  const Node* front = e.storage().get_node(r.evidence_uids.front());
  CHECK(front != nullptr);
  CHECK(front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
}

// 人名/专名归一域（--name-alias）：姓名别名表三通道贯通验收——
// M3 别名归一（守卫原样）、M1 词面注册（非称呼变体可提取）、M2 实体建档 +
// 称呼轮（老X/小X）摄入归并到 canonical 实体（INVOLVES）与 canonical 关键词（CONTAINS）。
static void test_apply_name_aliases() {
  const char* path = "/tmp/ame_test_names_zh.json";
  {
    std::ofstream f(path);
    f << "{\"source\":\"names_zh\",\"groups\":["
      << "{\"canonical\":\"李建国\",\"aliases\":[\"小李\",\"李经理\",\"建国\"]}]}";
  }
  Engine e;
  int applied = 0;
  CHECK(apply_name_aliases(e, path, &applied) == Err::Ok);
  CHECK(applied == 4);  // 组数 1 + 别名数 3
  // M3：严等价归一 + 守卫原样（alias==standard 跳过）
  CHECK(e.keywords().normalize("李经理") == "李建国");
  CHECK(e.keywords().normalize("小李") == "李建国");
  CHECK(e.keywords().normalize("李建国") == "李建国");
  CHECK(e.keywords().alias_of("小李") == "李建国");
  // M2：canonical 实体建档 + 全变体 Alias 节点
  const Node* ent = e.storage().find_by_name(NodeKind::Entity, "李建国");
  CHECK(ent != nullptr && ent->type == "person");
  CHECK(e.storage().find_all_by_name(NodeKind::Alias, "小李").size() == 1);
  // 摄入：非称呼变体（李经理）提取为关键词 → normalize 归并 → CONTAINS 挂 canonical
  e.remember_turn("张伟", "李经理把预算砍掉了一半。", 1700000000, "D1:1", "s1");
  const Node* kw_node = e.keywords().find("李建国");
  CHECK(kw_node != nullptr);
  bool contains_canon = false;
  for (auto& [ed, nb] : e.storage().neighbors(kw_node->uid, {RelType::CONTAINS}))
    if (nb->kind == NodeKind::Event && nb->attrs.count("dia_id") &&
        nb->attrs.at("dia_id") == "D1:1")
      contains_canon = true;
  CHECK(contains_canon);
  // 摄入：称呼轮（小李）→ M2 resolve 经 Alias 命中归并 canonical 实体（INVOLVES）
  auto r = e.remember_turn("张伟", "小李说数字要砍三成。", 1700000060, "D1:2", "s1");
  CHECK(r.code == Err::Ok);
  bool involves_canon = false;
  for (auto& [ed, nb] : e.storage().neighbors(ent->uid, {RelType::INVOLVES}))
    if (nb->kind == NodeKind::Event && nb->attrs.count("dia_id") &&
        nb->attrs.at("dia_id") == "D1:2")
      involves_canon = true;
  CHECK(involves_canon);
  // 幂等：重复加载不重复建档
  CHECK(apply_name_aliases(e, path, &applied) == Err::Ok);
  CHECK(e.storage().find_all_by_name(NodeKind::Entity, "李建国").size() == 1);
  // 坏文件：缺 groups → ParseError
  const char* bad = "/tmp/ame_test_names_bad.json";
  {
    std::ofstream f(bad);
    f << "{\"source\":\"names_zh\"}";
  }
  CHECK(apply_name_aliases(e, bad, nullptr) == Err::ParseError);
}
// 存储抽象（3.0 简化版：仅 graph 后端）：GraphStore 与直接 answer() 逐题一致
// （适配器零行为差），证据 dia_id 取自事件 attrs。
static void test_store_adapters() {
  auto loco = load_locomo(std::string(kFixtureDir) + "/locomo_mini.json");
  const BenchCase& c = loco[0];
  const BenchQuestion& q = c.questions[1];  // 两个金标 dia_id
  BenchConfig cfg;

  Engine e;
  GraphStore gs(&e);
  CHECK(&gs.engine() == &e);
  CHECK(gs.ingest_case(c).events == 9);
  auto ra = gs.answer_question(q, cfg);
  auto direct = answer(e, q, cfg);
  CHECK(ra.result.evidence_dia_ids == direct.evidence_dia_ids);  // 适配器零行为差
  CHECK(ra.result.abstained == direct.abstained);
  CHECK(!ra.evidence.empty());
  CHECK(ra.evidence.size() == ra.result.evidence_uids.size());
  bool graph_dia = false;
  for (auto& ev : ra.evidence)
    if (!ev.dia_id.empty()) graph_dia = true;
  CHECK(graph_dia);  // graph 证据 dia_id 从事件 attrs 取
}


// LLM 重排响应解析（容错）：编号序/去重/越界跳过/NONE/散文夹带/截 k
static void test_llm_rerank_parse() {
  // 常规编号序（0 基下标）
  auto v = parse_llm_rerank_ids("E3 E1 E7", 10, 3);
  CHECK(v.size() == 3 && v[0] == 2 && v[1] == 0 && v[2] == 6);
  // 去重 + 越界（E0/E11 对 n=10 无效）跳过
  v = parse_llm_rerank_ids("E2 E2 E0 E11 E5", 10, 5);
  CHECK(v.size() == 2 && v[0] == 1 && v[1] == 4);
  // NONE / 空响应 / 纯解释 → 空（降级信号）
  CHECK(parse_llm_rerank_ids("NONE", 10, 3).empty());
  CHECK(parse_llm_rerank_ids("", 10, 3).empty());
  CHECK(parse_llm_rerank_ids("这些证据都无法回答问题。", 10, 3).empty());
  // 散文中夹带编号（带词边界保护：the 里的 e 不误配）
  v = parse_llm_rerank_ids("我认为 E4 和 E2 最相关，the others are noise.", 10, 5);
  CHECK(v.size() == 2 && v[0] == 3 && v[1] == 1);
  // k 截断
  v = parse_llm_rerank_ids("E1 E2 E3", 10, 2);
  CHECK(v.size() == 2 && v[0] == 0 && v[1] == 1);
}

// LLM 重排通道（句级语义 spike，--llm-rerank N）：
//   BM25 终排把词面高频 decoy 置顶、金标压第 2 → LLM 返回 E2 后金标置顶；
//   LLM 空响应/无有效编号 → 优雅降级，保持 BM25 原序（默认 N=0 不触发另由 off 行为覆盖）。
static void test_llm_rerank_channel() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie", "Melanie loved The Silent Patient, a book she could not put down.",
                  ts0, "D1:1", "s1");  // 金标：词面弱匹配
  e.remember_turn("Caroline",
                  "Caroline asks what book to read next; did you finish your book?",
                  ts0 + 60, "D2:1", "s2");  // decoy：词面高频命中
  BenchQuestion q;
  q.id = "q-rerank-1";
  q.question = "What book was recommended to Melanie?";

  // 组1：LLM 选 E2（金标）→ 置顶
  {
    BenchConfig cfg;
    cfg.use_entity_profile = false;  // 只验标准管道重排
    cfg.rerank_mode = "llm";
    FakeLlmClient fake;
    fake.rules.push_back({"q-rerank-1", "E2"});
    cfg.llm_client = &fake;
    auto r = answer(e, q, cfg);
    CHECK(!r.evidence_uids.empty());
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D1:1");
    CHECK(fake.received.size() == 1);  // 每题一次调用
  }
  // 组2：LLM 空响应（模拟无网/失败）→ 降级 BM25 原序（decoy 居首）
  {
    BenchConfig cfg;
    cfg.use_entity_profile = false;
    cfg.rerank_mode = "llm";
    FakeLlmClient fake;
    fake.default_response = "";
    cfg.llm_client = &fake;
    auto r = answer(e, q, cfg);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
  }
  // 组3：LLM 返回无有效编号 → 同样降级
  {
    BenchConfig cfg;
    cfg.use_entity_profile = false;
    cfg.rerank_mode = "llm";
    FakeLlmClient fake;
    fake.default_response = "这些证据都不相关，无法回答。";
    cfg.llm_client = &fake;
    auto r = answer(e, q, cfg);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
  }
  // 组4：默认 cfg（llm_rerank=0）→ 通道关闭，行为与历史一致
  {
    BenchConfig cfg;
    cfg.use_entity_profile = false;
    FakeLlmClient fake;
    fake.default_response = "E1";  // 即便有客户端也不该被调用
    cfg.llm_client = &fake;
    auto r = answer(e, q, cfg);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
    CHECK(fake.received.empty());
  }
}

// 机制1 实体通道低置信放行（--entity-pass-threshold）：
//   通道榜首内容分 < 阈值 → 不整通道返回，候选并入标准管道池（去重）继续流程；
//   默认 -1（关）→ 整通道返回现状；槽位直查恒 1.0 永远高置信（隐式覆盖）。
static void test_entity_pass_through() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie", "My hobby is pottery, and this hobby keeps me busy.", ts0,
                  "D1:1", "s1");  // 通道候选：hobby ×2 → 内容分 ≈0.16（<0.15 会弃权，≥0.15 低置信）
  e.remember_turn("Melanie", "I went hiking with friends last weekend.", ts0 + 60, "D1:2",
                  "s1");  // 池内同伴事件：问题内容词零命中
  BenchQuestion q;
  q.id = "q-pass-1";
  q.question = "About Melanie zqzx hobby?";  // 内容词 about/zqzx/hobby：hobby 命中 → 通道分≈0.16（低置信带）

  // 组1：默认（阈值 -1 = 关）→ 整通道返回现状：证据仅通道候选 D1:1
  {
    BenchConfig cfg;  // entity_pass_threshold 默认 -1
    auto r = answer(e, q, cfg);
    CHECK(!r.abstained);
    // 通道整返回：1 条 dia_id 证据 + 会话标记 S:s1（answer_entity_channel 双标记惯例）
    CHECK(r.evidence_dia_ids.size() == 2 && r.evidence_dia_ids[0] == "D1:1" &&
          r.evidence_dia_ids[1] == "S:s1");
  }
  // 组2：阈值 0.5 → 低置信放行：标准管道接管，通道候选并入池（D1:2 经实体扩散也入池）
  {
    BenchConfig cfg;
    cfg.entity_pass_threshold = 0.5;
    auto r = answer(e, q, cfg);
    CHECK(!r.abstained);
    bool has_d11 = false, has_d12 = false;
    for (auto& d : r.evidence_dia_ids) {
      if (d == "D1:1") has_d11 = true;
      if (d == "D1:2") has_d12 = true;
    }
    CHECK(has_d11);  // 通道候选不丢（并入标准池）
    CHECK(has_d12);  // 标准管道自身取证在放行后生效（整通道返回时 D1:2 不可见）
  }
  // 组3：阈值 0 且通道弃权情形不回归——低分通道（<0.15）本来就走标准管道
  {
    BenchConfig cfg;
    cfg.entity_pass_threshold = 0.0;  // 0.16 ≥ 0 → 仍整通道返回
    auto r = answer(e, q, cfg);
    CHECK(!r.abstained);
    CHECK(r.evidence_dia_ids.size() == 2 && r.evidence_dia_ids[0] == "D1:1" &&
          r.evidence_dia_ids[1] == "S:s1");
  }
}

// 机制2 全池 LLM 重排（--llm-rerank-all）：窗口从 top-N 扩为全池；与 --llm-rerank 并存时全池优先。
static void test_llm_rerank_all() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie", "Melanie loved The Silent Patient, a book she could not put down.",
                  ts0, "D1:1", "s1");  // 金标：词面弱匹配
  e.remember_turn("Caroline",
                  "Caroline asks what book to read next; did you finish your book?",
                  ts0 + 60, "D2:1", "s2");  // decoy：词面高频命中
  BenchQuestion q;
  q.id = "q-rerank-all-1";
  q.question = "What book was recommended to Melanie?";
  BenchConfig cfg;
  cfg.use_entity_profile = false;
  cfg.rerank_mode = "llm";
  FakeLlmClient fake;
  fake.rules.push_back({"q-rerank-all-1", "E2"});
  cfg.llm_client = &fake;
  auto r = answer(e, q, cfg);
  CHECK(!r.evidence_uids.empty());
  const Node* front = e.storage().get_node(r.evidence_uids.front());
  CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D1:1");
  // 默认（全池关、N=0）→ 通道关闭（与 test_llm_rerank_channel 组4 同语义，不再重复断言）
}

// 任务A BM25 强命中豁免（--rerank-guard）：榜首内容分 ≥ 阈值时 LLM 重排不得挤走它。
static void test_rerank_guard() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie",
                  "Pottery class, pottery wheel, pottery glaze, pottery kiln, pottery studio — I love it all.",
                  ts0, "D1:1", "s1");  // 金标：pottery tf=5 → 单内容词 BM25 分 = 5/6 ≈ 0.833
  e.remember_turn("Caroline", "I went hiking with friends last weekend.", ts0 + 60, "D2:1",
                  "s2");  // decoy：零词面命中（BM25 0 分居次位）
  BenchQuestion q;
  q.id = "q-guard-1";
  q.question = "pottery?";  // 单内容词：idf 相消，榜首分 = tf/(tf+1) = 5/6 ≈ 0.833
  BenchConfig cfg;
  cfg.use_entity_profile = false;
  cfg.rerank_mode = "llm";
  FakeLlmClient fake;
  fake.rules.push_back({"q-guard-1", "E2"});  // LLM 误判 decoy 为相关
  cfg.llm_client = &fake;
  // 组1：豁免关（默认 -1）→ LLM 误判生效，decoy 居首
  {
    auto r = answer(e, q, cfg);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
  }
  // 组2：豁免 0.8（榜首内容分 = 5/6 ≈ 0.833 > 0.8）→ 保底首槽，LLM 误判被纠正
  {
    BenchConfig cfg2 = cfg;
    cfg2.rerank_guard = 0.8;
    auto r = answer(e, q, cfg2);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D1:1");
  }
}

// 任务B 便宜近似（--cheap-rerank）：特征逻辑重排器替代 LLM 打分（零 LLM）。
static void test_cheap_rerank() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie", "Melanie loved The Silent Patient, a book she could not put down.",
                  ts0, "D1:1", "s1");  // 金标：句级相关、词面弱
  e.remember_turn("Caroline",
                  "Caroline asks what book to read next; did you finish your book?",
                  ts0 + 60, "D2:1", "s2");  // decoy：词面高频
  // 玩具模型：唯 bm25_share 权重 -1（偏好低词面份额 = 句级信号替身）
  const char* mpath = "/tmp/ame_test_cheap_model.json";
  {
    std::ofstream f(mpath);
    f << "{\"features\":[\"bm25_share\"],\"weights\":[-1.0],\"bias\":0.0}";
  }
  BenchQuestion q;
  q.id = "q-cheap-1";
  q.question = "What book was recommended to Melanie?";
  BenchConfig cfg;
  cfg.use_entity_profile = false;
  cfg.cheap_rerank_path = mpath;
  cfg.rerank_mode = "cheap";
  // 组1：便宜重排 → 模型序替代 BM25 序，金标（低份额）居首
  {
    auto r = answer(e, q, cfg);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D1:1");
  }
  // 组2：模型路径不存在 → 加载失败优雅降级 BM25 原序（decoy 居首）
  {
    BenchConfig cfg2 = cfg;
    cfg2.cheap_rerank_path = "/tmp/ame_test_no_such_model.json";
    auto r = answer(e, q, cfg2);
    const Node* front = e.storage().get_node(r.evidence_uids.front());
    CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
  }
}
// 模式门控：bm25 档（默认）即使有 LLM 客户端/全池标志也不触发重排（Fake 零调用）；
// cheap 档未配模型路径 → 日志降级、行为 = BM25 原序。
static void test_rerank_mode_gating() {
  Engine e;
  const TimePoint ts0 = 1700000000;
  e.remember_turn("Melanie", "Melanie loved The Silent Patient, a book she could not put down.",
                  ts0, "D1:1", "s1");
  e.remember_turn("Caroline",
                  "Caroline asks what book to read next; did you finish your book?",
                  ts0 + 60, "D2:1", "s2");
  BenchQuestion q;
  q.id = "q-mode-gate";
  q.question = "What book was recommended to Melanie?";
  BenchConfig cfg;
  cfg.use_entity_profile = false;
  // bm25 档：全池标志 + 客户端齐备也不得触发（模式互斥于参数）
  cfg.rerank_mode = "bm25";
  FakeLlmClient fake;
  fake.default_response = "E2";  // 若误触发会把金标置顶——断言不触发
  cfg.llm_client = &fake;
  auto r = answer(e, q, cfg);
  CHECK(fake.received.empty());
  const Node* front = e.storage().get_node(r.evidence_uids.front());
  CHECK(front && front->attrs.count("dia_id") && front->attrs.at("dia_id") == "D2:1");
  // cheap 档未配路径 → 降级 BM25 原序（Fake 同样零调用）
  BenchConfig cfg2;
  cfg2.use_entity_profile = false;
  cfg2.rerank_mode = "cheap";
  cfg2.llm_client = &fake;
  auto r2 = answer(e, q, cfg2);
  CHECK(fake.received.empty());
  const Node* front2 = e.storage().get_node(r2.evidence_uids.front());
  CHECK(front2 && front2->attrs.count("dia_id") && front2->attrs.at("dia_id") == "D2:1");
}

int main() {
  test_load_fixtures();
  test_ingest_graph();
  test_metrics_known_values();
  test_abstention_gate();
  test_knowledge_update();
  test_order_events();
  test_snapshot_with_slots();
  test_apply_name_aliases();
  test_evidence_content_rerank();
  test_llm_rerank_parse();
  test_llm_rerank_channel();
  test_llm_rerank_all();
  test_entity_pass_through();
  test_rerank_guard();
  test_cheap_rerank();
  test_rerank_mode_gating();
  test_parse_session_date();
  test_store_adapters();
  return AME_TEST_REPORT();
}
