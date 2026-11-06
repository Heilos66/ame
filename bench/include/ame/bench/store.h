/*
 * 所属模块：bench 基准适配层——存储抽象（3.0 简化版：仅 1.x graph 后端）。
 * 本文件主体思路：窄接口 IStore（摄入一个 case + 逐题取证召回），
 *   GraphStore 包现有 Engine 调用（ingest()/answer()）；2.0 lmem 后端已随
 *   M15–M19 模块一并裁剪（dev-3.0 刀1，依据 docs/reports/ 归档的停用清单）。
 * 关键算法/数据结构：StoreAnswer 复用既有 AnswerResult（metrics 口径不变），
 *   StoreEvidence 携带逐条证据的 dia_id（取自事件 attrs）。
 * 依赖关系：依赖 bench.h；实现依赖 M12 engine（store.cpp）。
 */
#pragma once
#include "ame/bench/bench.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ame {

class Engine;

// 一次取证召回的逐条证据
struct StoreEvidence {
  std::string text;      // 证据文本（日期前缀 + 原文，可直接喂 LLM）
  std::string dia_id;    // 证据 dia_id（取自事件 attrs；无则空）
  int64_t ts = 0;        // 证据时刻
  double score = 0;      // 后端打分（仅供观察，不进指标）
};

// 一题取证结果：metrics 口径 + 逐条证据
struct StoreAnswer {
  AnswerResult result;                 // answer_text/abstained/score/evidence_dia_ids 沿用既有口径
  std::vector<StoreEvidence> evidence;
};

// 窄接口：存储后端的公共面（摄入 + 取证）
class IStore {
 public:
  virtual ~IStore() = default;
  virtual IngestStats ingest_case(const BenchCase& c) = 0;
  virtual StoreAnswer answer_question(const BenchQuestion& q, const BenchConfig& cfg) = 0;
};

// 1.x 适配器：包现有 Engine 调用（recall 结果的 dia_id 从事件 attrs 取——
// 摄入时 dia_id 落在事件 attrs）
class GraphStore final : public IStore {
 public:
  explicit GraphStore(Engine* engine) : engine_(engine) {}
  IngestStats ingest_case(const BenchCase& c) override;
  StoreAnswer answer_question(const BenchQuestion& q, const BenchConfig& cfg) override;
  Engine& engine() { return *engine_; }

 private:
  Engine* engine_;
};

}  // namespace ame
