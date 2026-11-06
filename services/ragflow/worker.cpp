// Standalone graph retrieval worker. No remote model calls.
#include "ame/engine/engine.h"
#include "ame/core/json.h"
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>

using ame::Json;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static const Json& required(const Json& value, const std::string& key) {
  auto field = value.find(key);
  if (!field) throw std::runtime_error("missing required field");
  return *field;
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("expected corpus, dictionary and cache paths");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("corpus unreadable");
    bool valid = false;
    auto corpus = Json::parse(std::string(std::istreambuf_iterator<char>(input), {}), &valid);
    if (!valid || corpus.kind != Json::Arr || corpus.arr.empty()) {
      throw std::runtime_error("invalid corpus");
    }
    ame::Engine engine("");
    if (engine.extractor().load_keyword_dict_cached(argv[2], "", 10000000, argv[3]) <= 0) {
      throw std::runtime_error("dictionary unavailable");
    }
    std::map<std::string, Json> metadata;
    // Equal timestamps prevent import order from manufacturing recency relevance.
    auto now = static_cast<ame::TimePoint>(std::time(nullptr));
    for (const auto& chunk : corpus.arr) {
      auto id = required(chunk, "id").as_str();
      auto text = required(chunk, "content").as_str();
      required(chunk, "document");
      if (id.empty() || text.empty()) throw std::runtime_error("invalid chunk");
      auto result = engine.remember_turn("knowledge", text, now, id, "travel");
      if (result.code != ame::Err::Ok || !metadata.emplace(result.event_id, chunk).second) {
        throw std::runtime_error("corpus import failed");
      }
    }
    Json ready = Json::object();
    ready.set("ready", Json::boolean(true));
    ready.set("chunks", Json::number(corpus.arr.size()));
    std::cout << ready.dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        auto request = Json::parse(line, &valid);
        if (!valid) throw std::runtime_error("invalid JSON");
        auto query = required(request, "query").as_str();
        auto raw_limit = required(request, "limit").as_num();
        if (query.empty() || query.size() > 40000 || !std::isfinite(raw_limit)
            || raw_limit < 1 || raw_limit > 50 || raw_limit != std::floor(raw_limit)) {
          throw std::runtime_error("invalid query or limit");
        }
        auto start = Clock::now();
        ame::SeedQuery seed;
        seed.text = query;
        seed.lex_topk = 30;
        seed.soft_min_sim = 0.2;
        auto result = engine.recall(seed, {}, 2, static_cast<int>(raw_limit), false);
        if (result.code != ame::Err::Ok && result.code != ame::Err::NotFound) {
          throw std::runtime_error("recall failed");
        }
        Json hits = Json::array();
        for (const auto& hit : result.events) {
          if (!hit.event || !std::isfinite(hit.score)) continue;
          auto item = metadata.at(hit.event->uid);
          item.set("score", Json::number(hit.score));
          hits.push(item);
        }
        Json response = Json::object();
        response.set("hits", hits);
        response.set("recall_ms", Json::number(elapsed_ms(start)));
        std::cout << response.dump() << std::endl;
      } catch (const std::exception&) {
        std::cout << "{\"error\":\"retrieval_failed\"}" << std::endl;
      }
    }
  } catch (const std::exception&) {
    std::cerr << "AME worker initialization failed" << std::endl;
    return 1;
  }
}
