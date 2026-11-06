cmake_minimum_required(VERSION 3.16)
project(ame CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

enable_testing()

# 模块依赖：core ← storage ← time/vector ← keyword/extractor ← entity/graph_writer
#          ← seed/diffusion ← rerank ← thinker ← feedback/learning ← engine
add_subdirectory(core)          # 公共类型/uid/JSON/日志
add_subdirectory(storage)       # M9  存储引擎
add_subdirectory(time)          # M10 时间服务
add_subdirectory(vector)        # M13 向量服务
add_subdirectory(keyword)       # M3  关键词管理器
add_subdirectory(extractor)     # M1  要素提取器
add_subdirectory(entity)        # M2  实体解析器
add_subdirectory(graph_writer)  # M4  图写入器
add_subdirectory(seed)          # M5  种子生成器
add_subdirectory(diffusion)     # M6  扩散引擎
add_subdirectory(rerank)        # M7  打分重排器
add_subdirectory(engine)        # M12 API 网关
add_subdirectory(bench)         # 基准适配层（LoCoMo/LongMemEval/BEAM，依赖 M12）
add_subdirectory(llm)           # LLM 接入层（DeepSeek 客户端 + 提取/答题/评审组件）

add_subdirectory(tests)
add_subdirectory(demo)

option(AME_BUILD_RAGFLOW_SERVICE "Build the standalone RAGFlow-compatible HTTP service" OFF)
if(AME_BUILD_RAGFLOW_SERVICE)
  add_subdirectory(services/ragflow)
endif()
