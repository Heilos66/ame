# AME 独立 RAGFlow 兼容检索服务

此模块随 AME fork 源码维护，通过 Python 标准库提供 HTTP，通过本仓库编译的 `ame_worker` 调用 C++ 检索引擎。不依赖文旅 Agent 仓库、Java、Maven 或 Docker。默认 AME 库构建保持不变，启用服务时设置 `AME_BUILD_RAGFLOW_SERVICE=ON`。

推荐使用 [一键部署与升级流程](DEPLOYMENT.md)：`deploy.sh upgrade --port 18088` 更新已有实例，`deploy.sh install --port 18089 --corpus /tmp/ame-corpus.json` 创建新实例。脚本检查依赖、保留 Token、验证接口，并在升级失败时恢复旧适配文件。下文保留接口说明和手动部署方式。

## 接口与参数

`POST /api/v1/retrieval`，`Content-Type: application/json`，`Authorization: Bearer <token>`。Token 文件仅保存原始值，不包含 `Bearer` 前缀。服务启动时读取 Token，变更后需要重启。`GET /health` 无需 Token。

以下两种请求均支持，只有 `question` 和 `page_size` 会传入 worker：

```json
{
  "dataset_ids": ["ame-travel"],
  "question": "test",
  "page": 1,
  "page_size": 3,
  "similarity_threshold": 0
}
```

```json
{
  "dataset_ids": ["ame-travel"],
  "question": "test",
  "page": 1,
  "page_size": 3,
  "similarity_threshold": 0.2,
  "knn_top_k": 64,
  "knn_num_candidates": 128,
  "rerank_id": "bge-reranker-v2-m3@maas-hw@OpenAI-API-Compatible",
  "rerank_candidates_count": 32
}
```

| 参数 | 行为 |
| --- | --- |
| `dataset_ids` | 必须是只包含本实例 `AME_DATASET_ID` 的数组，不跨知识库检索 |
| `question` | 非空白字符串，最多 10000 个 Unicode 字符 |
| `page_size` | 1～50 的整数，默认 3 |
| `page` | 仅支持 1，默认 1 |
| `similarity_threshold` | 接受但忽略，包括 0.2、null 或省略 |
| `knn_top_k` / `knn_num_candidates` | 接受但忽略 |
| `rerank_id` / `rerank_candidates_count` | 接受但忽略，包括空字符串或 null |

兼容参数不会启用远端重排或改变 AME 的图检索配置。AME 分数可以为负，不是 0～1 的相似度。响应保持 `code=0`、`data.chunks` 和耗时 `meta`，空命中也为成功。文旅 Agent 仍使用 `KNOWLEDGE_BACKEND=ame`，以免客户端按 RAGFlow 分数过滤结果。

Token 不匹配返回 401；必要参数错误返回 400；请求超过 262144 字节返回 413；请求体读取超时返回 408；队列满返回 503；worker 失败返回 502。每个请求必须有一个有效的 `Content-Length`，不支持 chunked 请求体。没有新增管理或写入接口；语料在启动时导入。

## 构建与验证

需要 CMake 3.16+、C++17 编译器、Python 3.10+。以下命令使用已检出本次改动的 AME fork，固定目录为 `/opt/ame`；命令均为 ASCII，不复用 Bash 变量。仓库来源为 `https://github.com/Heilos66/ame.git`，开发分支为 `codex/ragflow-compatible-retrieval`，分支推送后才能在 ECS 克隆它。

```bash
cmake -S /opt/ame -B /opt/ame/build \
  -DAME_BUILD_RAGFLOW_SERVICE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/ame/build --parallel 2
ctest --test-dir /opt/ame/build --output-on-failure
```

worker 产物为 `/opt/ame/build/services/ragflow/ame_worker`。CTest 同时执行核心测试、HTTP 契约测试及 HTTP 到真实 C++ worker 的检索测试，使用临时测试语料，无需业务知识或真实 Token。只运行以下 Python 命令时，真实 worker 测试会标记跳过，其余接口测试使用模拟 worker：

```bash
python3 -B -m unittest discover -s /opt/ame/services/ragflow/tests -v
```

## 已部署 18088 实例的最小升级

此前 `travel-ame-18088.service` 使用 `/opt/travel-ame-18088/server.py`，仅更新 AME 核心源码不会改变这个正在运行的 HTTP 适配器。可从 fork 复制新的适配器，继续使用已有 worker、语料、Token 和 systemd 配置，不必编译 C++。

先确认实际启动路径：

```bash
sudo systemctl show travel-ame-18088.service -p ExecStart
```

只有实际使用上述路径时执行以下命令；首次备份会保留，重复执行不会覆盖备份：

```bash
sudo cp -n /opt/travel-ame-18088/server.py \
  /opt/travel-ame-18088/server.py.before-ragflow-compat
sudo install -m 755 /opt/ame/services/ragflow/server.py \
  /opt/travel-ame-18088/server.py
sudo systemctl restart travel-ame-18088.service
sudo journalctl -u travel-ame-18088.service -n 60 --no-pager
curl --noproxy '*' -i --max-time 5 http://127.0.0.1:18088/health
```

等待日志出现 `ame_service status=ready`，再执行带调优参数的检索。下面直接读取第二个实例的 Token，不需要粘贴密钥：

```bash
sudo python3 - <<'PY'
import http.client
import json
from pathlib import Path

body = json.dumps({
    'dataset_ids': ['ame-travel-18088'],
    'question': '\u62f1\u5bb8\u6865',
    'page': 1, 'page_size': 3, 'similarity_threshold': 0.2,
    'knn_top_k': 64, 'knn_num_candidates': 128,
    'rerank_id': 'bge-reranker-v2-m3@maas-hw@OpenAI-API-Compatible',
    'rerank_candidates_count': 32,
})
connection = http.client.HTTPConnection('127.0.0.1', 18088, timeout=12)
try:
    connection.request('POST', '/api/v1/retrieval', body, {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + Path('/etc/travel-ame-18088/token').read_text().strip(),
    })
    response = connection.getresponse()
    result = json.loads(response.read())
    print('HTTP', response.status)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    assert response.status == 200 and result.get('code') == 0, 'Retrieval failed'
finally:
    connection.close()
PY
```

预期 HTTP 200 / `code=0`；是否命中需结合本实例语料确认。401 仍检查该实例 Token；400 仍检查知识库 ID、问题、分页和返回条数。18087 实例需按其实际发布目录更新 `server.py`，重启 `travel-ame.service`，并使用对应的 `ame-travel` 和 Token。

第二个实例的回滚命令：

```bash
sudo install -m 755 /opt/travel-ame-18088/server.py.before-ragflow-compat \
  /opt/travel-ame-18088/server.py
sudo systemctl restart travel-ame-18088.service
```

## 新部署的 systemd 模板

`ame-ragflow.service` 用于没有已有实例的新部署，默认端口 18087；与已有服务并存时必须调整端口、数据和凭据路径，不能争用原端口。语料 JSON 是数组，每项包含唯一 `id`、`document`、`content` 三个非空字符串。业务语料和 Token 不提交到 Git。

```bash
sudo install -d -m 755 /etc/ame
sudo install -m 644 /tmp/corpus.json /etc/ame/corpus.json
sudo python3 - <<'PY'
import os
import secrets

fd = os.open('/etc/ame/token', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as output:
    output.write(secrets.token_urlsafe(48) + '\n')
print('Token created')
PY
sudo install -m 644 /opt/ame/services/ragflow/ame-ragflow.service \
  /etc/systemd/system/ame-ragflow.service
sudo systemctl daemon-reload
sudo systemctl enable --now ame-ragflow.service
sudo journalctl -u ame-ragflow.service -n 60 --no-pager
curl --noproxy '*' -i --max-time 5 http://127.0.0.1:18087/health
```

Token 已存在时生成命令会失败并保留旧文件，不要随意替换已有凭据。加载完成后日志包含 `ame_service status=ready chunks=...`；检索成功记录 `ame_retrieval status=succeeded chunks=... elapsed_ms=...`。日志不含 Token、问题或知识正文，目前不单独打印 400/401 的拒绝原因。

| 环境变量 | 默认 / 要求 |
| --- | --- |
| `AME_WORKER` | 必填，编译后的 worker 路径 |
| `AME_CORPUS` | 必填，语料 JSON 路径 |
| `AME_DICTIONARY` | 必填，词典路径 |
| `AME_CACHE` | 必填，服务用户可写的词典缓存路径 |
| `AME_TOKEN_FILE` | 必填，原始 Token 文件，长度至少 32，无空白或控制字符 |
| `AME_DATASET_ID` | 默认 `ame-travel`，1～64 个英文字母、数字、下划线或连字符 |
| `AME_BIND` | Python 默认 `127.0.0.1`；systemd 模板为 `0.0.0.0` |
| `AME_PORT` | 默认 18087 |

服务限制 32 个连接、8 个等待或执行中的检索，单 worker 串行处理，服务端请求预算 8 秒。worker 失效后进程退出，由 systemd 重启；完整词典及语料会重新加载。不同知识库使用独立实例、目录、Token 和端口。私网调用需放行目标端口及调用方来源，HTTP 仍需要 Bearer Token。
