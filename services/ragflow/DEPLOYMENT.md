# AME 一键部署与升级

适用于 Huawei Cloud EulerOS 3.0 / x86_64 等使用 systemd 的 Linux ECS。本流程只需要 `Heilos66/ame` fork，不需要文旅工程源码。命令均为 ASCII，使用完整路径，不需要定义或复用 Bash 变量。

## 1. 选择操作

| 当前情况 | 使用操作 | 是否编译 C++ |
| --- | --- | --- |
| 已按原文档部署 `travel-ame-18088.service`，只更新参数兼容逻辑 | `upgrade --port 18088` | 否，复用已有 worker |
| 已部署原 `travel-ame.service`，端口为 18087 | `upgrade --port 18087` | 否 |
| 全新安装，或需要新增独立知识库实例 | `install`，提供空闲端口和语料 | 是，在 ECS 本机编译 |

脚本默认端口为 18088；本文新安装示例使用 **18089**，避免占用原有 18087/18088。`upgrade` 只更新 HTTP 适配器，不更新 C++ 核心或语料。`install` 发现目标端口或同名服务已存在会停止，不会覆盖它们。

## 2. 准备环境和源码

升级需要系统级 Python 3.10+ 和正在运行的 systemd；新安装使用 systemd 247+ 的凭据功能，另外需要 CMake 3.16+、C++17 编译器和 make。先检查：

```bash
python3 --version
systemctl --version
```

新安装还检查：

```bash
cmake --version
c++ --version
make --version
```

Huawei Cloud EulerOS 3.0 上缺少依赖时，使用已配置好的系统软件源安装：

```bash
sudo dnf install -y cmake make gcc gcc-c++ git python3 curl ca-certificates
```

安装后重新检查版本；`dnf install` 成功不代表 Python 已达到 3.10。systemd 通常已随系统提供。脚本不会自动升级系统软件，缺少依赖会停止并提示。Python 应安装在系统目录，例如 `/usr/bin`、`/usr/local` 或 `/opt`，不要依赖只有登录用户能访问的虚拟环境。新实例首次加载词典需要额外内存，模板为每个实例限制 2 GiB 内存和 1 核 CPU；同机多实例需预留资源。

如果 `/opt/ame` 尚不存在，拉取包含本次脚本的分支：

```bash
sudo git clone --branch codex/ragflow-compatible-retrieval --single-branch \
  https://github.com/Heilos66/ame.git /opt/ame
```

如果已经在 `/opt/ame` 检出该分支，更新它：

```bash
sudo git -C /opt/ame status --short --branch
sudo git -C /opt/ame pull --ff-only
```

确认脚本已经获取：

```bash
ls -l /opt/ame/services/ragflow/deploy.sh /opt/ame/services/ragflow/deploy.py
sudo bash /opt/ame/services/ragflow/deploy.sh --help
```

若本机还未将新增脚本提交并推送到远端，先完成推送；旧提交只有 `server.py`，没有一键部署脚本。Git 提示本地修改或分支不一致时，先核对当前代码，不要用强制重置覆盖。

## 3. 升级已有的 18088 实例

以下命令适用于原部署文档的路径和服务名：

```bash
sudo bash /opt/ame/services/ragflow/deploy.sh upgrade --port 18088
```

脚本执行以下操作：

1. 检查 `travel-ame-18088.service` 正在运行，并确认 `ExecStart` 指向 `/opt/travel-ame-18088/server.py`、端口为 18088。
2. 从服务配置读取知识库 ID，从 `/etc/travel-ame-18088/token` 读取原 Token。
3. 备份原脚本为 `server.py.backup-<UTC时间>`，原子替换为 fork 中的新版本。
4. 重启同一个服务，等待 `/health` 就绪，默认最多等待 180 秒。
5. 使用真实 Token 调用检索接口，携带 `similarity_threshold=0.2`、KNN 参数和重排参数，验证 HTTP 200 / `code=0`。
6. 如果重启或接口验证失败，恢复旧脚本并重启旧服务；命令仍返回失败，便于发现问题。

语料、worker、Token 和 systemd 配置均沿用已有文件。升级需要短暂重启，加载期间请求可能失败。成功输出示例：

```text
Backup: /opt/travel-ame-18088/server.py.backup-20261106-040000-123456
Retrieval verified: HTTP 200, chunks=0
Deployment succeeded
Service: travel-ame-18088.service
Port: 18088
Dataset: ame-travel-18088
Token file: /etc/travel-ame-18088/token
```

验证问题为 `test`，所以 `chunks=0` 也表示协议验证成功；真实知识召回质量需再用业务问题验证。脚本不打印 Token 或知识正文。

升级原 18087 实例：

```bash
sudo bash /opt/ame/services/ragflow/deploy.sh upgrade --port 18087
```

它会检查 `travel-ame.service`，更新 `/opt/travel-ame/current/server.py` 实际指向的文件，并保留同目录备份。此操作是有备份的原地升级；如果必须保持旧发布目录不可修改，请采用原来的新发布目录切换方式。

自定义安装路径时，可以显式指定，例如：

```bash
sudo bash /opt/ame/services/ragflow/deploy.sh upgrade \
  --port 18088 \
  --service travel-ame-18088 \
  --server /opt/travel-ame-18088/server.py \
  --token-file /etc/travel-ame-18088/token \
  --wait 240
```

目前自动升级要求服务直接使用 `python server.py` 启动，`AME_PORT`、`AME_DATASET_ID` 等配置直接写在 unit 的 `Environment=` 中，监听为 `0.0.0.0` 或 `127.0.0.1`，与此前模板一致。包装脚本、EnvironmentFile 或其他特殊启动方式需要按实际配置手动升级，脚本不会猜测并修改。

## 4. 在新端口安装独立实例

先把转换好的 `corpus.json` 上传或下载到 ECS 的 `/tmp/ame-corpus.json`。文件必须是非空数组，每项包含唯一 `id`、非空 `document` 和 `content`；不能直接使用 Markdown 原文。该语料留在 ECS，不提交到 Git。

一条命令完成编译、测试、文件安装、Token 创建、systemd 注册和接口验证：

```bash
sudo bash /opt/ame/services/ragflow/deploy.sh install \
  --port 18089 \
  --dataset-id ame-travel-18089 \
  --corpus /tmp/ame-corpus.json
```

脚本只编译必要的 worker，然后执行 HTTP 契约测试和真实 worker 检索测试。使用临时构建目录，避免复制旧仓库时遗留的 CMake 缓存指向错误目录。实际运行文件独立于 Git 工作目录，后续 `git pull` 不会直接改变正在运行的代码。

| 内容 | 新实例的位置 |
| --- | --- |
| systemd 服务 | `ame-ragflow-18089.service` |
| unit 文件 | `/etc/systemd/system/ame-ragflow-18089.service` |
| 发布文件 | `/opt/ame-ragflow/18089/releases/<UTC时间>/` |
| 版本记录 | 发布目录中的 `manifest.json`，包含源码提交号及代码、worker、语料和词典的 SHA-256 |
| Token | `/etc/ame-ragflow/18089/token`，权限 600 |
| 缓存 | `/var/lib/ame-ragflow-18089/` |
| 日志 | systemd journal |

首次安装随机生成 Token；重试时保留已有 Token。若要沿用一个已有 Token 文件，可增加 `--token-file /etc/travel-ame/token`，脚本会将其复制到新实例自己的 Token 文件。已有 Token 与指定值不一致时会停止，避免意外轮换。为了知识库隔离，通常让新实例生成独立 Token。

安装验证失败时会停止并移除本次新增的 systemd 服务，保留发布目录和 Token 便于检查；不会停止其他实例。修复问题后可以重试。如果只是回滚后服务无法恢复，脚本会返回失败，需要使用 journal 检查，不能视为部署成功。

## 5. 验证私网访问并配置调用方

在另一台同 VPC 的 ECS 上，以新实例为例执行：

```bash
curl --noproxy '*' -i --connect-timeout 5 --max-time 10 \
  http://172.16.0.154:18089/health
```

预期 HTTP 200 和 `{"ready":true}`。目标 ECS 安全组及系统防火墙需允许调用方访问 TCP 18089；18087 可访问不代表 18089 已放行。脚本不修改网络访问规则，也不绑定公网 IP。

AgentArts 对应连接配置：

```json
{
  "KNOWLEDGE_BACKEND": "ame",
  "RAGFLOW_ENABLED": "true",
  "RAGFLOW_URL": "http://172.16.0.154:18089/api/v1/retrieval",
  "RAGFLOW_DATASET_ID": "ame-travel-18089",
  "RAGFLOW_TOP_K": "3",
  "RAGFLOW_TIMEOUT": "10",
  "RAGFLOW_RETRIES": "0"
}
```

另将 `/etc/ame-ragflow/18089/token` 的原始内容配置为 `RAGFLOW_API_TOKEN`，不添加 `Bearer`。检查没有旧的 `RAGFLOW_API_TOKEN_FILE` 覆盖该值。不要把 Token 写入代码或提交到仓库。配置完成后发布运行时版本，用新 Session 验证。

如果只是原地升级 18088，原 URL、知识库 ID 和 Token 都不变，不需要为这次参数兼容改动重建文旅镜像。`similarity_threshold`、KNN 和重排参数仅接受并忽略；它们不会改变 AME 的图检索算法。

## 6. 日志与故障定位

查看新安装实例：

```bash
sudo systemctl status ame-ragflow-18089.service --no-pager
sudo journalctl -u ame-ragflow-18089.service -n 100 --no-pager
sudo journalctl -u ame-ragflow-18089.service -f
```

查看原升级实例：

```bash
sudo journalctl -u travel-ame-18088.service -n 100 --no-pager
```

| 现象 | 检查方向 |
| --- | --- |
| 脚本提示端口或服务已存在 | 已有实例使用 `upgrade`；新实例换空闲端口 |
| `ExecStart does not match` | 核对实际启动文件及 `--server`，不要更新另一个实例 |
| HTTP 401 | Token 是否匹配目标实例，服务是否加载旧 Token |
| HTTP 400 | 知识库 ID、问题、page_size 或 page；确认服务实际加载了新 server.py |
| 健康检查超时 | 查看 journal，检查语料、词典、内存、Python 版本和监听地址 |
| 本机正常，其他 ECS 超时 | 安全组、网络 ACL、系统防火墙及调用地址 |

部署脚本的回归测试覆盖成功安装、保护已有服务、备份、失败回滚、Token 保留及参数兼容验证；systemd 操作在测试中模拟。最终仍需在目标 ECS 执行安装和私网访问验证。
