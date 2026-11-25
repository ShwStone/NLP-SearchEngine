# 多数据集网站与本机服务部署

## 访问入口与数据范围

公开站点为 [https://search.shwstone.cn](https://search.shwstone.cn)。阿里云只提供静态文件、HTTPS、限流和反向代理。语料、模型、倒排索引与 Qdrant 均留在本机。

| 前端 ID | 数据 | 文档数 | 可用模式 | 本机端口 |
| --- | --- | --- | --- | --- |
| `news20k` | 团队自爬新闻 | 20,000 | lexical、dense、hybrid、adaptive | 18180 |
| `mini100k` | DuReader 固定子集 | 100,000 | lexical、dense、hybrid、adaptive | 18181 |
| `official8m` | DuReader 官方 passage | 8,096,668 | lexical、dense、hybrid、adaptive | 18086 |

三套数据源均已配置 Compact 词法索引与固定 BGE-small-zh-v1.5 Qdrant HNSW 向量索引，支持完整四模式检索、BM25/TF-IDF 切换、RM3 伪相关反馈扩展及浏览器四模式并行对比。本机检索进程与 SSH 反向隧道均由 `systemd --user` 托管。

## 请求链路

```text
Browser
  -> HTTPS search.shwstone.cn
  -> Aliyun Nginx
  -> 127.0.0.1:18081 on Aliyun, SSH remote forward
  -> 127.0.0.1:18083 on the retrieval host, online.gateway
  -> the selected online.server process
  -> Compact lexical index and/or loopback Qdrant
```

网关只接受配置中明确列出的 loopback upstream。目录响应不包含本机路径、Qdrant 地址或进程信息。旧的 `/search` 与 `/index/info` 默认转发到 20k，数据集路由使用以下接口。

```text
GET  /datasets
GET  /datasets/{id}/info
POST /datasets/{id}/search
GET  /gateway/health
```

示例请求。

```bash
curl -s https://search.shwstone.cn/datasets | python3 -m json.tool
curl -s https://search.shwstone.cn/datasets/official8m/info | python3 -m json.tool
curl -s -X POST https://search.shwstone.cn/datasets/official8m/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"国家法定节假日共多少天","mode":"lexical","k":5}'
```

## 本机目录

实际部署使用 Git 外的运行目录，下面用 `$RUNTIME` 表示。

```text
$RUNTIME/
  artifacts/dureader-mini100k-v1/{lexical,compact}
  config/{news20k,mini100k,official8m,datasets}.json
  config/qdrant-official-tuned.yaml
  python-deps/                  # isolated NumPy and Numba packages
  logs/
  runs/
```

20k 与 8M 继续使用团队已经验收的冻结产物。8M 在本机 NVMe 路径读取 compact 与 Qdrant 存储。`official8m.json` 对应 `deploy/config/duretrieval-official-fast.example.json`，数据集目录模板为 `deploy/config/datasets.example.json`。

## 构建 100k 词法产物

```bash
python3 -m offline.stream_build \
  --corpus /data/shwstone/Workspace/NLP-data/dureader-mini-100k-v1/local-release/corpus.jsonl \
  --out "$RUNTIME/artifacts/dureader-mini100k-v1/lexical" \
  --work-dir "$RUNTIME/artifacts/dureader-mini100k-v1/work" \
  --dataset dureader-mini100k-v1 --progress-every 10000

python3 -m offline.compact \
  --source "$RUNTIME/artifacts/dureader-mini100k-v1/lexical" \
  --out "$RUNTIME/artifacts/dureader-mini100k-v1/compact" --streaming
```

本次实测文档导入耗时 577.68 秒，compact 转换耗时 75.61 秒，最终词表为 1,231,998 项。两个步骤使用独立运行目录，未修改冻结语料。

## 可选加速依赖

生产词法回退路径只需要标准库。NumPy 加速和 Numba 解码使用隔离目录，不改写团队已有 venv。

```bash
uv pip install --python /path/to/serving-python \
  --target "$RUNTIME/python-deps" 'numpy==1.26.4' 'numba==0.60.0'
export PYTHONPATH="$RUNTIME/python-deps:$REPO"
```

`online.server` 在监听前完成 Numba 小块编译。`GET /datasets/official8m/info` 的 `retrieval.index_runtime` 会显示解码器、缓存预算、命中与淘汰统计。没有 NumPy 或 Numba 时服务继续走原实现。

## 启动与隧道

先按已有快照恢复文档启动 20k 与 8M Qdrant，并核对 collection 点数。8M 实例使用 `qdrant-official-tuned.yaml`，回环 HTTP 端口为 26333，工作线程为 8。

```bash
cd "$REPO"
deploy/start_local.sh "$RUNTIME" /path/to/serving-python
```

脚本启动三个 `online.server`、一个 `online.gateway` 和 SSH 反向隧道。PID 与日志写入 `$RUNTIME/logs`。`START_TUNNEL=0` 可只启动本机服务。脚本不自动恢复或覆盖 Qdrant 快照。

隧道模板位于 `deploy/systemd/nlp-search-site-tunnel.service`。安装用户服务前核对用户名、SSH alias 与密钥权限。远端转发只监听 `127.0.0.1:18081`，不会把本机服务暴露成新的公网端口。

```bash
install -Dm644 deploy/systemd/nlp-search-site-tunnel.service \
  ~/.config/systemd/user/nlp-search-site-tunnel.service
systemctl --user daemon-reload
systemctl --user enable --now nlp-search-site-tunnel.service
loginctl enable-linger "$USER"
```

`Restart=always` 会在 SSH 断开后每 5 秒重连，15 秒心跳用于发现失效连接。lingering 让用户服务在退出登录后继续运行。`deploy/start_local.sh` 检测到该单元后交给 systemd 管理，不会重复启动裸 SSH 进程。

## 发布前端

```bash
cd "$REPO"
deploy/aliyun/install_frontend.sh Aliyun
```

脚本要求 tracked 工作树干净，只打包当前 commit 中的 `web/` 与 Nginx 配置。远端以 commit SHA 建立 `/opt/nlp-search-engine/releases/$SHA`，先运行 `nginx -t`，再原子切换 `current` symlink 并 reload。旧 release 与更新前 Nginx 配置会保留。

回滚时把 `/opt/nlp-search-engine/current` 指回旧 release，恢复 `/opt/nlp-search-engine/operations` 中对应备份，再执行 `nginx -t && systemctl reload nginx`。本机检索服务无需随前端回滚而重建。

## 验收

```bash
curl -fsS http://127.0.0.1:18083/gateway/health
ssh Aliyun 'curl -fsS http://127.0.0.1:18081/gateway/health'
curl -fsS https://search.shwstone.cn/datasets

python3 -B -m unittest tests.test_gateway tests.test_serving_config
node --test tests/test_web.cjs
```

三个数据源都应各做一次 lexical 查询。20k 与 8M 再覆盖 dense、hybrid、adaptive，以及一条显式短语或布尔查询。检查响应的 `X-Search-Dataset`、实际执行模式、文档 ID、阶段耗时和错误语义。性能复验与原始收据见 [8M 在线调优记录](PERFORMANCE_TUNING.md)。
