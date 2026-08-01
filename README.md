# Hermes Agent

Hermes 是一个可以长期运行、跨入口工作的个人 AI Agent。它把模型调用、工具、
Skill、记忆、自动化和消息网关放进同一个 Agent Core：你可以在终端里使用它，
也可以让它在服务器上运行，再从消息平台或桌面端继续同一条工作流。

| 入口 | 适合场景 |
| --- | --- |
| CLI / TUI | 交互式开发、运维和研究 |
| Dashboard / Desktop | 浏览器或桌面端的可视化对话 |
| Gateway | Telegram、Discord、Slack、WhatsApp、Signal 等消息平台 |
| ACP | 把 Hermes 接入 VS Code、Zed 或 JetBrains |

- **当前版本：** `0.19.1`
- **官方文档：** <https://hermes-agent.nousresearch.com/docs/>
- **上游仓库：** <https://github.com/NousResearch/hermes-agent>
- **本地定制仓库：** <https://github.com/ump45nose/hermes-agent>
- **许可证：** MIT，见 [LICENSE](LICENSE)

## 你可以用 Hermes 做什么？

- **多入口共用一个 Agent Core：** CLI、TUI、Dashboard、Desktop 和 Gateway 不各自维护一套聊天逻辑。
- **把能力放在边缘：** Toolset、Skill、插件和 MCP 可以扩展工具，而不必不断扩大核心工具 schema。
- **按 Profile 隔离：** 配置、凭据、会话、记忆、Skill、MCP OAuth 和网关状态可以按身份或工作空间分开。
- **记忆与学习闭环：** Episode 检索、脱敏记忆和 Skill 候选都有明确来源与边界，可以审核、回滚。
- **自动化与协作：** Cron 适合无人值守任务；Kanban 适合有 claim、租约、子任务和交付回执的长任务。
- **可替换模型与后端：** provider 插件支持 Nous Portal、OpenAI、OpenRouter 和自定义兼容端点。

## 运行要求

- Linux、macOS、WSL2、Termux 或 Windows；Windows 原生安装使用 PowerShell 安装器。
- Python `>=3.11,<3.14`（项目会拒绝 Python 3.14，以避免部分 Rust 依赖退回源码构建）。
- 源码开发推荐安装 [uv](https://docs.astral.sh/uv/)。TUI、Desktop、特定 provider 和消息平台会按需需要 Node.js 或额外依赖。
- 至少准备一个模型 provider 的凭据；密钥只放在 Hermes 的 Secret scope，不要写入 README、Prompt、日志或 Git。

## 2 分钟开始

### 安装发布版

Linux、macOS、WSL2 和 Termux：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc  # zsh 使用 source ~/.zshrc
hermes doctor
hermes setup
hermes
```

Windows 原生 PowerShell：

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
hermes doctor
hermes setup
hermes
```

安装器会准备 Hermes 运行时需要的 Python、Node 和 Git Bash 等依赖。平台差异和卸载方法见官方 [Getting Started](https://hermes-agent.nousresearch.com/docs/getting-started/)。

### 从源码运行

```bash
git clone https://github.com/ump45nose/hermes-agent.git
cd hermes-agent
uv sync --extra all

uv run hermes doctor
uv run hermes setup
uv run hermes
```

`--extra all` 只安装仓库声明的开发/常用 extras；重量级 provider、搜索、语音、消息平台和终端后端仍可能在首次启用时按需安装。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `hermes` | 启动经典交互式 CLI |
| `hermes --tui` | 启动 Ink TUI |
| `hermes dashboard --host 127.0.0.1 --no-open` | 启动仅监听本机的 Dashboard |
| `hermes model` | 选择 provider 和模型 |
| `hermes tools` | 配置 Toolset 与按需能力 |
| `hermes doctor` | 检查依赖、配置和运行环境 |
| `hermes setup` | 首次设置或重新配置 |
| `hermes mcp list` / `test` / `warm` | 查看、探测并预热 MCP 连接 |
| `hermes gateway setup` / `start` / `status` | 配置、启动和检查消息网关 |
| `hermes cron list` | 查看定时任务 |
| `hermes kanban list` | 查看跨 Profile 的持久任务 |
| `hermes profile list` | 查看可用 Profile |

每个子命令都支持 `hermes <command> --help`。`mcp test` 或健康检查只证明连接可用；验收外部服务时还要真实调用目标工具并确认客户端收到结果。

## Docker 部署

仓库自带 `docker-compose.yml`，适合在服务器上运行 Gateway 和本地 Dashboard：

```bash
HERMES_UID="$(id -u)" HERMES_GID="$(id -g)" docker compose build
HERMES_UID="$(id -u)" HERMES_GID="$(id -g)" docker compose up -d
docker compose ps
```

Compose 将宿主机 `~/.hermes` 映射到容器 `/opt/data`，并以宿主 UID/GID 创建文件。Gateway 使用 host network；Dashboard 默认只绑定 `127.0.0.1`。远程访问请使用 SSH 隧道（例如 `ssh -L 9119:127.0.0.1:9119 user@host`）或带认证的反向代理，**不要**用 `--insecure --host 0.0.0.0` 暴露 Dashboard。

如果要启用 OpenAI-compatible API server，必须同时配置 `API_SERVER_HOST` 和 `API_SERVER_KEY`，并先阅读 [API Server 文档](website/docs/user-guide/api-server.md)。不要覆盖镜像默认的 `/init` entrypoint；它负责权限、Profile reconcile 和服务监督。

## Profile、配置与密钥

| 内容 | 默认位置 | 说明 |
| --- | --- | --- |
| 行为配置 | `~/.hermes/config.yaml` | 模型、Toolset、Gateway、Cron、Profile 等非秘密设置 |
| Secret scope | `~/.hermes/.env` | API key、token、密码；不要存普通开关或路径 |
| 日志 | `~/.hermes/logs/` | `agent.log`、`errors.log`、`gateway.log` |
| Profile 状态 | `~/.hermes/profiles/<name>/` | 独立配置、会话、记忆、Skill、OAuth 和 Gateway 状态 |

```bash
hermes profile list
hermes -p lingjun doctor
hermes -p lingjun gateway status
hermes logs --follow
```

Profile 名称不是权限本身。实际边界由 toolset、Secret scope、审批策略、MCP allowlist 和运行身份共同决定。不要在 shell 中全局 `source` 某个 Profile 的 `.env`。

## Tool、Skill、插件与 MCP

Hermes 遵循“核心保持窄、能力放在边缘”的设计：

1. 先看现有 Tool 或命令能否解决问题；
2. 可复用的流程写成 Skill；
3. 第三方或特定领域能力放到插件；
4. 需要结构化外部服务时使用 MCP；
5. 只有真正基础且无法由终端、文件或 MCP 完成的能力才进入核心工具。

MCP 的最小流程如下（具体 URL 和授权方式由服务端决定）：

```bash
hermes mcp add <name> --url <https://example.invalid/mcp>
hermes mcp test <name>
hermes mcp warm <name>
```

Skill 和 Toolset 的切换通常在下一个会话生效，以保护长对话的 prompt cache；需要立即失效时使用命令提供的显式 `--now` 选项。

## 记忆、自动化与协作

- **Memory：** 当前对话可形成脱敏 Episode；检索只召回少量相关内容，稳定经验写入 Profile-local `MEMORY.md`/`USER.md` 前可由 owner 审核。
- **Cron：** 适合日报、备份、巡检和定时研究。用 `hermes cron list`、`hermes cron add --help` 管理。
- **Kanban：** 适合跨 Profile 的 claim、租约、子任务、附件和终态交付。用 `hermes kanban --help` 查看完整命令。
- **Gateway：** 调度成功、工具执行成功、任务完成和消息送达是四个不同状态，排障时要分别核对。

## 安全基线

- 行为设置放 `config.yaml`；`.env` 只放 API key、token、密码等 secret。
- Dashboard 默认只监听本机；API server 对外开放时必须配置 key 并放在认证反代之后。
- 默认拒绝私有 URL、启用 secret redaction 和命令安全检查；发送文件时会拒绝 `/etc`、`/proc`、`~/.ssh`、`~/.aws`、`~/.hermes/.env`、`auth.json` 等敏感路径。
- 为不同身份使用不同 Profile；不要用“同一个 OS 用户能读到文件”代替 Profile ACL。
- 不要把 API key 复制到命令参数、日志、Prompt、Memory、Cron 定义、Issue 或提交信息。

## 开发与验证

```bash
uv sync --extra all
uv run hermes --help
uv run hermes doctor
```

修改核心代码时，先做与改动直接相关的静态、导入/编译或单路径 smoke 验证。需要运行测试时，使用仓库提供的 `scripts/run_tests.sh`，并尽量限定到目标文件或目录：

```bash
scripts/run_tests.sh tests/<target>.py -k <case>
```

核心代码按职责分层：`run_agent.py` / `model_tools.py` 负责 Agent loop 和工具编排；`agent/` 负责上下文与 provider；`gateway/` 负责消息平台；`hermes_cli/` 负责 CLI、Profile、Cron、Kanban 和 Dashboard；`tools/`、`plugins/`、`skills/` 负责扩展；`ui-tui/` 和 `apps/desktop/` 负责交互界面。

## 排障入口

1. 先运行 `hermes doctor`，确认实际使用的 Profile、Python 和 provider；
2. 查看 `hermes logs --follow`，区分配置未加载、凭据缺失、工具不可见和工具执行失败；
3. MCP 连接先 `hermes mcp test`，再真实调用目标工具；
4. Gateway 需要分别确认进程、会话、工具结果和消息送达；
5. Dashboard 远程访问优先检查 SSH 隧道或反代认证，不要直接改成公网无认证监听。

## 相关链接

- [官方文档](https://hermes-agent.nousresearch.com/docs/)
- [本地定制仓库](https://github.com/ump45nose/hermes-agent)
- [上游 Issue](https://github.com/NousResearch/hermes-agent/issues)
- [本地 Issue](https://github.com/ump45nose/hermes-agent/issues)
- [贡献指南](CONTRIBUTING.md)

## 许可证

本项目按 [MIT License](LICENSE) 发布。
