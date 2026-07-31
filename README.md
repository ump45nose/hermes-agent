# Hermes Agent

一个可以长期运行、跨入口工作的个人 AI Agent。Hermes 把模型调用、工具、技能、
记忆、定时任务和消息网关放在同一个 Agent Core 里：你可以在终端里使用它，也
可以让它在服务器上运行，再从 Telegram、Discord、Slack 等平台继续对话。

- 官方文档：<https://hermes-agent.nousresearch.com/docs/>
- 上游项目：<https://github.com/NousResearch/hermes-agent>
- 本地定制仓库：<https://github.com/ump45nose/hermes-agent>
- 许可证：MIT（见 [LICENSE](LICENSE)）

## Hermes 适合做什么？

Hermes 不只是一次性问答脚本。它更适合需要连续上下文、真实工具和无人值守
流程的工作：

| 能力 | 通俗解释 |
| --- | --- |
| 多种入口 | CLI、TUI、Dashboard、桌面端和消息网关共用同一个 Agent Core。 |
| 任意模型 | 通过 provider 插件连接 Nous Portal、OpenAI、OpenRouter 或自定义兼容端点，用 `hermes model` 切换。 |
| 工具和技能 | 工具负责执行，Skill 负责保存可重复的做事方法；MCP 可以把外部服务接进来。 |
| Profile 隔离 | 每个 Profile 有自己的配置、凭据、会话、记忆、Skill 和 Gateway，适合把不同身份或工作空间分开。 |
| 记忆闭环 | 已完成的对话可提炼为脱敏 Episode，按 Profile 检索；长期知识写入 `MEMORY.md`/`USER.md` 前还可以经过 owner 审核。 |
| 定时与协作 | Cron 负责无人值守任务，Kanban 负责跨 Profile 的持久任务和交付回执。 |
| 安全边界 | Secret scope、命令审批、工具集和 MCP 权限共同限制“谁能在什么场景调用什么”。 |

## 2 分钟开始

### 直接安装发布版

Linux、macOS、WSL2 和 Termux 可以使用官方安装器：

~~~bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc                 # 或 source ~/.zshrc
hermes doctor
hermes setup
hermes
~~~

Windows 原生 PowerShell：

~~~powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
hermes doctor
hermes setup
hermes
~~~

安装器会准备 Hermes 所需的 Python/Node/Git Bash 等运行依赖。Termux 和 Windows
的特殊说明见官方 [Getting Started](https://hermes-agent.nousresearch.com/docs/getting-started/)。

### 从本仓库开发

如果你要运行本地定制版本：

~~~bash
git clone https://github.com/ump45nose/hermes-agent.git
cd hermes-agent

# 需要先安装 uv：https://docs.astral.sh/uv/
uv sync --extra all

uv run hermes doctor
uv run hermes setup
uv run hermes
~~~

`[all]` 是一组适合常规开发的依赖，不代表强行安装所有 provider。许多重量级
模型、搜索、语音、消息平台和终端后端会在你第一次启用时按需安装，减少初始体积
和供应链风险。

## 常用入口

~~~bash
hermes                  # 交互式 CLI
hermes --tui            # Ink TUI
hermes model            # 选择模型/provider
hermes tools            # 配置工具集和按需能力
hermes doctor           # 检查配置与依赖
hermes setup            # 首次设置或重新配置
hermes update           # 更新安装
~~~

桌面端和浏览器 Dashboard 使用同一套 Agent Core；它们是不同的 UI 入口，不需要
复制一套聊天逻辑。

## 连接消息平台

网关把同一个 Agent 接到 Telegram、Discord、Slack、WhatsApp、Signal 以及其他
适配器。典型流程是：

~~~bash
hermes gateway setup
hermes gateway start
hermes gateway status
~~~

首次配置时按向导填写平台凭据和允许的用户。凭据只放在 Secret scope 管理的
`.env`/Profile 目录中，不要写进 Prompt、Skill、日志或 Git。

## Profile：把身份和工作空间分开

Profile 是完整的 Hermes 实例。它们共享代码，但分别保存：

- `config.yaml`：行为、模型、工具集和调度设置；
- `.env`：该 Profile 或根环境需要的密钥；
- 会话数据库、`MEMORY.md`、`USER.md`、Skill 和 Gateway 状态；
- MCP OAuth 状态与 Profile 级访问边界。

常用检查：

~~~bash
hermes profile list
hermes -p lingjun doctor
hermes -p lingjun gateway status
~~~

Profile 名称不是权限本身。真正的边界由配置的 toolset、Secret scope、审批策略、
MCP allowlist 和运行身份共同决定。

## 工具、Skill 与 MCP

Hermes 的设计是“核心保持窄，能力放在边缘”：

- 已启用的 Toolset 决定一个 Profile 真正能调用什么；
- Skill 保存稳定流程，不把一次性结果硬塞进系统提示；
- Tool Search 可以按需展开已授权工具的 schema，避免每轮都发送全部工具；
- MCP 用于接入外部的搜索、文档、浏览器或业务服务；
- 插件可以增加工具、CLI 子命令、模型 provider 或记忆 provider，而不必修改 Agent Core。

MCP 示例命令（具体 URL 和授权方式由服务端决定）：

~~~bash
hermes mcp list
hermes mcp add <name> --url <https://example.invalid/mcp>
hermes mcp test <name>
hermes mcp warm <name>
~~~

不要把 `hermes mcp test`、服务健康或工具发现当成业务结果；真正验收还要执行
一次目标工具并确认客户端收到结果。

## 记忆和学习闭环

Hermes 将“当前对话”和“长期知识”分开：

1. 完整的 user → final assistant turn 可以进入本 Profile 的 Episode 队列；
2. 提炼输入只包含用户/助手正文和脱敏的确定性 Tool receipt，不包含凭据、
   system prompt、reasoning 或原始 tool result；
3. FTS5/BM25（不可用时使用有界回退）按需召回少量 Episode 到当前请求；
4. 每日任务可以把有证据的稳定经验归纳到 Profile-local `MEMORY.md`/`USER.md`；
5. Skill/Tool 候选必须满足独立 Episode 阈值，并由有权限的 owner 审核后才应用。

这让“记住你的工作方式”有来源、有边界、可回滚，而不是把所有聊天内容永久
复制到每一次 Prompt。

## 自动化与协作

Cron 适合日报、备份、巡检和定时研究：

~~~bash
hermes cron list
hermes cron status
~~~

Kanban 适合需要 claim、租约、子任务、附件和终态交付的长任务：

~~~bash
hermes kanban --help
hermes kanban list
~~~

调度成功、工具执行成功、任务完成和消息送达是四个不同状态；排障时应分别
核对它们。

## 配置和安全建议

- 行为设置放在 `config.yaml`，`.env` 只放 API key、token、密码等 secret；
- 不要在命令参数、日志、Prompt、Memory、Cron 定义或任务正文中复制凭据；
- 为不同身份使用不同 Profile，不要用“同一个 OS 用户能读到文件”代替 Profile ACL；
- 新增工具前先判断是否可以用现有 Tool、Skill、插件或 MCP 解决；
- 修改 Prompt、Toolset 或 Memory 的命令应遵守缓存边界，通常下一个会话才生效；
- 动态基础设施事实属于运行时 shared-state，不要写进 SOUL、USER 或长期 Memory。

## 开发者入口

核心代码按职责分层：

~~~text
run_agent.py / model_tools.py  Agent loop 与工具编排
agent/                       Prompt、上下文、provider、Episode、运行时策略
gateway/                     消息平台与长连接生命周期
hermes_cli/                  CLI、Setup、Profile、Cron、Kanban、Dashboard
tools/                       工具实现与注册表
plugins/                     provider、memory、platform 和其他扩展
skills/                      内置 Skill
optional-skills/             默认不启用的重型/小众 Skill
ui-tui/                      Ink TUI
apps/desktop/                Electron 桌面端
hermes_state.py              SQLite 会话、Episode 和学习候选状态
~~~

开发环境：

~~~bash
uv sync --extra all
uv run hermes --help
uv run hermes doctor
~~~

本地定制仓库默认采用与改动直接相关的静态、编译/导入或单路径 smoke 验证；
不要把大范围回归套件当成每次修改的默认步骤。

## 获取帮助

- CLI 内置帮助：`hermes --help`、`hermes <command> --help`
- 官方文档：<https://hermes-agent.nousresearch.com/docs/>
- Issue：<https://github.com/ump45nose/hermes-agent/issues>
- 上游 Issue：<https://github.com/NousResearch/hermes-agent/issues>

如果遇到“能启动但没有结果”，请按顺序检查：配置是否加载、凭据是否在正确
Profile、目标工具是否真的可见、工具是否真实执行、结果是否被目标客户端收到。
