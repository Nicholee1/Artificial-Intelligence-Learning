
## Basic
### MCP
 **MCP（Model Context Protocol）** ：是由 Anthropic 于 2024 年 11 月开源发布的 AI 大模型标准化工具箱。它是一个开放协议，能统一 LLM 应用与外部数据源和工具间的通信协议，为 AI 开发给予标准化的上下文交互方式。MCP 可作为 AI 和外部工具之间的桥梁，借助标准化协议，自动替代人类去访问和操作浏览器、文件系统、Git 仓库等外部工具，使大模型可通过标准输入输出与 MCP Server 交流并调用，进而获取外部数据来完成任务。

### ADK
Agent Development Kit：提供代码优先的开发框架，支持直接定义代理行为、工具调用和编排逻辑；强调多代理协作，允许代理间动态交接任务；集成 Google Cloud 服务，支持从本地到云端的无缝部署。具备多代理架构，可通过组合专业代理构建模块化应用；拥有工具生态系统，支持预置工具或自定义 Python 函数；提供流程追踪功能，可可视化代理运行轨迹，便于调试优化；还设有安全护栏，对输入 / 输出进行规则验证，避免异常行为。

相似的竞品：
OpenAI Agents SDK/AutoGen/ LangChain/CrewAI/ AgentKit

在本项目中，要做好一道菜（用户的需求），LLM提供了厨师的能力，MCP提供了厨师统一的标准的工具，ADK就相当于厨师的大脑。
## Environment Setup
 
### 安装HomeBrew 
1. 安装：
``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"``
2. 配置环境变量：
```shell
echo >> /Users/<username>/.zprofile

echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/<username>/.zprofile

eval "$(/opt/homebrew/bin/brew shellenv)"
```
3. 验证：
   `brew version`
### 安装pyenv
`brew install pyenv-virtualenv`

配置环境变量：
`vi ~/.zshrc`
```bash
# 配置 pyenv
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# 配置 pyenv-virtualenv（让虚拟环境激活命令生效）
eval "$(pyenv virtualenv-init -)"
```
配置生效：
`source ~/.zshrc`
### 安装python 3.12.10
`pyenv install 3.12.10`





