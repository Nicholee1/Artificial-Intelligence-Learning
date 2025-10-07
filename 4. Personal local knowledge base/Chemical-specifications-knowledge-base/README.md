# 🔬 Chemical Specifications Knowledge Base

一个专业的化工技术文档知识库系统，集成了PDF文档处理、向量化存储、AI智能问答等功能。

## ✨ 主要特性

- 📄 **PDF文档处理**: 自动提取和结构化PDF内容
- 🔍 **向量化搜索**: 基于语义的智能文档检索
- 🤖 **AI智能问答**: 支持多种AI模型的问答系统
- 🌐 **Web API**: 完整的REST API接口
- 💬 **交互式聊天**: 命令行聊天界面
- ⚙️ **灵活配置**: 支持多种AI模型和配置选项

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repository-url>
cd chemical-specifications-knowledge-base

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 2. 配置AI模型

```bash
# 交互式配置
python scripts/cli.py config --setup

# 或手动编辑配置文件
vim config/ai_config.json
```

### 3. 处理PDF文档

```bash
# 将PDF文件放入 data/pdf/ 目录
cp your_documents.pdf data/pdf/

# 处理文档
python scripts/cli.py process
```

### 4. 开始使用

```bash
# 启动交互式聊天
python scripts/cli.py chat

# 启动Web API服务器
python scripts/cli.py api

# 运行测试
python scripts/cli.py test
```

## 📁 项目结构

```
chemical-specifications-knowledge-base/
├── src/chemical_kb/          # 主要源代码
│   ├── core/                 # 核心功能模块
│   ├── ai/                   # AI相关功能
│   ├── api/                  # Web API服务
│   └── utils/                # 工具函数
├── config/                   # 配置文件
├── data/                     # 数据目录
│   ├── pdf/                  # PDF文件
│   ├── json/                 # 处理后的JSON
│   └── vector_db/            # 向量数据库
├── scripts/                  # 可执行脚本
│   ├── cli.py               # 命令行界面
│   └── test_integration.py   # 集成测试
├── tests/                    # 测试文件
├── docs/                     # 文档
└── logs/                     # 日志文件
```

## 🛠️ 使用方法

### 命令行界面

```bash
# 查看帮助
python scripts/cli.py --help

# 交互式聊天
python scripts/cli.py chat

# 单次问答
python scripts/cli.py chat -q "管道设计的基本要求是什么？"

# 启动API服务器
python scripts/cli.py api --port 8000

# 配置管理
python scripts/cli.py config --list
python scripts/cli.py config --test ollama

# 处理PDF文档
python scripts/cli.py process --force

# 运行测试
python scripts/cli.py test
python scripts/cli.py test --integration
```

### Python API

```python
from chemical_kb import IntegratedPipeline, AIService, RAGPipeline

# 初始化管道
pipeline = IntegratedPipeline()

# 初始化AI服务
ai_service = AIService()

# 创建RAG管道
rag = RAGPipeline(pipeline.vector_store, ai_service)

# 智能问答
result = rag.generate_answer("管道设计的基本要求是什么？")
print(result['answer'])
```

### Web API

启动API服务器后，访问 http://localhost:5000 查看API文档。

主要端点：
- `GET /api/health` - 健康检查
- `POST /api/ask` - AI问答
- `GET /api/search` - 文档搜索
- `GET /api/documents` - 文档信息
- `GET /api/providers` - AI模型列表

## 🤖 支持的AI模型

| 提供者 | 模型示例 | 配置要求 |
|--------|----------|----------|
| OpenAI | gpt-3.5-turbo, gpt-4 | API密钥 |
| Claude | claude-3-sonnet | API密钥 |
| Ollama | llama3.1:8b, qwen2 | 本地服务 |
| 通义千问 | qwen-turbo | API密钥 |
| DeepSeek | deepseek-chat | API密钥 |

## ⚙️ 配置说明

### AI模型配置 (config/ai_config.json)

```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "type": "ollama",
      "base_url": "http://localhost:11434",
      "model": "llama3.1:8b",
      "enabled": true
    }
  }
}
```

### 环境变量

- `CHEMICAL_KB_ENV`: 环境模式 (development/production)
- `OPENAI_API_KEY`: OpenAI API密钥
- `ANTHROPIC_API_KEY`: Claude API密钥

## 🧪 测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python scripts/cli.py test --integration

# 运行特定测试
python -m pytest tests/test_core.py
```

## 📖 文档

- [项目架构](docs/ARCHITECTURE.md)
- [AI功能详解](docs/AI_README.md)
- [项目结构说明](PROJECT_STRUCTURE.md)

## 🔧 开发指南

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 代码格式化

```bash
black src/
flake8 src/
mypy src/
```

### 添加新功能

1. 在相应模块目录创建新文件
2. 更新 `__init__.py` 文件
3. 添加测试用例
4. 更新文档

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如遇问题，请：

1. 查看文档和FAQ
2. 运行 `python scripts/cli.py test --integration` 诊断问题
3. 提交 Issue 描述问题

## 🎯 路线图

- [ ] 支持更多文档格式 (Word, Excel, PPT)
- [ ] 多语言支持
- [ ] 实时协作功能
- [ ] 移动端应用
- [ ] 企业级部署方案

---

**Chemical Specifications Knowledge Base** - 让化工技术文档管理更智能！