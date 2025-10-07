# 📁 项目结构说明

## 重构后的项目结构

```
chemical-specifications-knowledge-base/
├── src/                          # 源代码目录
│   └── chemical_kb/              # 主包
│       ├── __init__.py           # 包初始化文件
│       ├── core/                 # 核心功能模块
│       │   ├── __init__.py
│       │   ├── pdf_processor.py  # PDF处理
│       │   ├── vector_store.py   # 向量存储
│       │   ├── pipeline.py       # 集成管道
│       │   └── search.py         # 搜索功能
│       ├── ai/                   # AI相关功能
│       │   ├── __init__.py
│       │   ├── service.py        # AI服务
│       │   ├── rag.py            # RAG管道
│       │   └── chat.py           # 聊天界面
│       ├── api/                  # API服务
│       │   ├── __init__.py
│       │   └── server.py         # API服务器
│       └── utils/                # 工具函数
│           ├── __init__.py
│           └── config.py         # 配置管理
├── config/                       # 配置文件目录
│   ├── ai_config.json           # AI模型配置
│   └── settings.py              # 系统设置
├── data/                         # 数据目录
│   ├── pdf/                      # PDF文件
│   ├── json/                     # 处理后的JSON文件
│   └── vector_db/                # 向量数据库
├── scripts/                      # 脚本文件
│   ├── cli.py                    # 命令行界面
│   └── test_integration.py      # 集成测试
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── test_ai.py               # AI功能测试
│   ├── test_api.py               # API测试
│   └── test_core.py              # 核心功能测试
├── docs/                         # 文档目录
│   ├── AI_README.md             # AI功能文档
│   ├── ARCHITECTURE.md          # 架构文档
│   └── api/                     # API文档
├── logs/                         # 日志文件目录
├── requirements.txt              # 依赖文件
├── setup.py                     # 安装脚本
├── README.md                    # 主文档
└── PROJECT_STRUCTURE.md         # 项目结构说明
```

## 重构说明

### 1. 目录结构优化

- **src/chemical_kb/**: 主包目录，包含所有核心功能
- **config/**: 配置文件集中管理
- **data/**: 数据文件统一存储
- **scripts/**: 可执行脚本
- **tests/**: 测试文件
- **docs/**: 文档文件

### 2. 文件重命名

| 原文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `chemical_pdf_processor.py` | `src/chemical_kb/core/pdf_processor.py` | PDF处理模块 |
| `vector.py` | `src/chemical_kb/core/vector_store.py` | 向量存储模块 |
| `integrated_pipeline.py` | `src/chemical_kb/core/pipeline.py` | 集成管道模块 |
| `ai_service.py` | `src/chemical_kb/ai/service.py` | AI服务模块 |
| `ai_chat.py` | `src/chemical_kb/ai/chat.py` | 聊天界面模块 |
| `api_server.py` | `src/chemical_kb/api/server.py` | API服务器模块 |
| `config_manager.py` | `src/chemical_kb/utils/config.py` | 配置管理模块 |

### 3. 导入路径更新

所有文件中的导入路径已更新为相对导入：

```python
# 原导入
from chemical_pdf_processor import ChemicalPDFProcessor
from vector import ChemicalVectorStore

# 新导入
from .pdf_processor import ChemicalPDFProcessor
from .vector_store import ChemicalVectorStore
```

### 4. 配置路径更新

- AI配置文件: `config/ai_config.json`
- PDF文件目录: `data/pdf/`
- 向量数据库: `data/vector_db/`
- 处理后的JSON: `data/json/`

### 5. 包结构

每个模块都有对应的 `__init__.py` 文件，支持：

```python
# 导入主包
from chemical_kb import IntegratedPipeline, AIService, RAGPipeline

# 导入特定模块
from chemical_kb.core import ChemicalPDFProcessor
from chemical_kb.ai import AIService
from chemical_kb.api import create_app
```

## 使用方法

### 1. 安装包

```bash
pip install -e .
```

### 2. 使用命令行工具

```bash
# 聊天界面
chemical-kb-chat

# API服务器
chemical-kb-api

# 配置管理
chemical-kb-config --setup

# 处理PDF
chemical-kb-pipeline
```

### 3. 使用Python API

```python
from chemical_kb import IntegratedPipeline, AIService, RAGPipeline

# 初始化管道
pipeline = IntegratedPipeline()

# 初始化AI服务
ai_service = AIService()

# 初始化RAG管道
rag = RAGPipeline(pipeline.vector_store, ai_service)

# 智能问答
result = rag.generate_answer("管道设计的基本要求是什么？")
```

### 4. 使用命令行界面

```bash
# 交互式聊天
python scripts/cli.py chat

# 启动API服务器
python scripts/cli.py api

# 运行测试
python scripts/cli.py test

# 配置管理
python scripts/cli.py config --setup

# 处理PDF
python scripts/cli.py process
```

## 优势

1. **模块化设计**: 功能清晰分离，易于维护
2. **标准化结构**: 符合Python包开发规范
3. **可安装性**: 支持pip安装和命令行工具
4. **可扩展性**: 易于添加新功能模块
5. **可测试性**: 独立的测试目录和测试框架
6. **文档化**: 完整的文档和说明

## 迁移指南

如果你之前使用的是旧版本，请按以下步骤迁移：

1. **更新导入语句**:
   ```python
   # 旧版本
   from chemical_pdf_processor import ChemicalPDFProcessor
   
   # 新版本
   from chemical_kb.core import ChemicalPDFProcessor
   ```

2. **更新文件路径**:
   - PDF文件移动到 `data/pdf/`
   - 配置文件移动到 `config/`
   - 向量数据库移动到 `data/vector_db/`

3. **使用新的启动方式**:
   ```bash
   # 旧版本
   python ai_chat.py
   
   # 新版本
   python scripts/cli.py chat
   # 或
   chemical-kb-chat
   ```

## 开发指南

### 添加新功能

1. 在相应的模块目录下创建新文件
2. 更新 `__init__.py` 文件
3. 添加相应的测试
4. 更新文档

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_core.py

# 运行集成测试
python scripts/test_integration.py
```

### 代码格式化

```bash
# 格式化代码
black src/

# 检查代码风格
flake8 src/

# 类型检查
mypy src/
```
