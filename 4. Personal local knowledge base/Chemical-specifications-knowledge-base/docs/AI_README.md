# 🤖 AI模型集成使用指南

本项目现已集成AI模型API，提供智能问答和文档分析功能。

## 📋 功能特性

- **多模型支持**: OpenAI、Claude、Ollama、通义千问、DeepSeek等
- **RAG架构**: 检索增强生成，结合向量搜索和AI生成
- **REST API**: 提供Web服务接口
- **交互式聊天**: 命令行聊天界面
- **配置管理**: 灵活的模型配置和管理

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置AI模型

#### 方法一：交互式配置
```bash
python config_manager.py --setup
```

#### 方法二：手动配置
编辑 `ai_config.json` 文件，设置API密钥和模型参数：

```json
{
  "default_provider": "openai",
  "providers": {
    "openai": {
      "type": "openai",
      "api_key": "your-api-key-here",
      "model": "gpt-3.5-turbo",
      "enabled": true
    }
  }
}
```

### 3. 处理PDF文档

```bash
python integrated_pipeline.py
```

### 4. 开始使用

#### 交互式聊天
```bash
python ai_chat.py
```

#### 启动Web API
```bash
python api_server.py
```

#### 运行测试
```bash
python test_ai_integration.py
```

## 🔧 配置说明

### 支持的AI模型

| 提供者 | 类型 | 模型示例 | 配置项 |
|--------|------|----------|--------|
| OpenAI | openai | gpt-3.5-turbo, gpt-4 | api_key, model, base_url |
| Claude | claude | claude-3-sonnet-20240229 | api_key, model |
| Ollama | ollama | llama2, codellama | base_url, model |
| 通义千问 | openai | qwen-turbo | api_key, model, base_url |
| DeepSeek | openai | deepseek-chat | api_key, model, base_url |

### 配置示例

#### OpenAI配置
```json
{
  "openai": {
    "type": "openai",
    "api_key": "sk-...",
    "model": "gpt-3.5-turbo",
    "base_url": null,
    "enabled": true
  }
}
```

#### 通义千问配置
```json
{
  "qwen": {
    "type": "openai",
    "api_key": "sk-...",
    "model": "qwen-turbo",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "enabled": true
  }
}
```

#### 本地Ollama配置
```json
{
  "ollama": {
    "type": "ollama",
    "base_url": "http://localhost:11434",
    "model": "llama2",
    "enabled": true
  }
}
```

## 🎯 使用方式

### 1. 命令行聊天界面

```bash
python ai_chat.py
```

支持的命令：
- `/help` - 显示帮助
- `/providers` - 查看可用模型
- `/switch <模型名>` - 切换模型
- `/search <关键词>` - 搜索文档
- `/history` - 查看对话历史
- `/quit` - 退出

### 2. Web API服务

启动服务器：
```bash
python api_server.py
```

访问 http://localhost:5000 查看API文档

#### 主要API端点

- `GET /api/health` - 健康检查
- `GET /api/documents` - 获取文档信息
- `POST /api/ask` - AI问答
- `GET /api/search` - 文档搜索
- `GET /api/providers` - 获取可用模型

#### API使用示例

```bash
# 健康检查
curl http://localhost:5000/api/health

# AI问答
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "管道设计的基本要求是什么？"}'

# 文档搜索
curl "http://localhost:5000/api/search?q=管道设计&n=3"
```

### 3. 编程接口

```python
from integrated_pipeline import IntegratedPipeline
from ai_service import AIService, RAGPipeline

# 初始化服务
pipeline = IntegratedPipeline()
ai_service = AIService()
rag_pipeline = RAGPipeline(pipeline.vector_store, ai_service)

# 智能问答
result = rag_pipeline.generate_answer("管道设计的基本要求是什么？")
print(result['answer'])
```

## 🛠️ 配置管理工具

### 查看配置
```bash
python config_manager.py --list
```

### 测试模型
```bash
python config_manager.py --test openai
```

### 启用/禁用模型
```bash
python config_manager.py --enable openai
python config_manager.py --disable claude
```

### 设置默认模型
```bash
python config_manager.py --default openai
```

### 设置配置项
```bash
python config_manager.py --set openai api_key "sk-..."
python config_manager.py --set openai model "gpt-4"
```

## 🧪 测试和调试

### 运行综合测试
```bash
python test_ai_integration.py
```

测试内容包括：
- 数据库状态检查
- AI提供者可用性测试
- 向量搜索功能测试
- RAG生成功能测试
- API端点测试

### 测试特定功能

```python
# 测试AI服务
from ai_service import AIService
ai_service = AIService()
print(ai_service.get_available_providers())

# 测试RAG管道
from ai_service import RAGPipeline
rag = RAGPipeline(vector_store, ai_service)
result = rag.generate_answer("测试问题")
```

## 🔍 故障排除

### 常见问题

1. **没有可用的AI提供者**
   - 检查 `ai_config.json` 配置
   - 确保至少有一个提供者被启用
   - 验证API密钥是否正确

2. **AI模型测试失败**
   - 检查网络连接
   - 验证API密钥和模型名称
   - 检查API配额和限制

3. **RAG生成质量差**
   - 确保知识库中有相关文档
   - 调整检索参数（n_context）
   - 尝试不同的AI模型

4. **向量搜索无结果**
   - 运行 `python integrated_pipeline.py` 处理PDF
   - 检查向量数据库是否包含文档

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 高级功能

### 自定义提示词

```python
result = rag_pipeline.generate_answer(
    query="问题",
    system_prompt="你是一个专业的化工工程师...",
    max_tokens=2000,
    temperature=0.7
)
```

### 批量处理

```python
questions = ["问题1", "问题2", "问题3"]
for question in questions:
    result = rag_pipeline.generate_answer(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
```

### 自定义检索参数

```python
result = rag_pipeline.generate_answer(
    query="问题",
    n_context=5,  # 检索更多上下文
    provider="claude"  # 指定特定模型
)
```

## 🔒 安全注意事项

1. **API密钥安全**
   - 不要在代码中硬编码API密钥
   - 使用环境变量或配置文件
   - 定期轮换API密钥

2. **数据隐私**
   - 敏感文档建议使用本地模型
   - 注意API服务商的数据使用政策

3. **访问控制**
   - 在生产环境中添加身份验证
   - 限制API访问频率

## 📈 性能优化

1. **模型选择**
   - 根据需求选择合适的模型
   - 平衡性能和成本

2. **缓存策略**
   - 对常见问题实现缓存
   - 减少重复的API调用

3. **批处理**
   - 批量处理多个问题
   - 提高处理效率

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 运行测试脚本诊断问题
3. 提交 Issue 描述问题
