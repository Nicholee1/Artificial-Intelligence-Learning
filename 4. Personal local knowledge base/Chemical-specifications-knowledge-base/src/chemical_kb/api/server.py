#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REST API服务器
提供化工文档知识库的Web服务接口
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from typing import Dict, Any
import logging
from datetime import datetime

from ..core.pipeline import IntegratedPipeline
from ..ai.service import AIService
from ..ai.rag import RAGPipeline

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求
    
    # 全局变量
    pipeline = None
    ai_service = None
    rag_pipeline = None

    def initialize_services():
        """初始化服务"""
        nonlocal pipeline, ai_service, rag_pipeline
        
        try:
            # 初始化集成管道
            pipeline = IntegratedPipeline()
            logger.info("集成管道初始化成功")
            
            # 初始化AI服务
            ai_service = AIService()
            logger.info("AI服务初始化成功")
            
            # 初始化RAG管道
            rag_pipeline = RAGPipeline(pipeline.vector_store, ai_service)
            logger.info("RAG管道初始化成功")
            
            return True
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            return False

    @app.route('/')
    def index():
        """主页"""
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>化工文档知识库API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .api-section { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
                .endpoint { margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #3498db; }
                .method { font-weight: bold; color: #e74c3c; }
                .url { font-family: monospace; background: #ecf0f1; padding: 2px 5px; border-radius: 3px; }
                .description { color: #7f8c8d; margin-top: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 化工文档知识库API</h1>
                
                <div class="api-section">
                    <h2>📚 文档管理</h2>
                    <div class="endpoint">
                        <span class="method">GET</span> <span class="url">/api/documents</span>
                        <div class="description">获取文档列表和数据库信息</div>
                    </div>
                    <div class="endpoint">
                        <span class="method">POST</span> <span class="url">/api/documents/process</span>
                        <div class="description">处理PDF文档并添加到知识库</div>
                    </div>
                </div>
                
                <div class="api-section">
                    <h2>🔍 搜索功能</h2>
                    <div class="endpoint">
                        <span class="method">GET</span> <span class="url">/api/search</span>
                        <div class="description">搜索相关文档内容</div>
                    </div>
                </div>
                
                <div class="api-section">
                    <h2>🤖 AI问答</h2>
                    <div class="endpoint">
                        <span class="method">POST</span> <span class="url">/api/ask</span>
                        <div class="description">基于文档内容的智能问答</div>
                    </div>
                    <div class="endpoint">
                        <span class="method">GET</span> <span class="url">/api/providers</span>
                        <div class="description">获取可用的AI模型提供者</div>
                    </div>
                </div>
                
                <div class="api-section">
                    <h2>⚙️ 系统管理</h2>
                    <div class="endpoint">
                        <span class="method">GET</span> <span class="url">/api/health</span>
                        <div class="description">检查系统健康状态</div>
                    </div>
                    <div class="endpoint">
                        <span class="method">GET</span> <span class="url">/api/config</span>
                        <div class="description">获取当前配置信息</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

    @app.route('/api/health')
    def health_check():
        """健康检查"""
        try:
            # 检查向量数据库
            db_info = pipeline.get_database_info() if pipeline else {}
            
            # 检查AI服务
            ai_providers = ai_service.get_available_providers() if ai_service else []
            
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": {
                    "status": "connected" if db_info else "disconnected",
                    "document_count": db_info.get('document_count', 0)
                },
                "ai_service": {
                    "status": "available" if ai_providers else "unavailable",
                    "providers": ai_providers
                }
            })
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500

    @app.route('/api/documents')
    def get_documents():
        """获取文档信息"""
        try:
            if not pipeline:
                return jsonify({"error": "管道未初始化"}), 500
            
            db_info = pipeline.get_database_info()
            return jsonify({
                "status": "success",
                "data": db_info,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/documents/process', methods=['POST'])
    def process_documents():
        """处理PDF文档"""
        try:
            if not pipeline:
                return jsonify({"error": "管道未初始化"}), 500
            
            data = request.get_json() or {}
            force_reprocess = data.get('force', False)
            
            # 处理所有PDF文件
            results = pipeline.process_all_pdfs(force_reprocess=force_reprocess)
            
            # 统计结果
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            return jsonify({
                "status": "success",
                "message": f"处理完成: {success_count}/{total_count} 个文件成功",
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/search')
    def search_documents():
        """搜索文档"""
        try:
            if not pipeline:
                return jsonify({"error": "管道未初始化"}), 500
            
            query = request.args.get('q', '')
            n_results = int(request.args.get('n', 5))
            doc_type = request.args.get('type')
            
            if not query:
                return jsonify({"error": "查询参数不能为空"}), 400
            
            results = pipeline.search_documents(query, n_results, doc_type)
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result['content'],
                    "metadata": result['metadata'],
                    "similarity": 1 - result['distance'],
                    "id": result['id']
                })
            
            return jsonify({
                "status": "success",
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/ask', methods=['POST'])
    def ask_question():
        """AI问答"""
        try:
            if not rag_pipeline:
                return jsonify({"error": "RAG管道未初始化"}), 500
            
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({"error": "缺少问题参数"}), 400
            
            question = data['question']
            n_context = data.get('n_context', 3)
            provider = data.get('provider')
            max_tokens = data.get('max_tokens', 1000)
            temperature = data.get('temperature', 0.7)
            
            # 生成回答
            result = rag_pipeline.generate_answer(
                query=question,
                n_context=n_context,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return jsonify({
                "status": "success",
                "question": question,
                "answer": result['answer'],
                "sources": result['sources'],
                "context_count": len(result['context']),
                "timestamp": result['timestamp']
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/providers')
    def get_providers():
        """获取可用的AI提供者"""
        try:
            if not ai_service:
                return jsonify({"error": "AI服务未初始化"}), 500
            
            providers = ai_service.get_available_providers()
            default_provider = ai_service.default_provider
            
            # 测试每个提供者
            provider_status = {}
            for provider in providers:
                provider_status[provider] = {
                    "available": ai_service.test_provider(provider),
                    "is_default": provider == default_provider
                }
            
            return jsonify({
                "status": "success",
                "providers": provider_status,
                "default": default_provider,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/config')
    def get_config():
        """获取配置信息"""
        try:
            config_file = "config/ai_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            return jsonify({
                "status": "success",
                "config": config,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        """404错误处理"""
        return jsonify({"error": "接口不存在"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """500错误处理"""
        return jsonify({"error": "服务器内部错误"}), 500

    # 初始化服务
    initialize_services()
    
    return app

def main():
    """主函数"""
    print("🚀 启动化工文档知识库API服务器")
    print("=" * 50)
    
    app = create_app()
    
    print("✅ 服务初始化成功")
    print("📚 文档管理: 已就绪")
    print("🔍 搜索功能: 已就绪")
    print("🤖 AI问答: 已就绪")
    
    print("\n🌐 API服务器启动中...")
    print("访问 http://localhost:5000 查看API文档")
    
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()