#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI智能问答交互界面
基于RAG的化工文档问答系统
"""

import os
import json
import argparse
from typing import Dict, Any, List
from ..core.pipeline import IntegratedPipeline
from .service import AIService
from .rag import RAGPipeline

class AIChatInterface:
    """AI聊天界面"""
    
    def __init__(self):
        self.pipeline = None
        self.ai_service = None
        self.rag_pipeline = None
        self.chat_history = []
        self.initialize_services()
    
    def initialize_services(self):
        """初始化服务"""
        try:
            print("🔄 正在初始化服务...")
            
            # 初始化集成管道
            self.pipeline = IntegratedPipeline()
            print("✅ 集成管道初始化成功")
            
            # 初始化AI服务
            self.ai_service = AIService()
            print("✅ AI服务初始化成功")
            
            # 初始化RAG管道
            self.rag_pipeline = RAGPipeline(self.pipeline.vector_store, self.ai_service)
            print("✅ RAG管道初始化成功")
            
            return True
        except Exception as e:
            print(f"❌ 服务初始化失败: {e}")
            return False
    
    def display_welcome(self):
        """显示欢迎信息"""
        print("\n" + "="*60)
        print("🤖 化工文档智能问答系统")
        print("="*60)
        
        # 显示数据库信息
        if self.pipeline:
            db_info = self.pipeline.get_database_info()
            doc_count = db_info.get('document_count', 0)
            print(f"📚 知识库文档数量: {doc_count}")
            
            if doc_count == 0:
                print("⚠️  知识库为空，请先运行 integrated_pipeline.py 处理PDF文档")
                return False
        
        # 显示AI提供者信息
        if self.ai_service:
            providers = self.ai_service.get_available_providers()
            default_provider = self.ai_service.default_provider
            
            if providers:
                print(f"🤖 可用AI模型: {', '.join(providers)}")
                print(f"🎯 默认模型: {default_provider}")
            else:
                print("⚠️  没有可用的AI模型，请配置 ai_config.json")
                return False
        
        print("\n💡 使用说明:")
        print("  - 直接输入问题开始对话")
        print("  - 输入 '/help' 查看帮助")
        print("  - 输入 '/providers' 查看AI模型")
        print("  - 输入 '/search <关键词>' 进行文档搜索")
        print("  - 输入 '/history' 查看对话历史")
        print("  - 输入 '/quit' 退出")
        print("="*60)
        
        return True
    
    def display_help(self):
        """显示帮助信息"""
        help_text = """
🔧 命令帮助:

基础命令:
  /help          - 显示此帮助信息
  /quit, /exit   - 退出程序
  /clear         - 清空对话历史
  /history       - 显示对话历史

AI模型管理:
  /providers     - 显示可用的AI模型
  /switch <模型名> - 切换AI模型
  /test <模型名>  - 测试AI模型

文档搜索:
  /search <关键词> - 搜索相关文档
  /docs          - 显示知识库信息

高级功能:
  /ask <问题>    - 使用指定参数提问
  /config        - 显示当前配置
  /reload        - 重新加载配置
        """
        print(help_text)
    
    def display_providers(self):
        """显示AI提供者信息"""
        if not self.ai_service:
            print("❌ AI服务未初始化")
            return
        
        providers = self.ai_service.get_available_providers()
        default_provider = self.ai_service.default_provider
        
        print("\n🤖 可用AI模型:")
        print("-" * 40)
        
        for provider in providers:
            status = "✅ 可用" if self.ai_service.test_provider(provider) else "❌ 不可用"
            default_mark = " (默认)" if provider == default_provider else ""
            print(f"  {provider}{default_mark}: {status}")
        
        if not providers:
            print("  ❌ 没有可用的AI模型")
            print("  💡 请配置 ai_config.json 文件")
    
    def switch_provider(self, provider_name: str):
        """切换AI提供者"""
        if not self.ai_service:
            print("❌ AI服务未初始化")
            return
        
        providers = self.ai_service.get_available_providers()
        if provider_name not in providers:
            print(f"❌ 模型 '{provider_name}' 不可用")
            print(f"可用模型: {', '.join(providers)}")
            return
        
        if not self.ai_service.test_provider(provider_name):
            print(f"❌ 模型 '{provider_name}' 测试失败")
            return
        
        self.ai_service.default_provider = provider_name
        print(f"✅ 已切换到模型: {provider_name}")
    
    def search_documents(self, query: str):
        """搜索文档"""
        if not self.pipeline:
            print("❌ 管道未初始化")
            return
        
        print(f"\n🔍 搜索: {query}")
        print("-" * 40)
        
        try:
            results = self.pipeline.search_documents(query, n_results=3)
            
            if not results:
                print("❌ 没有找到相关文档")
                return
            
            for i, result in enumerate(results, 1):
                similarity = 1 - result['distance']
                content = result['content'][:150].replace('\n', ' ')
                metadata = result['metadata']
                
                print(f"\n{i}. 相似度: {similarity:.2f}")
                print(f"   文档: {metadata.get('document_title', '未知')}")
                print(f"   类型: {metadata.get('type', 'unknown')}")
                print(f"   内容: {content}...")
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
    
    def display_docs_info(self):
        """显示知识库信息"""
        if not self.pipeline:
            print("❌ 管道未初始化")
            return
        
        db_info = self.pipeline.get_database_info()
        print(f"\n📚 知识库信息:")
        print(f"  文档数量: {db_info.get('document_count', 0)}")
        print(f"  集合名称: {db_info.get('collection_name', 'N/A')}")
        print(f"  存储路径: {db_info.get('persist_directory', 'N/A')}")
    
    def ask_question(self, question: str, provider: str = None, **kwargs):
        """提问"""
        if not self.rag_pipeline:
            print("❌ RAG管道未初始化")
            return
        
        print(f"\n🤔 问题: {question}")
        print("🤖 正在思考...")
        print("-" * 40)
        
        try:
            result = self.rag_pipeline.generate_answer(
                query=question,
                provider=provider,
                **kwargs
            )
            
            # 显示回答
            print(f"\n💡 回答:")
            print(result['answer'])
            
            # 显示来源
            if result['sources']:
                print(f"\n📚 参考来源:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['document']} (相似度: {source['similarity']:.2f})")
            
            # 保存到历史
            self.chat_history.append({
                'question': question,
                'answer': result['answer'],
                'sources': result['sources'],
                'timestamp': result['timestamp']
            })
            
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
    
    def display_history(self):
        """显示对话历史"""
        if not self.chat_history:
            print("📝 对话历史为空")
            return
        
        print(f"\n📝 对话历史 (共 {len(self.chat_history)} 条):")
        print("=" * 60)
        
        for i, entry in enumerate(self.chat_history, 1):
            print(f"\n{i}. 问题: {entry['question']}")
            print(f"   回答: {entry['answer'][:100]}...")
            print(f"   时间: {entry['timestamp']}")
            print("-" * 40)
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
        print("✅ 对话历史已清空")
    
    def reload_config(self):
        """重新加载配置"""
        try:
            self.ai_service = AIService()
            if self.pipeline:
                self.rag_pipeline = RAGPipeline(self.pipeline.vector_store, self.ai_service)
            print("✅ 配置重新加载成功")
        except Exception as e:
            print(f"❌ 配置重新加载失败: {e}")
    
    def process_command(self, user_input: str):
        """处理用户命令"""
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        # 处理命令
        if user_input.startswith('/'):
            parts = user_input[1:].split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ['quit', 'exit', 'q']:
                return False
            elif command == 'help':
                self.display_help()
            elif command == 'clear':
                self.clear_history()
            elif command == 'history':
                self.display_history()
            elif command == 'providers':
                self.display_providers()
            elif command == 'switch':
                if args:
                    self.switch_provider(args)
                else:
                    print("❌ 请指定模型名称")
            elif command == 'test':
                if args:
                    if self.ai_service and args in self.ai_service.get_available_providers():
                        status = "可用" if self.ai_service.test_provider(args) else "不可用"
                        print(f"模型 '{args}': {status}")
                    else:
                        print(f"❌ 模型 '{args}' 不存在")
                else:
                    print("❌ 请指定模型名称")
            elif command == 'search':
                if args:
                    self.search_documents(args)
                else:
                    print("❌ 请指定搜索关键词")
            elif command == 'docs':
                self.display_docs_info()
            elif command == 'config':
                if self.ai_service:
                    providers = self.ai_service.get_available_providers()
                    default = self.ai_service.default_provider
                    print(f"当前配置: 默认模型={default}, 可用模型={providers}")
                else:
                    print("❌ AI服务未初始化")
            elif command == 'reload':
                self.reload_config()
            elif command == 'ask':
                if args:
                    self.ask_question(args)
                else:
                    print("❌ 请指定问题")
            else:
                print(f"❌ 未知命令: {command}")
                print("输入 '/help' 查看帮助")
        else:
            # 普通问题
            self.ask_question(user_input)
        
        return True
    
    def run(self):
        """运行聊天界面"""
        if not self.display_welcome():
            return
        
        print("\n💬 开始对话 (输入 '/help' 查看帮助)")
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if not self.process_command(user_input):
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='化工文档智能问答系统')
    parser.add_argument('--provider', '-p', type=str, help='指定AI模型提供者')
    parser.add_argument('--question', '-q', type=str, help='直接提问（非交互模式）')
    args = parser.parse_args()
    
    # 创建聊天界面
    chat = AIChatInterface()
    
    if args.question:
        # 非交互模式
        if chat.rag_pipeline:
            chat.ask_question(args.question, provider=args.provider)
        else:
            print("❌ 服务初始化失败")
    else:
        # 交互模式
        chat.run()

if __name__ == "__main__":
    main()
