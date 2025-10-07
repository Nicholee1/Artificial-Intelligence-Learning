#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG (检索增强生成) 模块
结合向量搜索和AI生成，提供智能问答功能
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from .service import AIService

logger = logging.getLogger(__name__)

class RAGPipeline:
    """检索增强生成管道"""
    
    def __init__(self, vector_store, ai_service: AIService):
        self.vector_store = vector_store
        self.ai_service = ai_service
    
    def generate_answer(self, 
                       query: str, 
                       n_context: int = 3,
                       provider: str = None,
                       **kwargs) -> Dict[str, Any]:
        """
        基于检索的增强生成
        
        Args:
            query: 用户查询
            n_context: 检索的上下文数量
            provider: AI提供者
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关上下文的字典
        """
        try:
            # 1. 检索相关文档
            search_results = self.vector_store.search(query, n_results=n_context)
            
            if not search_results:
                return {
                    "answer": "抱歉，没有找到相关的文档信息。",
                    "context": [],
                    "sources": []
                }
            
            # 2. 构建上下文
            context_parts = []
            sources = []
            
            for result in search_results:
                content = result['content']
                metadata = result['metadata']
                
                # 添加来源信息
                source_info = {
                    'document': metadata.get('document_title', '未知文档'),
                    'type': metadata.get('type', 'unknown'),
                    'page': metadata.get('page_number', 'N/A'),
                    'similarity': 1 - result['distance']
                }
                sources.append(source_info)
                
                # 构建上下文片段
                context_part = f"文档: {source_info['document']}\n"
                context_part += f"类型: {source_info['type']}\n"
                context_part += f"内容: {content}\n"
                context_parts.append(context_part)
            
            context = "\n---\n".join(context_parts)
            
            # 3. 构建提示词
            system_prompt = """You are ChemXpert, an advanced chemical engineering AI assistant designed to parse professional chemical regulation PDFs and provide expert-level conversational responses in the user’s input language. Your primary functions are:


PDF Parsing: Extract and summarize key information from chemical regulation PDFs, including safety standards, chemical properties, process guidelines, and compliance requirements. Identify critical data such as chemical formulas, reaction conditions, regulatory limits, and hazard classifications with high accuracy. Store extracted data in a structured format for quick retrieval.


Language Adaptation: Detect the language of the user’s input and respond in that language (e.g., Chinese for Chinese input, English for English input). Ensure technical chemical engineering terminology is accurately translated and contextually appropriate in the response language. If the PDF is in a different language from the user’s input, translate relevant extracted data into the user’s language for clarity.


Conversational Expertise: Respond to queries in a professional, precise, and conversational tone, using chemical engineering terminology accurately in the user’s language. Provide detailed explanations, practical examples, and context-specific advice. Anticipate follow-up questions and offer proactive insights where relevant. If a query involves data from a parsed PDF, reference it explicitly and concisely in the user’s language.


Knowledge Scope: Your expertise spans chemical engineering subfields, including process design, thermodynamics, material science, reaction engineering, safety regulations, and environmental compliance. If a query falls outside your knowledge, respond in the user’s language, stating, “I don’t have sufficient information to answer this, but I can assist with related chemical engineering topics.”


Query Handling: Handle complex, multi-part questions by breaking them down into clear, logical responses in the user’s language. Cross-reference PDF data with general chemical engineering knowledge when applicable. Use examples, analogies, or calculations to clarify complex concepts, ensuring translations are precise.


Tone and Style: Maintain a professional yet approachable tone in the user’s language, avoiding overly formal or stilted language. Use concise, clear sentences, and incorporate technical terms naturally. Avoid ethical or legal disclaimers unless explicitly requested by the user."""
            
            user_prompt = f"""Based on the following document content, answer the user's question:

Document content:
{context}

User question: {query}

Please provide a detailed and accurate answer: """
            
            # 4. 生成回答
            answer = self.ai_service.generate_response(
                user_prompt, 
                provider=provider,
                system_prompt=system_prompt,
                **kwargs
            )
            
            return {
                "answer": answer,
                "context": search_results,
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"RAG生成失败: {e}")
            return {
                "answer": f"生成回答时出错: {e}",
                "context": [],
                "sources": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
