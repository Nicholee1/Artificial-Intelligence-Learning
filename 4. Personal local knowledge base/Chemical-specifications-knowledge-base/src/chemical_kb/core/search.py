#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索功能模块
"""

from typing import List, Dict, Any, Optional
from .vector_store import ChemicalVectorStore

class ChemicalSearch:
    """化工文档搜索类"""
    
    def __init__(self, vector_store: ChemicalVectorStore):
        self.vector_store = vector_store
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               doc_type: str = None) -> List[Dict[str, Any]]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            doc_type: 文档类型过滤
        
        Returns:
            搜索结果列表
        """
        filter_metadata = None
        if doc_type:
            filter_metadata = {"type": doc_type}
        
        return self.vector_store.search(query, n_results, filter_metadata)

def main():
    """主函数 - 简单的搜索界面"""
    print("🔍 化工文档搜索")
    print("=" * 30)
    
    try:
        # 初始化向量存储
        vector_store = ChemicalVectorStore()
        search_engine = ChemicalSearch(vector_store)
        
        # 检查数据库状态
        info = vector_store.get_collection_info()
        if info.get('document_count', 0) == 0:
            print("❌ 数据库为空，请先运行管道处理PDF文档")
            return
        
        print(f"📊 数据库中有 {info['document_count']} 个文档")
        print("输入 'quit' 退出\n")
        
        while True:
            try:
                query = input("搜索: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not query:
                    continue
                
                # 搜索
                results = search_engine.search(query, n_results=3)
                
                if not results:
                    print("❌ 没有找到相关结果")
                    continue
                
                print(f"\n找到 {len(results)} 个结果:")
                for i, result in enumerate(results, 1):
                    similarity = 1 - result['distance']
                    content = result['content'][:100].replace('\n', ' ')
                    print(f"{i}. [{similarity:.2f}] {content}...")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 搜索出错: {e}")
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    main()