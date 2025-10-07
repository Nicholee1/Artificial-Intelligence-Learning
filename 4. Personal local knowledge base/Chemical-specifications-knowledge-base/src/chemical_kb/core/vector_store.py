import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import re

class ChemicalVectorStore:
    """化工专业文档向量化存储系统"""
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "chemical_documents",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量存储系统
        
        Args:
            persist_directory: ChromaDB持久化目录
            collection_name: 集合名称
            model_name: 文本向量化模型名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 初始化文本向量化模型
        print(f"正在加载文本向量化模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("模型加载完成")
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"使用现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "化工专业文档向量存储"}
            )
            print(f"创建新集合: {collection_name}")
    
    def load_json_data(self, json_path: str) -> Dict[str, Any]:
        """加载PDF处理后的JSON数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载JSON数据: {json_path}")
            return data
        except Exception as e:
            print(f"加载JSON数据失败: {e}")
            return {}
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将长文本分割成小块，便于向量化
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            overlap: 重叠大小
        
        Returns:
            文本块列表
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句号、换行符等位置分割
            if end < len(text):
                # 寻找合适的分割点
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in ['。', '\n', '；', '！', '？', '.', ';', '!', '?']:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_document_chunks(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从JSON数据中提取文档块，用于向量化
        
        Args:
            json_data: PDF处理后的JSON数据
        
        Returns:
            文档块列表
        """
        chunks = []
        doc_id = str(uuid.uuid4())
        metadata = json_data.get('metadata', {})
        
        # 1. 处理完整文本内容
        full_text = json_data.get('text_content', '')
        if full_text:
            text_chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'id': f"{doc_id}_text_{i}",
                    'content': chunk,
                    'type': 'text',
                    'source': 'full_text',
                    'metadata': {
                        'document_title': metadata.get('title', ''),
                        'page_count': metadata.get('page_count', 0),
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                })
        
        # 2. 处理页面内容
        pages = json_data.get('pages', [])
        for page in pages:
            page_text = page.get('text', '')
            if page_text:
                page_chunks = self.chunk_text(page_text, chunk_size=300)
                for i, chunk in enumerate(page_chunks):
                    chunks.append({
                        'id': f"{doc_id}_page_{page['page_number']}_{i}",
                        'content': chunk,
                        'type': 'page_text',
                        'source': f"page_{page['page_number']}",
                        'metadata': {
                            'document_title': metadata.get('title', ''),
                            'page_number': page['page_number'],
                            'page_width': page.get('width', 0),
                            'page_height': page.get('height', 0),
                            'chunk_index': i
                        }
                    })
        
        # 3. 处理表格数据
        tables = json_data.get('tables', [])
        for table in tables:
            # 表格标题和内容
            table_content = f"表格 {table.get('table_number', '')} (第{table.get('page_number', '')}页)\n"
            
            # 添加表头
            headers = table.get('headers', [])
            if headers:
                table_content += "表头: " + " | ".join([str(h) for h in headers if h]) + "\n"
            
            # 添加表格数据
            data = table.get('data', [])
            for row in data[:10]:  # 只取前10行
                row_text = " | ".join([str(v) for v in row.values() if v])
                if row_text.strip():
                    table_content += row_text + "\n"
            
            if table_content.strip():
                shape = table.get('shape', [3,3])
                shape_str = ','.join(map(str, shape)) 
                chunks.append({
                    'id': f"{doc_id}_table_{table.get('page_number', 0)}_{table.get('table_number', 0)}",
                    'content': table_content.strip(),
                    'type': 'table',
                    'source': f"page_{table.get('page_number', 0)}",
                    'metadata': {
                        'document_title': metadata.get('title', ''),
                        'page_number': table.get('page_number', 0),
                        'table_number': table.get('table_number', 0),
                        'table_shape': shape_str
                    }
                })
        
        # 4. 处理结构化数据中的章节
        structured_data = json_data.get('structured_data', {})
        sections = structured_data.get('sections', [])
        for i, section in enumerate(sections):
            section_content = f"{section.get('title', '')}\n{section.get('content', '')}"
            if section_content.strip():
                chunks.append({
                    'id': f"{doc_id}_section_{i}",
                    'content': section_content.strip(),
                    'type': 'section',
                    'source': 'structured_data',
                    'metadata': {
                        'document_title': metadata.get('title', ''),
                        'section_title': section.get('title', ''),
                        'section_index': i
                    }
                })
        
        # 5. 处理技术规格
        tech_specs = structured_data.get('technical_specifications', {})
        for spec_type, spec_values in tech_specs.items():
            if spec_values and isinstance(spec_values, list):
                spec_content = f"{spec_type}: " + ", ".join([str(v) for v in spec_values[:20]])
                if spec_content.strip():
                    chunks.append({
                        'id': f"{doc_id}_spec_{spec_type}",
                        'content': spec_content,
                        'type': 'technical_spec',
                        'source': 'technical_specifications',
                        'metadata': {
                            'document_title': metadata.get('title', ''),
                            'spec_type': spec_type,
                            'spec_count': len(spec_values)
                        }
                    })
        
        print(f"提取了 {len(chunks)} 个文档块")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本的向量表示"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"生成向量失败: {e}")
            return []
    
    def is_document_exists(self, json_path: str) -> bool:
        """检查文档是否已存在于向量数据库中"""
        try:
            # 从JSON路径生成文档ID前缀
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            doc_prefix = base_name.replace('_structured', '')
            
            # 获取所有文档ID
            all_docs = self.collection.get()
            existing_ids = all_docs.get('ids', [])
            
            # 检查是否有以该文档前缀开头的ID
            for doc_id in existing_ids:
                if doc_id.startswith(doc_prefix):
                    return True
            return False
        except:
            return False
    
    def add_documents(self, json_path: str, force_reload: bool = False) -> bool:
        """
        将PDF处理后的JSON数据添加到向量数据库
        
        Args:
            json_path: JSON文件路径
            force_reload: 是否强制重新加载
        
        Returns:
            是否添加成功
        """
        try:
            # 检查文档是否已存在
            if not force_reload and self.is_document_exists(json_path):
                print(f"⏭️ 文档已存在于向量数据库: {os.path.basename(json_path)}")
                return True
            
            # 加载JSON数据
            json_data = self.load_json_data(json_path)
            if not json_data:
                return False
            
            # 提取文档块
            chunks = self.extract_document_chunks(json_data)
            if not chunks:
                print("没有提取到有效的文档块")
                return False
            
            # 准备数据
            texts = [chunk['content'] for chunk in chunks]
            ids = [chunk['id'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # 生成向量
            print("正在生成向量...")
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                print("向量生成失败")
                return False
            
            # 添加到ChromaDB
            print("正在添加到向量数据库...")
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"成功添加 {len(chunks)} 个文档块到向量数据库")
            return True
            
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
        
        Returns:
            搜索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.generate_embeddings([query])[0]
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"获取集合信息失败: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """删除指定文档"""
        try:
            self.collection.delete(ids=[document_id])
            print(f"成功删除文档: {document_id}")
            return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            # 获取所有文档ID
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
            print("成功清空集合")
            return True
        except Exception as e:
            print(f"清空集合失败: {e}")
            return False

def main():
    """主函数 - 演示向量化存储系统的使用"""
    print("🔬 化工专业文档向量化存储系统")
    print("=" * 50)
    
    # 初始化向量存储系统
    vector_store = ChemicalVectorStore()
    
    # 示例：添加PDF处理后的JSON数据
    json_file = "./PDF/KLDL-03c-04-05PD-B58-2021 管道专业详细设计工程设计文件内容和深度统一规定_structured.json"
    
    if os.path.exists(json_file):
        print(f"正在处理文件: {json_file}")
        success = vector_store.add_documents(json_file)
        
        if success:
            print("✅ 文档添加成功")
            
            # 显示集合信息
            info = vector_store.get_collection_info()
            print(f"📊 集合信息: {info}")
            
            # 示例搜索
            print("\n🔍 示例搜索:")
            queries = [
                "管道设计规范",
                "设备布置图",
                "材料规格表",
                "设计文件编号"
            ]
            
            for query in queries:
                print(f"\n查询: {query}")
                results = vector_store.search(query, n_results=3)
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['content'][:100]}...")
                    print(f"     类型: {result['metadata'].get('type', 'unknown')}")
                    print(f"     来源: {result['metadata'].get('source', 'unknown')}")
        else:
            print("❌ 文档添加失败")
    else:
        print(f"❌ JSON文件不存在: {json_file}")
        print("请先运行PDF处理程序生成JSON文件")

if __name__ == "__main__":
    main()
