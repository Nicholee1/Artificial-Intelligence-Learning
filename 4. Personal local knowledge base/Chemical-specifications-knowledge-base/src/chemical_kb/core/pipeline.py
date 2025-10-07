#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成管道：PDF处理 + 向量化存储
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from .pdf_processor import ChemicalPDFProcessor
from .vector_store import ChemicalVectorStore

class IntegratedPipeline:
    """集成管道：从PDF到向量数据库的完整流程"""
    
    def __init__(self, 
                 pdf_directory: str = "data/pdf",
                 vector_db_path: str = "data/vector_db",
                 collection_name: str = "chemical_documents"):
        """
        初始化集成管道
        
        Args:
            pdf_directory: PDF文件目录
            vector_db_path: 向量数据库路径
            collection_name: 集合名称
        """
        self.pdf_directory = pdf_directory
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # 初始化向量存储系统
        self.vector_store = ChemicalVectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name
        )
        
        # 处理状态跟踪
        self.processed_files = set()
        self.status_file = os.path.join(vector_db_path, "processing_status.json")
        self.load_processing_status()
    
    def load_processing_status(self):
        """加载处理状态"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                print(f"📋 加载处理状态: {len(self.processed_files)} 个文件已处理")
            else:
                self.processed_files = set()
        except Exception as e:
            print(f"⚠️ 加载处理状态失败: {e}")
            self.processed_files = set()
    
    def save_processing_status(self):
        """保存处理状态"""
        try:
            os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'last_updated': str(datetime.now())
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存处理状态失败: {e}")
    
    def is_file_processed(self, pdf_path: str) -> bool:
        """检查文件是否已处理"""
        filename = os.path.basename(pdf_path)
        return filename in self.processed_files
    
    def mark_file_processed(self, pdf_path: str):
        """标记文件为已处理"""
        filename = os.path.basename(pdf_path)
        self.processed_files.add(filename)
        self.save_processing_status()
    
    def get_json_path(self, pdf_path: str) -> str:
        """获取对应的JSON文件路径"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(os.path.dirname(pdf_path), f"{base_name}_structured.json")
    
    def check_file_freshness(self, pdf_path: str, json_path: str) -> bool:
        """检查PDF文件是否比JSON文件更新"""
        try:
            pdf_mtime = os.path.getmtime(pdf_path)
            json_mtime = os.path.getmtime(json_path)
            return pdf_mtime > json_mtime
        except:
            return True  # 如果无法比较，认为需要重新处理
    
    def process_pdf_to_vectors(self, pdf_path: str, force_reprocess: bool = False) -> bool:
        """
        处理单个PDF文件：提取内容 -> 生成JSON -> 向量化存储
        
        Args:
            pdf_path: PDF文件路径
            force_reprocess: 是否强制重新处理
        
        Returns:
            是否处理成功
        """
        try:
            filename = os.path.basename(pdf_path)
            json_path = self.get_json_path(pdf_path)
            
            # 检查是否需要处理
            if not force_reprocess:
                if self.is_file_processed(pdf_path):
                    if os.path.exists(json_path):
                        # 检查文件新鲜度
                        if not self.check_file_freshness(pdf_path, json_path):
                            print(f"⏭️ 跳过已处理文件: {filename}")
                            # 检查是否已加载到向量数据库
                            if not self.vector_store.is_document_exists(json_path):
                                print(f"🔄 加载到向量数据库: {filename}")
                                success = self.vector_store.add_documents(json_path)
                                return success
                            return True
                        else:
                            print(f"🔄 PDF文件已更新，重新处理: {filename}")
                    else:
                        print(f"⚠️ 标记为已处理但JSON文件不存在，重新处理: {filename}")
                else:
                    print(f"🆕 新文件，开始处理: {filename}")
            else:
                print(f"🔄 强制重新处理: {filename}")
            
            print(f"\n🔄 开始处理PDF: {pdf_path}")
            
            # 1. PDF处理
            print("1️⃣ 提取PDF内容...")
            processor = ChemicalPDFProcessor(pdf_path)
            result = processor.process_full_document()
            
            if not result:
                print("❌ PDF处理失败")
                return False
            
            # 2. 保存JSON文件
            print("2️⃣ 保存结构化数据...")
            json_path = processor.save_structured_data()
            
            # 3. 向量化存储
            print("3️⃣ 向量化存储...")
            success = self.vector_store.add_documents(json_path, force_reload=force_reprocess)
            
            if success:
                print(f"✅ 成功处理: {pdf_path}")
                # 标记为已处理
                self.mark_file_processed(pdf_path)
                return True
            else:
                print(f"❌ 向量化存储失败: {pdf_path}")
                return False
                
        except Exception as e:
            print(f"❌ 处理PDF时出错: {e}")
            return False
    
    def process_all_pdfs(self, force_reprocess: bool = False) -> Dict[str, bool]:
        """
        处理目录中的所有PDF文件
        
        Args:
            force_reprocess: 是否强制重新处理所有文件
        
        Returns:
            处理结果字典
        """
        results = {}
        
        if not os.path.exists(self.pdf_directory):
            print(f"❌ PDF目录不存在: {self.pdf_directory}")
            return results
        
        # 查找所有PDF文件
        pdf_files = []
        for file in os.listdir(self.pdf_directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.pdf_directory, file))
        
        if not pdf_files:
            print(f"❌ 在 {self.pdf_directory} 中没有找到PDF文件")
            return results
        
        print(f"📁 找到 {len(pdf_files)} 个PDF文件")
        
        # 统计需要处理的文件
        new_files = []
        updated_files = []
        skipped_files = []
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            json_path = self.get_json_path(pdf_path)
            
            if force_reprocess:
                new_files.append(pdf_path)
            elif self.is_file_processed(pdf_path):
                if os.path.exists(json_path):
                    if self.check_file_freshness(pdf_path, json_path):
                        updated_files.append(pdf_path)
                    else:
                        skipped_files.append(pdf_path)
                else:
                    new_files.append(pdf_path)
            else:
                new_files.append(pdf_path)
        
        print(f"📊 处理统计:")
        print(f"   🆕 新文件: {len(new_files)}")
        print(f"   🔄 更新文件: {len(updated_files)}")
        print(f"   ⏭️ 跳过文件: {len(skipped_files)}")
        
        # 处理需要处理的文件
        all_process_files = new_files + updated_files
        for pdf_path in all_process_files:
            filename = os.path.basename(pdf_path)
            print(f"\n{'='*60}")
            print(f"处理文件: {filename}")
            print(f"{'='*60}")
            
            success = self.process_pdf_to_vectors(pdf_path, force_reprocess)
            results[filename] = success
        
        # 记录跳过的文件
        for pdf_path in skipped_files:
            filename = os.path.basename(pdf_path)
            results[filename] = True  # 标记为成功（已存在）
        
        return results
    
    def search_documents(self, 
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
            搜索结果
        """
        filter_metadata = None
        if doc_type:
            filter_metadata = {"type": doc_type}
        
        return self.vector_store.search(query, n_results, filter_metadata)
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        return self.vector_store.get_collection_info()
    
    def interactive_search(self):
        """简化的交互式搜索界面"""
        print("\n🔍 搜索界面 (输入 'quit' 退出)")
        print("-" * 40)
        
        while True:
            try:
                query = input("\n搜索: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not query:
                    continue
                
                # 执行搜索
                results = self.search_documents(query, n_results=3)
                
                if not results:
                    print("❌ 没有找到相关结果")
                    continue
                
                print(f"\n找到 {len(results)} 个结果:")
                for i, result in enumerate(results, 1):
                    similarity = 1 - result['distance']
                    content = result['content'][:100].replace('\n', ' ')
                    print(f"{i}. [{similarity:.2f}] {content}...")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 搜索出错: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='化工专业文档集成处理管道')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='强制重新处理所有PDF文件')
    parser.add_argument('--no-search', action='store_true',
                       help='处理完成后不启动交互式搜索')
    parser.add_argument('--search-only', action='store_true',
                       help='只进行搜索，不处理PDF文件')
    args = parser.parse_args()
    
    print("🔬 化工专业文档集成处理管道")
    print("=" * 60)
    
    if args.force:
        print("🔄 强制重新处理模式")
    if args.search_only:
        print("🔍 仅搜索模式")
    
    # 初始化管道
    pipeline = IntegratedPipeline()
    
    # 检查是否有已处理的JSON文件（仅用于首次加载）
    json_files = []
    if os.path.exists("./PDF"):
        for file in os.listdir("./PDF"):
            if file.endswith('_structured.json'):
                json_files.append(os.path.join("./PDF", file))
    
    # 检查是否需要首次加载JSON文件到向量数据库
    db_info = pipeline.get_database_info()
    if db_info.get('document_count', 0) == 0 and json_files:
        print(f"📁 发现 {len(json_files)} 个已处理的JSON文件，正在加载到向量数据库...")
        
        # 直接加载到向量数据库（不重新处理PDF）
        for json_file in json_files:
            print(f"正在加载: {os.path.basename(json_file)}")
            success = pipeline.vector_store.add_documents(json_file, force_reload=False)
            if success:
                print("✅ 加载成功")
            else:
                print("❌ 加载失败")
    
    # 如果不是仅搜索模式，则处理PDF文件
    if not args.search_only:
        # 智能处理PDF文件（自动跳过已处理的文件）
        print("\n🔄 开始智能处理PDF文件...")
        results = pipeline.process_all_pdfs(force_reprocess=args.force)
    else:
        # 仅搜索模式，跳过PDF处理
        print("\n⏭️ 跳过PDF处理，直接进入搜索模式")
        results = {}
    
    # 显示处理结果
    print("\n📊 处理结果汇总:")
    print("-" * 40)
    success_count = 0
    processed_count = 0
    skipped_count = 0
    
    for filename, success in results.items():
        if success:
            success_count += 1
            # 检查是否是新处理的文件
            if pipeline.is_file_processed(os.path.join("./PDF", filename)):
                processed_count += 1
            else:
                skipped_count += 1
        else:
            print(f"❌ {filename}: 处理失败")
    
    print(f"✅ 成功: {success_count} 个文件")
    print(f"🆕 新处理: {processed_count} 个文件")
    print(f"⏭️ 跳过: {skipped_count} 个文件")
    
    # 显示数据库信息
    info = pipeline.get_database_info()
    print(f"\n📊 向量数据库信息: {info}")
    
    # 启动交互式搜索
    if not args.no_search:
        # 检查向量数据库是否有数据
        db_info = pipeline.get_database_info()
        if db_info.get('document_count', 0) > 0:
            pipeline.interactive_search()
        else:
            print("❌ 向量数据库为空，无法进行搜索")
    else:
        print("✅ 处理完成，跳过交互式搜索")

if __name__ == "__main__":
    main()
