import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import pandas as pd
import json
import base64
from PIL import Image
import io
import re
from typing import Dict, List, Any, Optional
import os

class ChemicalPDFProcessor:
    """化工专业PDF文档处理器，专门用于处理设计规定等复杂文档"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.extracted_data = {
            'metadata': {},
            'pages': [],
            'tables': [],
            'images': [],
            'text_content': '',
            'structured_data': {}
        }
    
    def extract_metadata(self) -> Dict[str, Any]:
        """提取PDF元数据"""
        try:
            with fitz.open(self.pdf_path) as doc:
                metadata = doc.metadata
                self.extracted_data['metadata'] = {
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'creator': metadata.get('creator', ''),
                    'producer': metadata.get('producer', ''),
                    'creation_date': metadata.get('creationDate', ''),
                    'modification_date': metadata.get('modDate', ''),
                    'page_count': len(doc)
                }
        except Exception as e:
            print(f"提取元数据时出错: {e}")
        
        return self.extracted_data['metadata']
    
    def extract_text_content(self) -> str:
        """使用多种方法提取文本内容，智能去重确保最佳效果"""
        all_text = []
        text_sources = []
        
        # 方法1: pdfplumber - 保持布局
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                pdfplumber_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pdfplumber_text.append(text)
                if pdfplumber_text:
                    all_text.append('\n'.join(pdfplumber_text))
                    text_sources.append('pdfplumber')
        except Exception as e:
            print(f"pdfplumber提取失败: {e}")
        
        # 方法2: PyMuPDF - 快速提取
        try:
            doc = fitz.open(self.pdf_path)
            pymupdf_text = []
            for page in doc:
                text = page.get_text()
                if text:
                    pymupdf_text.append(text)
            doc.close()
            if pymupdf_text:
                all_text.append('\n'.join(pymupdf_text))
                text_sources.append('pymupdf')
        except Exception as e:
            print(f"PyMuPDF提取失败: {e}")
        
        # 方法3: PDFMiner - 精确提取
        try:
            miner_text = extract_text(self.pdf_path)
            if miner_text:
                all_text.append(miner_text)
                text_sources.append('pdfminer')
        except Exception as e:
            print(f"PDFMiner提取失败: {e}")
        
        # 智能合并和去重
        if len(all_text) == 1:
            # 只有一个方法成功，直接使用
            final_text = all_text[0]
        else:
            # 多个方法成功，选择最长的（通常最完整）
            final_text = max(all_text, key=len)
            print(f"使用 {text_sources[all_text.index(final_text)]} 的提取结果（最完整）")
        
        self.extracted_data['text_content'] = final_text
        self.extracted_data['text_sources'] = text_sources
        return final_text
    
    def extract_tables(self) -> List[Dict[str, Any]]:
        """提取表格数据，保持结构"""
        tables_data = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table_num, table in enumerate(tables):
                        if table and len(table) > 1:  # 确保表格有数据
                            # 转换为DataFrame便于处理
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # 清理数据
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            table_info = {
                                'page_number': page_num,
                                'table_number': table_num + 1,
                                'headers': table[0] if table[0] else [],
                                'data': df.to_dict('records'),
                                'shape': df.shape,
                                'raw_table': table
                            }
                            
                            tables_data.append(table_info)
                            
        except Exception as e:
            print(f"提取表格时出错: {e}")
        
        self.extracted_data['tables'] = tables_data
        return tables_data
    
    def extract_images(self) -> List[Dict[str, Any]]:
        """提取图像和图表"""
        images_data = []
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # 获取图像
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # 确保不是CMYK
                        img_data = {
                            'page_number': page_num + 1,
                            'image_number': img_index + 1,
                            'width': pix.width,
                            'height': pix.height,
                            'colorspace': pix.colorspace.name if pix.colorspace else 'Unknown',
                            'base64_data': base64.b64encode(pix.tobytes("png")).decode(),
                            'format': 'png'
                        }
                        
                        images_data.append(img_data)
                    
                    pix = None  # 释放内存
            
            doc.close()
            
        except Exception as e:
            print(f"提取图像时出错: {e}")
        
        self.extracted_data['images'] = images_data
        return images_data
    
    def process_pages(self) -> List[Dict[str, Any]]:
        """逐页处理，提取每页的完整信息（避免重复提取表格）"""
        pages_data = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_info = {
                        'page_number': page_num,
                        'text': page.extract_text() or '',
                        'tables': [],
                        'images': [],
                        'width': page.width,
                        'height': page.height
                    }
                    
                    # 只提取该页的表格信息，不重复处理
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            page_info['tables'].append({
                                'table_number': table_num + 1,
                                'data': df.to_dict('records'),
                                'headers': table[0] if table[0] else []
                            })
                    
                    pages_data.append(page_info)
            
        except Exception as e:
            print(f"处理页面时出错: {e}")
        
        self.extracted_data['pages'] = pages_data
        return pages_data
    
    def structure_for_llm(self) -> Dict[str, Any]:
        """为LLM优化数据结构"""
        structured_data = {
            'document_type': '化工专业设计规定',
            'summary': self._generate_summary(),
            'sections': self._extract_sections(),
            'key_tables': self._identify_key_tables(),
            'technical_specifications': self._extract_technical_specs(),
            'images_description': self._describe_images(),
            'full_text': self.extracted_data['text_content'],
            'metadata': self.extracted_data['metadata']
        }
        
        self.extracted_data['structured_data'] = structured_data
        return structured_data
    
    def _generate_summary(self) -> str:
        """生成文档摘要"""
        text = self.extracted_data['text_content']
        if not text:
            return "无法生成摘要"
        
        # 提取前几段作为摘要
        paragraphs = text.split('\n\n')
        summary_paragraphs = [p.strip() for p in paragraphs[:3] if p.strip()]
        return '\n'.join(summary_paragraphs)
    
    def _extract_sections(self) -> List[Dict[str, str]]:
        """提取文档章节"""
        text = self.extracted_data['text_content']
        sections = []
        
        # 查找章节标题（数字开头或特定关键词）
        section_patterns = [
            r'^\d+\.?\s+[^\n]+',  # 数字开头的标题
            r'^[一二三四五六七八九十]+、[^\n]+',  # 中文数字标题
            r'^第[一二三四五六七八九十\d]+[章节条][^\n]+',  # 第X章/节/条
        ]
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in section_patterns:
                if re.match(pattern, line):
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'title': line,
                        'content': ''
                    }
                    break
            else:
                if current_section:
                    current_section['content'] += line + '\n'
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _identify_key_tables(self) -> List[Dict[str, Any]]:
        """识别关键表格"""
        key_tables = []
        
        for table in self.extracted_data['tables']:
            # 根据表格内容判断是否为关键表格
            headers = table.get('headers', [])
            if any(keyword in str(headers).lower() for keyword in 
                   ['规格', '尺寸', '材料', '性能', '参数', '标准', '等级']):
                key_tables.append({
                    'page': table['page_number'],
                    'table_number': table['table_number'],
                    'headers': headers,
                    'data': table['data'][:5],  # 只取前5行
                    'description': f"第{table['page_number']}页的{table['table_number']}号表格"
                })
        
        return key_tables
    
    def _extract_technical_specs(self) -> Dict[str, Any]:
        """提取技术规格参数"""
        text = self.extracted_data['text_content']
        specs = {
            'materials': [],
            'dimensions': [],
            'standards': [],
            'parameters': []
        }
        
        # 提取材料规格
        material_patterns = [
            r'材料[：:]\s*([^\n]+)',
            r'材质[：:]\s*([^\n]+)',
            r'([A-Z0-9]+)\s*钢',
            r'不锈钢\s*([A-Z0-9]+)'
        ]
        
        for pattern in material_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            specs['materials'].extend(matches)
        
        # 提取尺寸规格
        dimension_patterns = [
            r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*[×x]?\s*(\d+(?:\.\d+)?)?',
            r'直径[：:]\s*(\d+(?:\.\d+)?)\s*mm',
            r'长度[：:]\s*(\d+(?:\.\d+)?)\s*mm'
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text)
            specs['dimensions'].extend(matches)
        
        return specs
    
    def _describe_images(self) -> List[Dict[str, str]]:
        """描述图像内容"""
        descriptions = []
        
        for img in self.extracted_data['images']:
            descriptions.append({
                'page': img['page_number'],
                'image_number': img['image_number'],
                'dimensions': f"{img['width']}x{img['height']}",
                'description': f"第{img['page_number']}页的第{img['image_number']}个图像，尺寸{img['width']}x{img['height']}像素"
            })
        
        return descriptions
    
    def deduplicate_and_merge_data(self):
        """去重和合并数据，避免重复处理"""
        print("正在去重和合并数据...")
        
        # 合并表格数据（避免重复）
        all_tables = []
        table_ids = set()  # 用于去重
        
        # 从extract_tables获取的表格
        for table in self.extracted_data.get('tables', []):
            table_id = f"page_{table['page_number']}_table_{table['table_number']}"
            if table_id not in table_ids:
                all_tables.append(table)
                table_ids.add(table_id)
        
        # 从process_pages获取的表格（如果不同）
        for page in self.extracted_data.get('pages', []):
            for table in page.get('tables', []):
                table_id = f"page_{page['page_number']}_table_{table['table_number']}"
                if table_id not in table_ids:
                    # 转换为统一格式
                    unified_table = {
                        'page_number': page['page_number'],
                        'table_number': table['table_number'],
                        'headers': table['headers'],
                        'data': table['data'],
                        'shape': [len(table['data']), len(table['headers'])]
                    }
                    all_tables.append(unified_table)
                    table_ids.add(table_id)
        
        self.extracted_data['tables'] = all_tables
        print(f"合并后共有 {len(all_tables)} 个表格")
        
        # 合并图像数据（避免重复）
        all_images = []
        image_ids = set()
        
        for img in self.extracted_data.get('images', []):
            img_id = f"page_{img['page_number']}_img_{img['image_number']}"
            if img_id not in image_ids:
                all_images.append(img)
                image_ids.add(img_id)
        
        self.extracted_data['images'] = all_images
        print(f"合并后共有 {len(all_images)} 个图像")
    
    def save_structured_data(self, output_path: str = None):
        """保存结构化数据到新文件，不覆盖原PDF"""
        if not output_path:
            # 生成新的JSON文件，不覆盖原PDF
            base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            output_path = os.path.join(os.path.dirname(self.pdf_path), f"{base_name}_structured.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"结构化数据已保存到: {output_path}")
        print(f"原PDF文件保持不变: {self.pdf_path}")
        return output_path
    
    def process_full_document(self):
        """完整处理文档"""
        print("开始处理化工专业PDF文档...")
        
        # 1. 提取元数据
        print("1. 提取元数据...")
        self.extract_metadata()
        
        # 2. 提取文本内容（智能去重）
        print("2. 提取文本内容...")
        self.extract_text_content()
        
        # 3. 提取表格
        print("3. 提取表格数据...")
        self.extract_tables()
        
        # 4. 提取图像
        print("4. 提取图像和图表...")
        self.extract_images()
        
        # 5. 逐页处理
        print("5. 逐页处理...")
        self.process_pages()
        
        # 6. 去重和合并数据
        print("6. 去重和合并数据...")
        self.deduplicate_and_merge_data()
        
        # 7. 结构化处理
        print("7. 为LLM优化数据结构...")
        self.structure_for_llm()
        
        print("文档处理完成！")
        return self.extracted_data

def main():
    """主函数"""
    pdf_path = "./PDF/KLDL-03c-04-05PD-B58-2021 管道专业详细设计工程设计文件内容和深度统一规定.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    # 创建处理器
    processor = ChemicalPDFProcessor(pdf_path)
    
    # 完整处理文档
    result = processor.process_full_document()
    
    # 保存结果
    output_file = processor.save_structured_data()
    
    # 打印摘要信息
    print("\n=== 处理结果摘要 ===")
    print(f"文档页数: {result['metadata'].get('page_count', 'Unknown')}")
    print(f"提取表格数量: {len(result['tables'])}")
    print(f"提取图像数量: {len(result['images'])}")
    print(f"文本长度: {len(result['text_content'])} 字符")
    
    # 显示关键表格
    if result['structured_data']['key_tables']:
        print("\n=== 关键表格 ===")
        for table in result['structured_data']['key_tables']:
            print(f"- {table['description']}")
            print(f"  表头: {table['headers']}")

if __name__ == "__main__":
    main()
