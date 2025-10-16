import fitz  # PyMuPDF
import camelot
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
import base64
import io
import os
import re
import json
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


class ChemicalPDFProcessor:
    """化工专业PDF文档处理器，专门用于处理设计规定等复杂文档"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.extracted_data = {
            "text": "",
            "tables": [],
            "images": [],
            "shapes": [],
            "page_count": 0
        }
        self.clip_model = None
        self.clip_processor = None
        self.init_clip_model()

    def init_clip_model(self):
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        except Exception as e:
            print(f"初始化CLIP模型时出错: {e}")

    def extract_full_content(self) -> Dict[str, Any]:
        self.extract_text_content()
        self.extract_tables()
        self.extract_images()
        return self.extracted_data

    def extract_text(self) -> str:
        """保持向后兼容：调用 extract_text_content"""
        return self.extract_text_content()
    



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
        """多源提取 + 页眉页脚去除 + 编码清洗"""
        candidates: List[Tuple[str, str]] = []  # (source, text)

        # 方法1: pdfplumber - 保持布局
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.extracted_data["page_count"] = len(pdf.pages)
                pages = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    pages.append(t)
                text = "\n".join(pages)
                if text.strip():
                    candidates.append(("pdfplumber", text))
        except Exception as e:
            print(f"pdfplumber提取失败: {e}")

        # 方法2: PyMuPDF - 快速提取
        try:
            doc = fitz.open(self.pdf_path)
            pages = []
            for p in doc:
                t = p.get_text("text") or ""
                pages.append(t)
            doc.close()
            text = "\n".join(pages)
            if text.strip():
                candidates.append(("pymupdf", text))
        except Exception as e:
            print(f"PyMuPDF提取失败: {e}")

        # 方法3: PDFMiner - 精确提取
        try:
            miner_text = pdfminer_extract_text(self.pdf_path) or ""
            if miner_text.strip():
                candidates.append(("pdfminer", miner_text))
        except Exception as e:
            print(f"PDFMiner提取失败: {e}")

        if not candidates:
            self.extracted_data['text_content'] = ""
            self.extracted_data['text_sources'] = []
            return ""

        # 选择内容最丰富的来源
        best_source, best_text = max(candidates, key=lambda x: len(x[1]))

        # 去除页眉页脚（基于跨页重复的首行/尾行启发式）
        cleaned = self._remove_headers_footers(best_text)
        cleaned = self._clean_chemical_symbols(cleaned)

        self.extracted_data['text_content'] = cleaned
        self.extracted_data['text_sources'] = [s for s, _ in candidates]
        return cleaned
    
    def extract_tables_with_camelot(self) -> List[Dict[str, Any]]:
        """使用Camelot提取表格数据（内部工具，供 extract_tables 调用）"""
        tables_data: List[Dict[str, Any]] = []
        lattice_tables = None
        stream_tables = None
        try:
            lattice_tables = camelot.read_pdf(
                self.pdf_path,
                flavor='lattice',
                pages='all'
            )
        except Exception as e:
            print(f"Camelot lattice 提取失败: {e}")
        try:
            stream_tables = camelot.read_pdf(
                self.pdf_path,
                flavor='stream',
                pages='all'
            )
        except Exception as e:
            print(f"Camelot stream 提取失败: {e}")

        combined = []
        if lattice_tables:
            combined += [(t, 'lattice') for t in list(lattice_tables)]
        if stream_tables:
            combined += [(t, 'stream') for t in list(stream_tables)]

        seen: set = set()
        for table, flavor in combined:
            bbox = getattr(table, 'bbox', None)
            df = table.df.copy() if hasattr(table, 'df') else pd.DataFrame()
            if not df.empty:
                df = df.dropna(how='all').dropna(axis=1, how='all')

            # 更稳健的去重键：优先使用bbox；否则使用(page, shape, headers)
            if bbox is not None:
                try:
                    bbox_key = tuple(round(float(v), 2) for v in bbox)
                except Exception:
                    bbox_key = ('no-bbox',)
                key = (getattr(table, 'page', None), bbox_key)
            else:
                headers_tuple = tuple(df.iloc[0].tolist()) if not df.empty else ()
                key = (getattr(table, 'page', None), df.shape, headers_tuple)
            if key in seen:
                continue
            seen.add(key)

            context_text = None
            if bbox is not None and getattr(table, 'page', None) is not None:
                try:
                    context_text = self._get_table_context(getattr(table, 'page', 1) - 1, bbox)
                except Exception:
                    context_text = None

            # 提取准确率（如果可用）
            accuracy = None
            try:
                accuracy = getattr(table, 'accuracy', None)
                if accuracy is None and hasattr(table, 'parsing_report'):
                    pr = getattr(table, 'parsing_report', {})
                    if isinstance(pr, dict):
                        accuracy = pr.get('accuracy')
            except Exception:
                accuracy = None

            table_info = {
                'page_number': getattr(table, 'page', None),
                'table_number': len(tables_data) + 1,
                'bbox': getattr(table, 'bbox', None),
                'headers': df.iloc[0].tolist() if not df.empty else [],
                'data': df.iloc[1:].to_dict('records') if not df.empty else [],
                'shape': df.shape if not df.empty else (0, 0),
                'accuracy': accuracy,
                'context': context_text,
                'source': f'camelot-{flavor}'
            }
            tables_data.append(table_info)

        return tables_data

    def extract_tables(self) -> List[Dict[str, Any]]:
        tables = []

        camelot_tables = self.extract_tables_with_camelot()
        tables.extend(camelot_tables)
        self.extracted_data['tables'] = tables
        return tables

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
    
    def _clean_chemical_symbols(self, text: str) -> str:
        """清理化工特殊符号，修复常见的编码问题"""
        symbol_map = {
            "ï¼š": "：", "â„ƒ": "℃", "ï¼ž": ">", "ï¼œ": "<",
            "ï¼…": "%", "â€“": "–", "â€”": "—", "ï¼Œ": "，",
            "ï¼Ž": "．", "Î¦": "φ", "MPa": "MPa", "m³": "m³"
        }
        for wrong, correct in symbol_map.items():
            text = text.replace(wrong, correct)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """启发式去页眉页脚：移除在全文中高频重复且较短的行"""
        lines = [ln.strip() for ln in text.split("\n")]
        freq: Dict[str, int] = {}
        for ln in lines:
            if not ln:
                continue
            if 3 <= len(ln) <= 80:
                freq[ln] = freq.get(ln, 0) + 1
        # 统计阈值：出现次数>=3 且占比较高的候选视为页眉/页脚
        threshold = max(3, int(0.02 * len(lines)))
        headers_footers = {ln for ln, c in freq.items() if c >= threshold}
        if not headers_footers:
            return text
        cleaned_lines = [ln for ln in lines if ln not in headers_footers]
        return "\n".join(cleaned_lines)

    def _get_table_context(self, page_index: int, bbox: Any, margin: int = 20) -> Optional[str]:
        """获取表格周边的上下文文本（基于 pdfplumber 裁剪）"""
        try:
            if bbox is None:
                return None
            x0, y0, x1, y1 = bbox
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_index < 0 or page_index >= len(pdf.pages):
                    return None
                page = pdf.pages[page_index]
                # 扩大边界获取上下文
                rx0 = max(0, x0 - margin)
                ry0 = max(0, y0 - margin)
                rx1 = min(page.width, x1 + margin)
                ry1 = min(page.height, y1 + margin)
                region = page.crop((rx0, ry0, rx1, ry1))
                return (region.extract_text() or "").strip()
        except Exception:
            return None

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
    pdf_path = ""
    
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
