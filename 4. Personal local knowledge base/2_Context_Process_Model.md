## PDF读取
```shell
pyenv virtualenv 3.12.10 CSKB 
pyenv activate CSKB
pip install -r requirments.txt
```

转换pdf内容成Json file

### 多工具文本提取
三个文本提取工具选择文本最长的
* pdfplumber：`pdfplumber_text.append(text)`
* PyMuPDF:`page.get_text()`
* PDFMiner: `extract_text(self.pdf_path)`
### 表格提取
pdfplumber: `page.extract_tables()`
`pd.DataFrame(table[1:], columns=table[0])`

### 图片提取
pyMuPDF：
`image_list = page.get_images()`
`pix = fitz.Pixmap(doc, xref)`
