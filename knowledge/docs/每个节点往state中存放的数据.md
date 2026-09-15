1. input:
    {
        "import_file_path": 导入文件路径(初始的pdf或者md)
        "file_dir": 导入(出)文件目录（存放pdf解析后产物的目录）
    } 

2. entry_node:
    {
        "is_pdf_read_enabled" = 是否启用 PDF 读取
        "pdf_path" =  PDF 文件路径
        "file_title" = 文件标题（不含扩展名）
    }

3. pdf_to_md_node:
    {
        "md_path" = 转换后Markdown 文件路径
    }

4. md_image_node:
    {
        "md_content" = # Markdown 文档内容(将图片上下文传给VLM获取照片摘要并回填到md文档中)
    }

5. document_split_node:
    {
        
    }
