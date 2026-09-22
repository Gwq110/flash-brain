import logging
from pathlib import Path
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1.获取state中import_file_path和file_dir是否为空
        self.log_step(step_name="step1",message="判断import_file_path，file_dir是否为空")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        if not import_file_path:
            self.logger.error(f"import_file_path not found in state {state}")
            raise StateFieldError(node_name="entry_node",field_name="import_file_path",expected_type=str)

        if not file_dir:
            self.logger.error(f"file_dir not found in state {state}")
            raise StateFieldError(node_name="entry_node", field_name="file_dir", expected_type=str)


        #2.import_file_path和file_dir是否是有效地址
        self.log_step(step_name="step2",message="判断import_file_path，file_dir是否真实存在")
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        if not import_file_path_obj.exists():
            self.logger.error(f"import_file_path not exists")
            raise StateFieldError(node_name="entry_node",field_name="import_file_path",expected_type=Path)

        if not file_dir_obj.exists():
            self.logger.error(f"file_dir not exists")
            raise StateFieldError(node_name="entry_node",field_name="import_file_path",expected_type=Path)


        #3.判断文件类型（pdf,md)
        #3.1 获取文件后缀
        self.log_step(step_name="step3",message="识别文件真实类型")
        #3.2 先重置两个读取标志位，保证标志位只由真实文件类型决定
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = False
        suffix = import_file_path_obj.suffix
        if suffix == ".pdf":
            # 如果是pdf，则设置is_pdf_read_enabled为Ture，并设置pdf_path的值
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            #抛异常
            raise ValidationError(message="unsupported file type",node_name=self.name)

        #4.获取文件标题
        self.log_step(step_name="step4",message="获取文件标题")
        file_title = import_file_path_obj.stem

        state["file_title"] = file_title
        #5.将上述获取的内容全设置到state中返回
        return state

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    entry_node = EntryNode()
    init_state = {
        "import_file_path": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\input_dir\万用表RS-12的使用.pdf",
        "file_dir": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir"
    }
    state = entry_node(init_state)
    print(state)
