import json
import re
from pathlib import Path
from typing import Tuple, Dict, List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 参数校验
        new_md_content, file_title, max_content_length, min_content_length = self._validate_state(state)
        #2. 按照标题切分
        sections_by_head = self._split_by_head(new_md_content, file_title)
        #3. 二次切分或者合并
        final_sections = self._split_and_merge(sections_by_head, min_content_length,max_content_length)
        #4. 将切分完的内容，组装成后续节点可以直接使用的chunks
        final_chunks = self._assemble_chunks(final_sections)
        state["chunks"] = final_chunks
        #5.将chunks备份成json文件，方便后续测试
        self._backup_chunks(final_chunks,state)
        return state


    def _validate_state(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        #统一换行符
        md_content = state["md_content"]
        new_md_content = md_content.replace("\r", "\n").replace("\r\n", "\n\n")
        #核验配置信息，切片最大长度>切片最小长度
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if max_content_length < min_content_length or not max_content_length or not min_content_length or max_content_length <= 0 or min_content_length <= 0:
            self.logger.error(f"max_content_length={max_content_length},min_content_length={min_content_length}配置信息有误，请检查")
            raise ValueError("切片长度参数配置错误")
        #获取文档标题（兜底使用）
        file_title = state["file_title"]
        #将信息返回给下一步使用
        return new_md_content, file_title, max_content_length, min_content_length


    #安装标题切分文档
    @staticmethod
    def _split_by_head(new_md_content, file_title) -> List[Dict[str,Any]]:
        # 每一个标题下对应的内容为一个section
        # {
        #       "body": "收集到的所有行"
        #       “title”:"当前内容的标题"
        #       "parent_title": 当前内容的父标题（最麻烦）
        #       "file_title": 文档标题（最简单）
        # }
        #1. md_content按行切分
        md_lines = new_md_content.split("\n")
        #2. 遍历md_content没一会，判断是否遇到标题，排除代码块干扰
        # 定义匹配md标题的正则表达式
        heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)")
        code_fence_pattern = re.compile(r'^\s*```')
        in_code_fence = False
        # 定义hierarchy列表存储各级标题内容，用于寻找父表态
        hierarchy = [""]*7
        body = []
        current_title = ""
        current_title_level = 0
        #声明容器用于存储所有的块
        final_sections = []

        def _flush() -> None:
            # {
            #       "body": "收集到的所有行"
            #       “title”:"当前内容的标题"
            #       "parent_title": 当前内容的父标题（最麻烦）
            #       "file_title": 文档标题（最简单）
            # }
            if not current_title and not body:
                return None
            content = "\n".join(body)
            title = current_title if current_title else file_title
            # 收集parent_title,从当前标题往前遍历hierarchy数组
            parent_title = ""
            for index in range(current_title_level - 1, 0, -1):
                if hierarchy[index]:
                    parent_title = hierarchy[index]
                    break
            section = {
                "body": content,
                "title": title ,
                "parent_title": parent_title if parent_title else title,
                "file_title": file_title
            }
            final_sections.append(section)
            return None


        for  md_line in md_lines:
            if code_fence_pattern.match(md_line):
                in_code_fence = not in_code_fence

            match = heading_pattern.match(md_line)
            if not in_code_fence and match:
                #如果当前行不在代码块中而且是标题行
                #收集上一个标题块的section
                _flush()
                #为当前标题赋值
                current_title = md_line
                #清空body
                body = []
                #将当前标题添加到hierarchy中并记录当前标题层级,"#"的个数
                #使用正则匹配的捕获组
                current_title_level = len(match.group(1))
                hierarchy[current_title_level] = current_title
                #清空hierarchy中层级比当前标题层级低的
                for i in range(current_title_level+1,7):
                    hierarchy[i] = ""

            else:
                #当前行并不是标题行
                body.append(md_line)
        # 循环结束后，保存最后一个 section
        if current_title or body:
            _flush()
        #3.
        return final_sections


    #二次切分或者合并
    def _split_and_merge(self, sections_by_head:List[Dict[str,Any]], min_content_length:int, max_content_length:int) -> List[Dict[str,Any]]:
        #1. 声明一个current_sections收集二次切分后的sections
        current_sections = []
        #2. 遍历current_sections
        for section in sections_by_head:
            #将比较长的sections切割
            split_sections = self._split_long_section(section,max_content_length)
            current_sections.extend(split_sections)

        #3. 对小于min_content_length的进行合并
        final_sections = self.merge_short_section(current_sections,min_content_length)
        return final_sections


    @staticmethod
    def _split_long_section(section:Dict[str,Any], max_content_length:int) -> List[Dict[str,Any]]:
        #将超过max_content_length的进行二次切分
        # {
        #       "body": "收集到的所有行"
        #       “title”:"当前内容的标题"
        #       "parent_title": 当前内容的父标题（最麻烦）
        #       "file_title": 文档标题（最简单）
        # }
        title = section.get("title")
        #防止标题过长，先用切片截取一下
        if len(title) > 80:
            title = title[:80]
        title_prifix = title + "\n\n"
        body = section.get("body")
        #对body中的表格进行处理
        if "<table>" in body:
            #将二维表格扁平化，用语言描述信息
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body
        #计算长度
        total_length = len(body)+len(title_prifix)
        if total_length <= max_content_length:
            #不用切分直接返回
            return [section]

        #需要切分，使用`RecursiveCharacterTextSplitter`文本切分器
        text_spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            chunk_size=max_content_length - len(title_prifix),
            keep_separator=True, #保留分隔符
            chunk_overlap=0 #重叠部分大小，重叠部分向量相同，如果检索内容与重叠部分语义相似会导致检索出两段语义不同的chunks，降低检索精确度
        )
        #使用切分器对body进行切分
        split_body_list = text_spliter.split_text(body)
        #防御性编程，如果没有切分，直接返回
        if len(split_body_list) == 1:
            return [section]
        #封装sections并返回
        result_sections = []
        for index,split_body in enumerate(split_body_list):
            section = {
                "title": title + f"_{index+1}" ,
                "body": split_body,
                "parent_title": section.get("parent_title"),
                "file_title": section.get("file_title"),
            }
            result_sections.append(section)
        return result_sections

    @staticmethod
    def merge_short_section(current_sections:List[Dict[str,Any]], min_content_length:int) -> List[Dict[str,Any]]:

        #1. 获取下标为0的section，记录为current_section
        current_section = current_sections[0]
        #2. 定义final_sections用于收集合并后的section列表
        final_sections = []
        #3. 从下标为1的section遍历到最后, 遍历出来的section记录为next_section
        for index in range(1,len(current_sections)):
            next_section = current_sections[index]
            # 判断next_section与current_section是否是同源: parent_title是否相同
            same_parent_title =  next_section.get("parent_title") == current_section.get("parent_title")
            #判断current_section的长度是否小于min_content_length
            current_section_body = current_section.get("body")
            is_short = len(current_section_body) < min_content_length
            #如果上面两个条件都满足，则使用current_section合并next_section:
            if is_short and same_parent_title:
                #使用"\n\n"拼接current_section与next_section的body
                merge_body = current_section_body + "\n\n" + next_section.get("body")
                current_section["body"] = merge_body
                #将合并后的section的标题title设置成他的parent_title，因为合并后用谁的标题都不合适，改成parent_title最合适
                current_section["title"] = current_section.get("parent_title")
            else:
                #如果不满足，则表示当前section不需要合并:
                #将current_section添加到final_sections中
                final_sections.append(current_section)
                #将current_section的指针指向next_section继续往下遍历
                current_section = next_section

        #4. 遍历完之后，将最后一个current_section添加到final_sections中
        final_sections.append(current_section)
        #5. 返回final_sections
        return final_sections

    @staticmethod
    def _assemble_chunks(final_sections) -> List[Dict[str,Any]]:
        chunks = []
        # 1. 遍历每一个section
        for section in final_sections:
            body = section.get('body')
            title = section.get('title')
            parent_title = section.get('parent_title')
            file_title = section.get('file_title')

            content = f"{title}\n\n{body}"

            chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            })
        return chunks


    def _backup_chunks(self, final_chunks:List[Dict[str,Any]],state:ImportGraphState) :
        file_dir = state["file_dir"]
        file_title = state["file_title"]
        md_path = state["md_path"]
        #获取文件输出目录
        md_path_obj = Path(md_path)
        file_dir_obj = Path(file_dir)
        backup_dir_obj = file_dir_obj / md_path_obj.stem
        #判断路径是否存在，不存在就创建
        if not backup_dir_obj.exists():
            backup_dir_obj.mkdir()
        #指定备份路径
        backup_file_path = backup_dir_obj / "chunks.json"
        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks,f,ensure_ascii=False,indent=4)
        except Exception as e:
            self.logger.warning(f"{file_title}: 备份切分结果失败，但不影响整个流程: {e}")


if "__main__" == __name__:
    md_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用_new.md"

    with open(md_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    md_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md"

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_dir": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir",
        "file_title":"test_spilt_1"
    }

    node = DocumentSplitNode()
    node(init_state)
