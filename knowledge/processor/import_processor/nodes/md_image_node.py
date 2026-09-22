import base64
import logging
import re
from logging import Logger
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAI
from knowledge.processor.import_processor import config
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients


class MdFileHandler():
    def __init__(self,logger:Logger, node_name:str) :
        self.node_name = node_name
        self.logger = logger

    def read_md(self,md_path: str) -> tuple[str, Path, Path]:
        #1. 校验md_path是否为空
        if not md_path:
            self.logger.error("md_path 不能为空")
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)
        #2. 校验md_path是否存在
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            self.logger.error(f"md_path：{md_path_obj} 不存在")
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str,message="md_path路径无效，文件不存在")
        #3. 获取md文件的内容到内存
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content=f.read()
        #4. 获取md文件的图片的目录路径
        image_dir_obj = md_path_obj.parent / "images"
        #5. 返回md_path_obj,md_content,image_dir_obj
        return md_content,md_path_obj,image_dir_obj

    def backup_md(self,md_content:str,md_path:str) -> None:
        #构建新文件路径
        md_path_obj = Path(md_path)
        backup_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)
        #将新文件写入到目标目录
        try:
            with open(backup_md_path_obj,"w",encoding="utf-8") as f:
                f.write(md_content)
        except Exception as e:
            self.logger.error(f"备份md文件失败：{e}")
        self.logger.info(f"备份md文件成功：文件路径{backup_md_path_obj}")


#照片上下文对象
class _ImageContext:
    pre_context: str
    post_context: str
    head_title: str
    def __init__(self,pre_context:str,post_context:str,head_title:str):
        self.pre_context = pre_context
        self.post_context = post_context
        self.head_title = head_title

#图片信息对象
class _ImageInfo:
    image_name: str
    image_path: Path
    image_context: _ImageContext
    def __init__(self, image_name:str, image_path:Path, image_context:_ImageContext):
        self.image_name = image_name
        self.image_path = image_path
        self.image_context = image_context


class _ImageScanner():
    def __init__(self, logger:Logger, config: config.ImportConfig) :
        self.logger = logger
        self.config = config

    def scan_image_dir(self,  image_dir_obj: Path, md_content: str) -> List[_ImageInfo]:
        #创建用于存储照片信息的list容器
        image_info_list = []
        # 2. 按行切分md文件
        md_lines = md_content.split("\n")
        # 1.1 图片目录不存在时（例如直接上传的md文件），跳过图片处理
        if not image_dir_obj.is_dir():
            self.logger.warning(f"图片目录不存在，跳过图片处理：{image_dir_obj}")
            return image_info_list
        # 3，遍历图片目录的每一个图片文件，与上面md_lines中的进行匹配
        for image_file in image_dir_obj.iterdir():
            # 过滤子目录
            if not image_file.is_file():
                continue
            # 过滤非图片目录
            if image_file.suffix not in self.config.image_extensions:
                continue
            #定义图片文件匹配正则表达式
            image_file_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file.name) + r".*?\)")
            #定义代码块标识符正则表达式
            code_fence_pattern = re.compile(r'^\s*```')
            #标识本行是否在代码块中,默认没在
            in_code_fence = False
            for index, md_line in enumerate(md_lines):
                if  code_fence_pattern.match(md_line):
                    in_code_fence = not in_code_fence #反转状态
                #如果在代码块中或者没有匹配到照片文件就跳过
                if in_code_fence or not image_file_pattern.match(md_line):
                    continue
                #已经匹配到图片，开始查找上下文
                # 匹配标题的正则表达式
                heading_pattern = re.compile(r'^#{1,6}\s+(.+)$')
                #查找图片的上文
                head_title,head_content = self._find_heading_content(md_lines, heading_pattern, index)
                #查找图片的下文
                next_context = self._find_next_content(md_lines, heading_pattern, index)
                #如果上下文太长，需要截取
                img_content_length = self.config.img_content_length
                #截取上下文内容
                extract_next_context = self._extract_image_context(next_context, img_content_length)
                extract_head_context = self._extract_image_context(head_content, img_content_length,True)
                #组装照片上下文对象
                image_context = _ImageContext(post_context=extract_next_context, pre_context=extract_head_context, head_title=head_title)
                #组装照片信息对象
                image_info = _ImageInfo(image_name=image_file.name, image_path=image_dir_obj/image_file.name, image_context=image_context)
                #存入数据
                image_info_list.append(image_info)

        return image_info_list



    #查找图片上文的方法
    def _find_heading_content(self,md_lines: List[str], heading_pattern: re.Pattern, image_index: int) -> tuple[str, list[str]]:
        # 定义代码块标识符正则表达式
        code_fence_pattern = re.compile(r'^\s*```')
        # 标识本行是否在代码块中,默认没在
        in_code_fence = False
        #定义找到的最近的标题
        head_title_index = -1
        head_title = ""
        #向上遍历，直到遇到第一个标题，range的语法是 左闭右开，目前表示范围为[image_index-1,-1)
        for i in range(image_index-1,-1,-1):
            if code_fence_pattern.match(md_lines[i]):
                in_code_fence = not in_code_fence
            #匹配标题
            if  not in_code_fence and heading_pattern.match(md_lines[i]):
                head_title_index = i
                head_title = md_lines[head_title_index].strip()
                break
        #获取上文切片，语法 [),左闭右开
        head_content = md_lines[head_title_index+1:image_index]
        return head_title,head_content

    # 查找图片下文的方法

    def _find_next_content(self,md_lines: List[str], heading_pattern: re.Pattern, image_index: int) -> List[str]:
        # 定义代码块标识符正则表达式
        code_fence_pattern = re.compile(r'^\s*```')
        # 标识本行是否在代码块中,默认没在
        in_code_fence = False
        # 定义找到的最近的标题
        next_title_index =len(md_lines)
        # 向上遍历，直到遇到第一个标题，range的语法是 左闭右开，目前表示范围为[image_index-1,-1)
        for i in range(image_index+1, len(md_lines)):
            if code_fence_pattern.match(md_lines[i]):
                in_code_fence = not in_code_fence
            # 匹配标题
            if not in_code_fence and heading_pattern.match(md_lines[i]):
                next_title_index = i
                break
        # 获取上文切片，语法 [),左闭右开
        head_content = md_lines[image_index+1:next_title_index]
        return  head_content

    def _extract_image_context(self, context: list[str],img_content_length: int, is_head: bool = False) -> str:
        image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
        #定义存储当前段落上下文的容器
        current_graph = []
        #定义存储最终段落上下文的容器
        final_graph = []
        #遍历所有上下文
        for line in context:
            #如果当前为空行或者其他图片就尝试将当前current_graph内容写入到final_graph，并清空current_graph
            if not line.strip() or image_pattern.match(line):
                if current_graph:

                    final_graph.append("\n".join(current_graph))
                    current_graph = []
            else:
                current_graph.append(line)
        #循环结束后防止最后一行不是空行或者照片而且current_graph有内容的情况，再写入一次
        if current_graph:
            final_graph.append("\n".join(current_graph))

        #遍历final_graph,对长度进行判断，如果超过了最大长度就进行截取
        extract_graphs = []
        #如果是截取上文需要反转段落，因为距离图片越近的信息与照片关系越密切
        if is_head:
            final_graph.reverse()

        total_length = 0
        for graph in final_graph:
            if total_length > img_content_length:
                break
            extract_graphs.append(graph)
            total_length += len(graph)+len("\n\n")

        #反转后段落要保持原来顺序
        if is_head:
            extract_graphs.reverse()

        return "\n\n".join(extract_graphs)

class _VLMSummarizer:
    def __init__(self,logger: Logger,config: config):
        self.logger = logger
        self.config = config

    def summarize_all_image(self,document_name:str,image_info_list:List[_ImageInfo]):
        summaries = {}
        # 1. 获取vlm客户端对象
        try:
            vlm_client = AIClients.get_vlm_client()
        except ConnectionError as e:
            self.logger.warning(f"获取VLM客户端失败，无法提取图片摘要信息: {str(e)}")
            #获取客户端失败，则将所有图片摘要设置为"暂无摘要"
            for image_info in image_info_list:
                summaries[image_info.image_name] = "暂无摘要"
        #2.遍历每一张图片信息，逐个获取摘要并保存
        for image_info in image_info_list:
            image_summarize = self._summarize_sigle_image(document_name, image_info, vlm_client)
            summaries[image_info.image_name] = image_summarize
        return summaries

    def _summarize_sigle_image(self, document_name : str, image_info : _ImageInfo,  vlm_client: OpenAI) -> str:
        #1. 构建图片上下文字符串#组装final_context
        image_context = image_info.image_context
        final_context = "\n\n".join([image_context.head_title, image_context.pre_context, image_context.post_context])
        #2.读取图片信息，要转换为base64格式
        # 2. 读取图片内容
        try:
            with open(image_info.image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            self.logger.warning(f"读取图片文件失败，无法提取图片摘要信息: {str(e)}")
            return "暂无摘要"

        # 3. 执行vlm
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ]
                }
            ]
            vlm_result = vlm_client.chat.completions.create(
                model=self.config.vlm_model,
                messages=messages
            )
            return vlm_result.choices[0].message.content.strip() if vlm_result.choices else "暂无摘要"
        except Exception as e:
            self.logger.warning(f"调用VLM模型提取图片摘要信息失败: {str(e)}")
            return "暂无摘要"


class ImageUploader:
    def __init__(self,logger: Logger,config: config):
        self.logger = logger
        self.config = config


    def upload_and_replace(self,md_content: str, document_name : str,image_info_list :List[_ImageInfo],image_summarizes:Dict[str,str]):
        #上传图片
        remote_urls = self._upload_all(document_name, image_info_list)
        #替换md中的内容，返回替换之后的
        md_lines = md_content.split("\n")
        new_md_content = self._replace_md_content(md_lines, remote_urls, image_summarizes)
        #返回新的md内容
        return new_md_content

    def _upload_all(self,document_name : str,image_info_list :List[_ImageInfo]):
        remote_urls = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            self.logger.warning(f"获取MinIO客户端失败，无法上传图片: {str(e)}")
            #兜底，所有图片使用原来路径
            for image_info in image_info_list:
                remote_urls[image_info.image_name] = image_info.image_path
            return remote_urls

        #将图片上传到minIo
        #构建object_name: 其实就是文件存储到MinIo上的路径。 文档名/图片名   例如：万用表RS-12的使用/1.jpg
        for image_info in image_info_list:
            object_name = document_name + "/" + image_info.image_name
            bucket_name = self.config.bucket_name
            try:
                minio_client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=image_info.image_path
                )
            except Exception as e:
                remote_urls[image_info.image_name] = image_info.image_path
            #组装文件再MinIo上的url
            remote_url = self.config.get_minio_base_url() + "/" + bucket_name + "/" + object_name
            remote_urls[image_info.image_name] = remote_url

        return remote_urls

    def _replace_md_content(self, md_lines: List[str], remote_urls: Dict[str,str], image_summarizes: Dict[str,str]) -> str:
        # 定义正则表达式匹配md中的图片，定义两个匹配组出来，第一个匹配组用于匹配摘要，第二个匹配组用于匹配路径
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        # 定义代码块标识符正则表达式
        code_fence_pattern = re.compile(r'^\s*```')
        #是否在代码块中
        in_code_dence = False
        #声明一个容器用于存储替换后的md_lines
        new_md_lines = []
        for line in md_lines:
            if code_fence_pattern.match(line):
                in_code_dence = not in_code_dence
            match = pattern.match(line)
            if not in_code_dence and match:
                image_name = Path(match[2]).name
                new_md_lines.append(f"![{image_summarizes[image_name]}]({remote_urls[image_name]})")
            else:
                #不需要替换
                new_md_lines.append(line)

        return "\n".join(new_md_lines)


class MdImageNode(BaseNode):
    name = "md_image_node"
    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(logger=self.logger, node_name=self.name)
        self.image_dir_handler = _ImageScanner(logger=self.logger, config=self.config)
        self.image_summarizer = _VLMSummarizer(logger=self.logger, config=self.config)
        self.image_uploader = ImageUploader(logger=self.logger, config=self.config)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 读取md文件内容到内存
        md_path = state.get("md_path")
        md_content,md_path_obj,image_dir_obj = self.md_file_handler.read_md(md_path)

        #2. 提取图片上下文，获取图片信息
        image_info_list = self.image_dir_handler.scan_image_dir(image_dir_obj, md_content)
        #4. 根据图片上下文，使用vlm模型，获取图片的描述信息
        image_summarizes = self.image_summarizer.summarize_all_image(md_path_obj.stem, image_info_list)
        #5. 将图片上传到minIO，获取url，将图片的描述信息，minio中图片的url回填到md中
        new_md_content = self.image_uploader.upload_and_replace(md_content=md_content, document_name=md_path_obj.stem,
                                                         image_info_list=image_info_list,
                                                         image_summarizes=image_summarizes)
        # print(new_md_content)

        #6. 将回填后的内容，备份成新md文件（便于测试观察）
        self.md_file_handler.backup_md(md_content=new_md_content,md_path=md_path)

        state["md_content"] = new_md_content
        return state

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = MdImageNode()
    state = {
        "md_path": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md",
    }
    node(state)