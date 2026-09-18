import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class ItemNameRecognitionNode(BaseNode):
    node_name = "item_name_recognition_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 参数的校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        #2. 调用LLM 提取文档的商品名
        #2.1 准备提取商品名上下文
        context = self._build_item_name_context(chunks, item_name_chunk_k, item_name_chunk_size)
        #2.2 调用LLM识别商品名
        item_name = self._recognition_item_name(context, file_title)
        # print(f"item_name: {item_name}")
        #3. 将商品名进行向量化，获取混合向量
        hybrid_vectors = self._embedding_item_name(item_name)
        # print(f"hybrid_vectors: {hybrid_vectors}")
        #4. 存入向量数据库
        self._insert_to_milvus(item_name, hybrid_vectors)
        #5. 为了下游节点方便使用，将商品名写入chunks
        for chunk in chunks:
            if not isinstance(chunk, Dict):
                continue
            chunk["item_name"] = item_name

        state["item_name"] = item_name
        state["chunks"] = chunks

        #备份chunks，方便后面节点测试
        self._backup_chunks(chunks,state)
        return state



    def _validate_state(self, state) -> Tuple[str,List[Dict[str,Any]],int,int]:
        # 1.获取file_title，作为兜底使用，当没有提取到商品名的时候，使用file_title作为商品名
        file_title = state.get("file_title")
        # 2.校验file_title是否为空, 如果为空, 则表示前面出现了问题, 应该抛出异常
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)        # 3.获取chunks, 也就是上一步拆分之后的结果, 这个内容可以提供作为LLM的上下文信息, 让LLM进行商品名的提取
        chunks = state.get("chunks")
        # 4.判断chunks是否为空，且其类型是否是list，如果不满足则抛出异常
        if not chunks and type(chunks) is list:
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=list)
        # 5.从config中获取item_name_chunk_k和item_name_chunk_size，分别表示商品名提取能使用的最大chunk数量，以及最大的长度
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size
        # 6.判断item_name_chunk_k和item_name_chunk_size是否为空，并且是否大于0，如果不满足则抛出异常
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise StateFieldError(node_name=self.name, field_name="item_name_chunk_k", message="item_name_chunk_k must be greater than 0")

        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise StateFieldError(node_name=self.name, field_name="item_name_chunk_size",  message="item_name_chunk_size must be greater than 0")

        # 7.返回file_title、chunks、item_name_chunk_k、item_name_chunk_size
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size


    def _build_item_name_context(self, chunks:List[Dict[str,Any]], item_name_chunk_k, item_name_chunk_size) -> str:
        #定义容器用于存储最终上下文
        final_context = []
        total_size = 0
        #遍历前item_name_chunk_k个chunks
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, Dict):
                continue
            content = chunk.get("content")
            splice_context = f"【切片】-{index+1}-{content}"
            final_context.append(splice_context)
            total_size += len(splice_context)
            if total_size > item_name_chunk_size:
                break

        return "\n".join(final_context)


    def _recognition_item_name(self, context:str, file_title:str) -> str:
        # 1. 创建LLM对象，如果对象创建失败，则降级成使用file_title作为商品名
        try:
            llm_client = AIClients.get_llm_client(False)
        except Exception as e:
            self.logger.warning(f"获取llm对象失败，详情：{e}")
            return file_title
        # 2. 组装提示词，包含SystemPrompt、和UserPrompt
        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        system_message = SystemMessage(content=system_prompt)
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context=context,
        )
        human_message = HumanMessage(content=user_prompt)
        # 3. 调用大模型
        try:
            llm_res = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self.logger.warning(f"执行llm提取item_name失败,详情：{e}")
            return file_title
        # 4. 解析响应数据
        item_name = llm_res.content.strip()
        if item_name == "UNKNOWN":
            return file_title

        return item_name


    def _embedding_item_name(self, item_name:str) -> Dict[list,list]:
        #1. 创建embedding_client
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.warning(f"获取向量化模型失败：{e}")
            return None
        #2. 调用嵌入模型进行嵌入,获取混合向量
        hybrid_vectors = generate_bge_m3_hybrid_vectors(embedding_client, [item_name])
        return hybrid_vectors


    def _insert_to_milvus(self, item_name, hybrid_vectors):
        #创建milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取Milvus客户端失败,错误信息:{e}")
            raise MilvusError(message=f"获取Milvus客户端失败,错误信息:{e}",node_name=self.name)
        #判断存储item_name的colletion是否存在，不存在就创建
        collection_name = self.config.item_name_collection
        if not milvus_client.has_collection(collection_name):
            #创建表
            # Schema 用来规定 Collection 里面有哪些字段，以及每个字段是什么类型。
            schema = milvus_client.create_schema()
            # 2. 添加字段
            schema.add_field(
                field_name="pk",
                datatype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=10
            )
            schema.add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024 #维度
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            #添加索引
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
            #创建collection
            collection = milvus_client.create_collection(collection_name=collection_name, schema=schema,
                                                         index_params=index_params)

        #构建插入数据
        #获取稠密向量
        dense_vector = hybrid_vectors.get("dense")[0]
        # 获取稀疏向量
        sparse_vector = hybrid_vectors.get("sparse")[0]
        if dense_vector and sparse_vector:
            return

        insert_data = {
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector
        }
        milvus_client.insert(
            collection_name=collection_name,
            data=insert_data,
            timeout = 30
        )

    def _backup_chunks(self, chunks, state):
        # 1. 指定备份文件的路径
        # 1.1 获取文件输出的目录
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        # 1.2 判断该目录是否真实存在，如果不存在则创建目录
        backup_dir.mkdir(parents=True, exist_ok=True)
        # 1.3 指定备份文件的路径
        backup_file_path = backup_dir / "chunks_item_name.json"

        # 2. 将chunks写入到备份文件中
        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks_item_name.json失败,但是不影响主流程")


if __name__ == "__main__":
    json_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())
    md_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md"

    test_state = {
        "md_path": md_path,
        "file_title" : "万用表RS-12的使用",
        "file_dir": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir",
        "chunks":chunks
    }
    node = ItemNameRecognitionNode()
    node(test_state)