import json
from pathlib import Path
from typing import Any
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class BgeEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_chunks_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 参数校验
        chunks = self._validate(state)
        #2. 每一个切片都嵌入成向量
        embedding_chunks = self._embedding_chunks(chunks)
        #3. 备份embedding_chunks，方便后续节点测试
        self._backup_chunks(embedding_chunks,state)

        state["chunks"] = embedding_chunks
        return state


    def _validate(self, state:ImportGraphState) -> list[dict[str, Any]]:
        #1. 获取chunks
        chunks = state.get("chunks")
        #2. 校验chunks
        if not chunks or not isinstance(chunks, list):
            self.logger.error("chunks is empty or not a list")
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=list)
        #检验每一个chunk的类型
        for chunk in chunks:
            if not isinstance(chunk, dict):
                self.logger.error("chunk {} is not a dict".format(chunk))
                raise ValidationError(node_name=self.name, message=f"chunk {chunk} is not a dict")
        # 返回chunks
        return chunks


    def _embedding_chunks(self, chunks:list[dict[str, Any]]) -> list[dict[str, Any]]:
        #1. 获取嵌入模型对象
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)
        #2. 从配置中获取`embedding_batch_size` ,这个配置表示批量嵌入的阈值，一批次最多嵌入多少条数据
        embedding_batch_size = self.config.embedding_batch_size
        #3. 获取chunks的总长度
        chunks_length = len(chunks)
        #4. 声明变量final_chunks，用于存储嵌入向量后的chunks
        final_chunks = []
        #5. 按照`batch_size`分批次遍历chunks
        for batch_start in range(0, chunks_length, embedding_batch_size):
            # 获取当前批次的chunks
            batch_chunks = chunks[batch_start:batch_start + embedding_batch_size]
            #获取要嵌入的内容，其实就是遍历当前批次的chunks，组装每一个chunk中的item_name和content，组装成新的数组
            document_list = [f"{chunk.get("item_name")}\n{chunk.get("content")}" for chunk in batch_chunks]
            #对document_list进行向量化
            try:
                embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, document_list)
            except EmbeddingError as e:
                self.logger.error(f"向量化错误：模型调用失败、向量生成异常，详情:{str(e)}")
                raise EmbeddingError(node_name=self.name,message=f"向量化错误：模型调用失败、向量生成异常，详情:{str(e)}")

            #遍历当前批次chunks
            #获取每一个chunk的稠密向量,稀疏向量
            dense_vector_list = embedding_result.get("dense")
            sparse_vector_list = embedding_result.get("sparse")

            #设置每一个chunk的稠密向量,设置每一个chunk的稀疏向量
            for index,chunk in enumerate(batch_chunks):
                chunk["dense_vector"] = dense_vector_list[index]
                chunk["sparse_vector"] = sparse_vector_list[index]
                final_chunks.append(chunk)

        return final_chunks

    def _backup_chunks(self, embedding_chunks, state):
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
        backup_file_path = backup_dir / "chunks_embedding.json"

        # 2. 将chunks写入到备份文件中
        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks_item_name.json失败,但是不影响主流程")
        pass


if __name__ == "__main__":
    chunks_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\chunks_item_name.json"
    chunks_path_obj = Path(chunks_path)

    with open(chunks_path_obj, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    state = {
        "md_path": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md",
        "file_dir": r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir",
        "chunks":chunks
    }

    node = BgeEmbeddingChunksNode()
    node(state)














