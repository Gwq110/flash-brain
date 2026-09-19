import json
from pathlib import Path
from typing import  Tuple, Any
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.storage_clients import StorageClients



class _IndexParamsBuilder:

    @classmethod
    def build_index_params(cls,milvus_client: MilvusClient):
        #创建索引
        index_params = milvus_client.prepare_index_params()
        #添加索引
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
            metric_type="IP"
        )
        return index_params



class _MilvusSchemaBuilder():

    @classmethod
    def build_schema(cls,milvus_client: MilvusClient,dim:int):
        #创建schema
        schema = milvus_client.create_schema()
        #标量
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            auto_id=True,
            is_primary=True
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        #向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        return schema


class MilvusImportNode(BaseNode):
    name = "milvus_import_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 参数校验
        final_chunks,dim = self._validate_state(state)

        #2. 向量入库
        self._insert_data(final_chunks, dim)

        print(final_chunks)
        return state

    def _validate_state(self, state) -> Tuple[list[dict[str,Any]],int]:
        #1. 获取chunks
        chunks = state.get("chunks")
        #2. chunks是否为空，是否为list
        if not chunks or not isinstance(chunks, list):
            self.logger.error("chunks is empty or not a list")
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        #3. 遍历每一个chunk，是否是dict,是否有向量
        final_chunks = []
        dim = 1024
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            #获取稠密向量和稀疏向量
            dense_vector = chunk.get("dense_vector")
            sparse_vector = chunk.get("sparse_vector")
            if not dense_vector or not sparse_vector:
                continue
            #获取dim(维度）,后续创建collection时需要指定维度，
            dim = len(dense_vector)
            final_chunks.append(chunk)

        #判断final_chunks是否为空
        if not final_chunks:
            self.logger.error(f"所有chunk都没有向量，导入失败")
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        return final_chunks,dim


    def _insert_data(self, final_chunks:list[dict[str,Any]], dim:int):
        # 1.创建MilvusClient
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取MilvusClient失败{e}")
            raise MilvusError(node_name=self.name,message = f"获取MilvusClient失败{e}")

        # 2.从配置中获取collection_name
        collection_name = self.config.chunks_collection
        # 3.判断collection是否存在，如果不存在则需要创建
        if not milvus_client.has_collection(collection_name):
            #3.1 创建schema(表结构）
            schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
            #3.2 创建IndexParams （索引参数）
            params = _IndexParamsBuilder.build_index_params(milvus_client)
            #3.3 创建collection
            milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=params)
        # 4. 插入数据，获取自增长的id
        insert_result = milvus_client.insert(
            collection_name=collection_name,
            data=final_chunks
        )
        # 5. 将自增长的id回填到chunks中
        ids = insert_result.get("ids")
        for i,chunk in enumerate(final_chunks):
            chunk["chunk_id"] = ids[i]





if __name__ == "__main__":
    chunks_path = r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\chunks_embedding.json"
    chunks_path_obj = Path(chunks_path)

    with open(chunks_path_obj, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    state = {
        "chunks": chunks
    }
    node = MilvusImportNode()
    node(state)
