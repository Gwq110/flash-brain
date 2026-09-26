from typing import List, Tuple
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError, MilvusError, EmbeddingError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1. 参数校验
        rewritten_query,item_names = self._validate_state(state)
        #2. 获取依赖，嵌入模型和向量
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败,{e}")
            raise EmbeddingError(node_name=self.name,message=f"获取嵌入模型失败,{e}")
        #获取数据库客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败,{e}")
            raise MilvusError(node_name=self.name,message=f"获取milvus客户端失败,{e}")

        #3. 生成稠密向量和稀疏向量
        embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, [rewritten_query])
        #4. 构建过滤条件
        expr, expr_params = item_names_filter(item_names)
        #5. 创建搜索请求
        hybrid_search_request = create_hybrid_search_requests(
            dense_vector=embedding_result.get("dense")[0],
            sparse_vector=embedding_result.get("sparse")[0],
            expr=expr,
            expr_params=expr_params,
            limit=self.config.embedding_search_limit
        )
        #6. 执行混合检索
        hybrid_search_result = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_request,
            limit=self.config.embedding_search_limit,
            output_fields=["item_name","title","content"]
        )
        #返回结果（只返回本节点负责的字段，避免并行节点同时写同一 key 冲突）
        return {"embedding_chunks": hybrid_search_result[0]}


    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state["rewritten_query"]
        item_names = state["item_names"]
        #校验属性
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query不能为空且类型必须为str,{rewritten_query}")
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"item_names不能为空且类型必须为list,{item_names}")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query,item_names


if __name__ == "__main__":
    #1. 创建state
    state = {
        "rewritten_query": "RS PRO RS-12 数字万用表怎么使用？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }
    node = VectorSearchNode()
    final_state = node(state)
    print(final_state)
