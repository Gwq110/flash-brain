from typing import Tuple, List
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.import_processor.exceptions import EmbeddingError
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import MilvusError, LLMError, StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE, HYDE_SYSTEM_PROMPT_TEMPLATE
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class HydeSearchNode(BaseNode):
    name = "hyde_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        #2. 调用大模型生成假设性答案
        hy_document = self._generate_hy_document(rewritten_query, item_names)
        #3. 获取嵌入模型对象和milvus客户端对象
        # 获取嵌入模型对象
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败,{e}")
            raise EmbeddingError(node_name=self.name, message=f"获取嵌入模型失败,{e}")
        # 获取数据库客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败,{e}")
            raise MilvusError(node_name=self.name, message=f"获取milvus客户端失败,{e}")
        #4. 将用户的问题和假设性答案进行拼接
        embedding_input = f"{rewritten_query}\n{hy_document}"
        #5. 将拼接内容进行向量化
        embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, [embedding_input])
        #6. 构建过滤条件
        expr, expr_params = item_names_filter(item_names)
        # 5. 创建搜索请求
        hybrid_search_request = create_hybrid_search_requests(
            dense_vector=embedding_result.get("dense")[0],
            sparse_vector=embedding_result.get("sparse")[0],
            expr=expr,
            expr_params=expr_params,
            limit=self.config.hyde_search_limit
        )
        # 6. 执行混合检索
        hybrid_search_result = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_request,
            limit=self.config.hyde_search_limit,
            output_fields=["item_name", "title", "content"]
        )
        #9. 返回本节点负责的字段（只返回部分更新，避免并行节点同时写同一 key 冲突）
        return {"hyde_embedding_chunks": hybrid_search_result[0]}

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


    def _generate_hy_document(self, rewritten_query:List[str], item_names:List[str]) -> str:
        # 获取大模型对象
        try:
            llm_client = AIClients.get_llm_client(response_format = False)
        except Exception as e:
            self.logger.error(f"获取大模型失败,{e}")
            raise LLMError(node_name=self.name, message=f"获取嵌入模型失败,{e}")
        #构建上下文
        hyde_user_prompt  = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names = item_names,
            rewritten_query = rewritten_query
        )
        hyde_system_prompt = HYDE_SYSTEM_PROMPT_TEMPLATE.format(item_names=item_names)

        system_message= SystemMessage(content=hyde_system_prompt)
        human_message = HumanMessage(content=hyde_user_prompt)

        #调用大模型
        try:
            llm_result = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self.logger.error(f"调用大模型客户端失败,{e}")
            raise LLMError(node_name=self.name, message=f"调用大模型客户端失败,{e}")

        return llm_result.content


if __name__ == "__main__":
    #1. 创建state
    state = {
        "rewritten_query": "RS PRO RS-12 数字万用表怎么使用？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }
    node = HydeSearchNode()
    final_state = node(state)
    print(final_state)

