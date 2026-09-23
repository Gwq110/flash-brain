from typing import Any, List, Tuple, Dict
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class RRFMergeNode(BaseNode):
    name = "rrf_merge_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1. 获取混合检索和假设性文档嵌入检索的结果
        vector_search_chunks = state["embedding_chunks"] or []
        hyde_search_chunks = state["hyde_embedding_chunks"] or []
        #2. 对两路检索的结果进行格式规整化
        embedding_chunks = self._format_doc(vector_search_chunks)
        hyde_embedding_chunks = self._format_doc(hyde_search_chunks)

        #3. 将两路结果组装成列表，并且给设置权重
        rrf_inputs = [(embedding_chunks,1.0),(hyde_embedding_chunks,1.0)]


        #进行RRF融合
        sorted_chunks = self._rrf_merge(rrf_inputs)

        state["rrf_chunks"] = sorted_chunks
        return state


    @staticmethod
    def _format_doc(vector_search_chunks: List[dict[str,Any]]) -> List[dict[str,Any]]:
        """
        输入:["id":1,"distance":0.02,"entity":{'item_name': 'RS PRO RS-12 数字万用表','title': '万用表RS-12的使用','content': '万用表}]
        输出:["chunk_id":1,"title":标题,"content":内容]
        """
        format_chunks = []
        for chunk in vector_search_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue
            entity = chunk.get("entity")
            if not entity or not isinstance(entity, dict):
                continue
            title = entity.get("title")
            content = entity.get("content")
            if not content :
                continue
            format_chunks.append({
                "chunk_id": chunk["id"],
                "title": title,
                "content": content
            })
        return format_chunks

    def _rrf_merge(self, rrf_inputs:List[Tuple[List[dict[str,Any]],float]]) -> List[Dict[str, Any]]:
        #声明一个容器用于存储每个文档的总得分
        chunk_score = {}
        #声明一个容器，用来存储最终结果[{id,title,content,score}]
        chunk_data = {}
        #获取平滑指数
        k = self.config.rrf_k
        #遍历rrf_inputs
        for rrf_input in rrf_inputs:
            #获取当前路的权重和chunks
            chunks, weight = rrf_input

            # 遍历每一路检索结果的每一条
            for rank,chunk in enumerate(chunks,1): #代表下标从1开始
                #获取chunk_id
                chunk_id = chunk.get("chunk_id")
                #通过公式计算当前chunk在当前检索路径的得分
                score = weight/(k+rank)
                #统计得分
                chunk_score[chunk_id] = chunk_score.get(chunk_id, 0) + score
                #将当前文档的内容设置到chunk_data中
                chunk_data[chunk_id] = {**chunk,"score":chunk_score[chunk_id]}
        #将chunk_data收集到列表中，按照分数进行排序，排序完之后最后的结果不要分数
        result_chunks = chunk_data.values()
        #先排序
        sorted_chunks = sorted(result_chunks, key=lambda x: x["score"], reverse=True)
        return sorted_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"id":1,"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"id":2,"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"id":3,"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"id":2,"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"id":1,"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"id":3,"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RRFMergeNode()
    result = rrf_node(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")