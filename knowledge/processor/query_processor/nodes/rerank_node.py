import math
from typing import List, Dict, Any
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients, _BgeCrossEncoderRerankClient


class RerankNode(BaseNode):
    name = "rerank_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1. 获取用户的问题
        user_query = state["rewritten_query"]
        #2. 获取rrf合并后的结果,网络搜索的结果
        rrf_merge_docs = state["rrf_chunks"]
        web_search_docs = state["web_search_docs"]
        #3. 将这两个结果先进行规整化
        rrf_formated_docs = self._format_rrf_docs(rrf_merge_docs)
        web_search_formated_docs = self._format_web_search_docs(web_search_docs)
        #4. 将两个结果合并
        rrf_formated_docs.extend(web_search_formated_docs)
        final_docs = rrf_formated_docs
        #5. 使用重排序模型计算文档得分
        doc_score_pairs = self._refine_rerank(user_query, final_docs)
        #6. 按照得分从高到底进行排序
        sorted_doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x["score"], reverse=True)
        #7.动态截断，断崖检测
        reranked_docs = self._cliff_cutoff(sorted_doc_score_pairs)
        state["reranked_docs"] = reranked_docs

        return state


    def _format_rrf_docs(self, rrf_merge_docs:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        #规整后最终结果
        """
        {
            "chunk_id":1,"
            "title":"",
            "content":"",
            "source":""
        """
        formated_docs = []
        for chunk in rrf_merge_docs:
            chunk_id = chunk["chunk_id"]
            title = chunk["title"]
            content = chunk["content"]
            formated_docs.append({
                "chunk_id": chunk_id,
                "title": title,
                "content": content,
                "source": "local"
            })
        return formated_docs


    def _format_web_search_docs(self, web_search_docs):
        # 规整后最终结果
        """
        {
            "title":"",
            "content":"",
            "source":""
        """
        formated_docs = []
        for chunk in web_search_docs:
            url = chunk["url"]
            title = chunk["title"]
            content = chunk["snippet"]
            formated_docs.append({
                "title": title,
                "content": content,
                "url": url,
                "source": "web"
            })
        return formated_docs


    def _refine_rerank(self, user_query:str, final_docs:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        #1. 构建Q-D对
        question_document_pairs = [(user_query, docs["content"]) for docs in final_docs]
        #2. 创建rerank模型对象
        try:
            rerank_client:_BgeCrossEncoderRerankClient = AIClients.get_bge_m3_rerank_client()
        except Exception as e:
            self.logger.warn(f"获取rerank模型对象失败，{e}")
            return [{**doc,"score":None} for doc in final_docs]
        #调用模型计算得分
        try:
            scores = rerank_client.compute_score(question_document_pairs)
            return [{**doc,"score":self._sigmoid(score)} for doc,score in zip(final_docs, scores)]
        except Exception as e:
            self.logger.warn(f"调用rerank模型计算相关性得分失败，{e}")
            return [{**doc, "score": None} for doc in final_docs]

    @staticmethod
    def _sigmoid(score: float) -> float:
        #  使用 sigmoid 将模型原始分数归一化到 (0, 1)
        # math.exp(-score) 等价于e的（-score）次方
        # 随着 score 的增大，math.exp(-score) 的结果会迅速趋近于 0；随着 score 的减小（变为负数），结果会迅速增大
        return 1.0 / (1.0 + math.exp(-score))


    def _cliff_cutoff(self, sorted_doc_score_pairs):
        #获取相关参数
        rerank_max_top_k = self.config.rerank_max_top_k
        rerank_min_top_k = self.config.rerank_min_top_k
        rerank_gap_abs = self.config.rerank_gap_abs
        #定义截取的上下边界
        upper_bound = min(rerank_max_top_k,len(sorted_doc_score_pairs))
        lower_bound = min(rerank_min_top_k,upper_bound)

        #遍历文档列表，计算最大断崖点的下标
        max_score_gap = 0
        cut_index = upper_bound
        for i in range(0,upper_bound-1):
            current_score = sorted_doc_score_pairs[i]["score"]
            next_score = sorted_doc_score_pairs[i+1]["score"]
            score_gap = current_score - next_score
            if current_score is None or next_score is None:
                continue
            if score_gap > rerank_gap_abs and score_gap > max_score_gap:
                #变更最大插值
                max_score_gap = score_gap
                #变更断崖点
                cut_index = i+1
        cut_index = max(lower_bound,cut_index)

        return sorted_doc_score_pairs[:cut_index]




if __name__ == "__main__":

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")