import json
import re
from logging import Logger
from typing import Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE, ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class _ItemNameAligner:
    def __init__(self,logger:Logger, name:str, config:QueryConfig):
        self.logger = logger
        self._name = name
        self._config = config

    def search_and_align(self,item_names:List[str]) -> Tuple[List[str],List[str]]   :
        #1. 混合检索向量数据库
        search_result = self._search_vector(item_names)
        print("=" * 40)
        print(f"向量数据库初始检索结果：{search_result}")

        if not search_result:
            return [],[]
        #2. 根据混合向量检索到结果做对齐【confirmed/options 】
        confirmed, options = self._align(search_result)
        print("="*40)
        print(f"confirmed: {confirmed}")
        print(f"options: {options}")
        #4. 返回确定的confirmed容器和options容器
        return confirmed, options


    def _search_vector(self, item_names) -> List[Dict[str,Any]]:
        final_search_result = []
        #1. 创建嵌入对象模型
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"创建嵌入模型失败，{e}")
            return final_search_result

        #2. 对item_names进行向量化
        try:
            hybrid_vectors = generate_bge_m3_hybrid_vectors(embedding_client, item_names)
            dense_vector_list = hybrid_vectors.get("dense")
            sparse_vector_list = hybrid_vectors.get("sparse")
        except Exception as e:
            self.logger.error(f"item_names进行向量化失败，{e}")
            return final_search_result

        #创建milvus客户端对象
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"创建milvus客户端对象失败，{e}")
            return final_search_result

        #3. 创建混合索引请求
        #3.1 对item_names进行遍历
        for index, item_name in enumerate(item_names):
            search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector_list[index],
                sparse_vector=sparse_vector_list[index],
                limit=self._config.embedding_search_limit
            )
            #执行混合检索,结果是在向量数据库中对item_name检索topk条数据
            milvus_search_results =  execute_hybrid_search_query(
                milvus_client = milvus_client,
                collection_name = self._config.item_name_collection,
                search_requests = search_requests,
                limit=self._config.embedding_search_limit,
                output_fields=["item_name"]
            )
            #遍历结果数组并组装匹配结果,
            matches = []
            for search_result in milvus_search_results[0]:
                matches.append({
                    "item_name": search_result["entity"]["item_name"],
                    "score": search_result["distance"]
                })

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })
        return final_search_result


    def _align(self, search_result:List[Dict[str,Any]]) -> Tuple[List[str],List[str]]:
        #1. 定义两个容器用来存储已经确定的商品名和没有待定的商品名
        confirmed = []
        options = []
        #2. 遍历检索结果
        for result in search_result:
            extracted_name = result["extracted_name"]
            matches = result["matches"]
            #对matchs按照score从大到小排序
            matches = sorted(matches, key=lambda x: x["score"], reverse=True)
            #找到高可置信的结果
            high_result = [m["item_name"] for m in matches if m["score"] >= self._config.item_name_high_confidence]
            #如果有高可信的结果
            # 如果有高可信结果
            if high_result:

                # 1. 优先判断：向量数据库结果中是否有与大模型提取结果完全一致的
                exact_match_name = [item for item in high_result if item == extracted_name]

                if exact_match_name:
                    if exact_match_name not in confirmed:
                        confirmed.append(exact_match_name)

                # 2. 没有完全匹配，但只有一个高可信结果
                elif len(high_result) == 1:
                    if high_result[0] not in confirmed:
                        confirmed.append(high_result[0])

                # 3. 没有完全匹配，并且有多个高可信结果
                else :
                    # 判断第一名和第二名之间是否存在明显分差
                    max_score = matches[0]["score"]
                    second_score = matches[1]["score"]
                    if max_score - second_score > self._config.item_name_score_gap:
                        #存在明显分差
                        max_score_item_name = matches[0]["item_name"]
                        if max_score_item_name not in confirmed: #去重
                            confirmed.append(max_score_item_name)
                    else:
                        #不存在存在明显分差
                        for item in high_result[:self._config.item_name_max_options]:
                            if item not in confirmed and item not in options:
                                options.append(item)

            else:
                #没有高可置信结果
                #收集中等可置信结果
                middle_result = [m["item_name"] for m in matches if m["score"] >= self._config.item_name_mid_confidence ]
                if middle_result:
                    for item in middle_result[:self._config.item_name_max_options]:
                        if item not in confirmed and item not in options:
                            options.append(item)

        return confirmed,options







class  _ItemNameExtractor:
    def __init__(self,logger:Logger, name:str):
        self.logger = logger
        self._name = name

    def extract_item_name(self,original_query:str,history_text:str):
        # 定义大模型返回的结构
        llm_result = {
            "item_names": [],
            "rewritten_query": original_query
        }
        #1. 创建llm对象
        try:
            llm_client = AIClients.get_llm_client()
        except Exception as e:
            return llm_result
        #2. 组装prompt消息
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            query=original_query,
            history_text=history_text
        )
        system_prompt = ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=user_prompt)

        #3. 调用大模型拿到结果
        try:
            llm_res = llm_client.invoke([system_message, human_message])
        except Exception as e:
            return llm_result

        #4. 清洗结果
        llm_content = llm_res.content

        if not llm_content:
            return llm_result
        #5. 返回清洗结果
        return self._clean_and_parse(llm_content)

    def _clean_and_parse(self, llm_content:str) -> Dict[str,Any]:
        #1. 去掉代码块
        #1.1 去掉前面的 ```
        content = re.sub(r"^```(?:json)?\s*","",llm_content)
        #1.2 去掉后面的```
        content = re.sub(r"\s*```$","",content)

        #2. 将llm_content进行反序列化
        llm_content_obj : Dict[str,Any] = json.loads(content)
        #2.1 获取item_names
        original_item_names = llm_content_obj.get("item_names")
        #2.2 判断original_item_names的类型
        if not isinstance(original_item_names,list):
            item_names = []
        else:
            # 将original_item_names中的每一个字符串去空格之后，收集到item_names中
            item_names = [item_name.strip() for item_name in original_item_names if isinstance(item_name,str) and item_name.strip()]

        #2.3 获取rewritten_query
        original_rewritten_query = llm_content_obj.get("rewritten_query")
        #2.4 判断original_rewritten_query的类型
        if not isinstance(original_rewritten_query,str):
            rewritten_query = ""
        else:
            rewritten_query = original_rewritten_query.strip()

        #3. 返回Dict
        return {"item_names":item_names,"rewritten_query":rewritten_query}


class ItemNameComfirmedNode(BaseNode):
    name = "item_name_comfirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = _ItemNameAligner(self.logger, self.name,self.config)


    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
                主要职责：
                1. 利用LLM从用户原始查询中提取商品名以及改写原始问查询（我喜欢你）
                1.1 如果LLM提取到了商品名，才进行第2步 去milvus对齐
                1.2 如果LLM没有提取到商品名，直接返回
                2. 根据Milvus中存储的商品名进行对齐（目的：检索更加的准确：三路检索都会利用该节点提取到的商品名，因此直接用LLM提取到商品名的话 下游三路检索在过滤的时候，过滤条件极其不准确。导致检索到的噪音很多 LLM最终输出的幻觉很高）
                最终不是要LLM的商品名 而是要Milvus中存储的商品名：因为milvus中没一个chunK都会关联milvus自己的商品名
                3. 决策（该走下去，还是回头）

                利用两个容器，产生三个分支：第一个分支去检索  第二个分支：给用户确认  第三个分支：抱歉
                1. confirmed:如果是精确的商品名--->给confirmed添加精确的商品名
                2. options:商品名不是精确，可是找到多个相似的---->给options中添加找到的多个不精确的商品名。

                state['answer']不要给，进行三路检索
                获取到三路检索结果
                把三路检索到的结果(RRF  RERANKER)给LLM
                LLM生成答案,在state['answer']
                state['answer']:就返回：
                1. 返回候选商品名【不精确】，给用户下一步确认使用
                2. 没有任何商品名，返回抱歉，没有找到您询问的关于任何商品的名字
                Args:
                    state:
                Returns:
        """
        #1. 获取用户问题
        original_query = state.get('original_query')

        #TODO 2. 获取历史对话
        history_text = ""

        #3. 获取大模型提取的用户提问中的商品名
        llm_result = self._extractor.extract_item_name(original_query, history_text)
        print(llm_result)

        #4. 根据item_names做判断,进行商品名的对齐
        confirmed, options = self._item_name_aligner.search_and_align(llm_result["item_names"])

        #5. 决策
        self._decide(confirmed, options, state, llm_result["rewritten_query"])

        return state

    def _decide(self, confirmed:List[str], options:List[str], state:QueryGraphState, rewritten_query:str):
        #判断confirmed里面是否有数据
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = f"我不能确认你指的是哪款产品,您是在询问以下产品:{'、'.join(options)}吗"
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"



if __name__ == "__main__":
    node = ItemNameComfirmedNode()
    state = {
        "original_query": '特斯拉model3怎么开启自动驾驶',
    }

    final_state = node(state)
    print("=" * 40)
    print(f"final_state:{final_state}")
