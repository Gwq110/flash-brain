from typing import Tuple, List, Dict, Any
from langchain_openai import ChatOpenAI
from knowledge.processor.query_processor import state
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message, get_recent_messages
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #获取用户的问题
        user_query = state.get('rewritten_query')
        #获取annswer
        answer = state.get('answer')
        #获取是否要流式输出
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')
        if answer:
            #如果已经有了answer说明在答案生成节点之前已经有回答了，直接输出
            self._push_and_exist_answer(answer, is_stream, task_id)
        else:
            #在答案生成节点之前，还没有回答，调用大模型生成回答
            #组装提示词
            prompt = self._build_answer_prompt(state,self.config.max_context_chars)
            #调用大模型生成答案
            llm_answer = self._generate_answer(state, prompt)
            state["answer"] = llm_answer
            #流式输出完之后吗，最终要告诉前端SSE通道关闭了
            if is_stream :
                push_sse_event(task_id=task_id,event=SSEEvent.FINAL,data={})

        # 保存历史对话信息到mongoDB
        self._save_history(state)



        return state



    def _push_and_exist_answer(self, answer, is_stream, task_id: str):
        #判断到底是流式输出还是非流式输出
        if is_stream:
            #流式输出，使用sse队列
            push_sse_event(task_id=task_id,event=SSEEvent.FINAL,data={'answer':answer})
        else:
            #非流式输出
            set_task_result(task_id=task_id,key="answer",value=answer)



    def _build_answer_prompt(self, state, max_context_chars) -> str:
        #获取用户的问题
        rewritten_query = state.get("rewritten_query")
        #获取商品列表
        item_names = state.get("item_names") or []
        #获取热rerank重排序后的文档列表
        reranked_docs = state.get("reranked_docs") or []
        #对reranked_docs进行格式规整化，并且进行截断
        formatted_content, used_chars = self._format_retrieval_content(reranked_docs, max_context_chars)
        #获取历史对话
        history = state.get("history") or []
        #TODO 对历史对话进行规整化，并且进行截断
        formatted_history = self._format_history(history, max_context_chars-used_chars)

        prompt = ANSWER_PROMPT.format(
            context = formatted_content,
            history = formatted_history,
            item_names = item_names,
            question = rewritten_query
        )
        return prompt


    def _generate_answer(self, state:QueryGraphState, prompt,):
        #1. 创建大模型对象
        llm_client = AIClients.get_llm_client(response_format=False)
        llm_result = ""
        #2. 判断是流式输出还是非流式输出
        if state.get('is_stream'):
            # 2.1 流式输出
            llm_result = self._llm_stream(state, llm_client, prompt)
        else:
            #2.2 非流式输出
            llm_result = self._llm_invoke(llm_client, prompt)
            #将生成的内容添加到任务队列中
            set_task_result(task_id=state.get("task_id"),key="answer",value=llm_result)
        return llm_result


    @staticmethod
    def _format_retrieval_content(reranked_docs, max_context_chars) -> Tuple[str, int]:
        """
        格式示例
        [1] [source=local] [chunk_id=chunk_001] [title=操作指导] [score=5.0600]
        测量直流电压时，将旋钮转到DCV档位，红表笔接VΩmA孔，黑表笔接COM孔。

        [2] [source=web] [url=https://example.com] [title=电压测量指南] [score=3.9600]
        注意：测量前请确认档位与量程，避免误接导致损坏或触电。
        """
        #定义已经使用的长度
        used_length = 0
        final_content = []
        #1. 遍历reranked_docs
        for index, doc in enumerate(reranked_docs, start=1):
            metadata_content = [f"[{index}]"]
            for field,template in [("chunk_id","[chunk_id={}]"),
                                   ("source", "[source={}]"),
                                   ("title","[title={}]"),
                                   ("url","[url={}]")]:
                # 获取字段值
                field_value = doc.get(field,"")
                # 判断字段值是否有
                if field_value:
                    metadata_content.append(template.format(field_value))
            #拼接当前文档的分数
            score = float(doc.get("score"))
            metadata_content.append(f"[score={score:.6f}]")
            #拼接当前文档的内容
            content = doc.get("content")
            format_chunk = " ".join(metadata_content) + "\n" + content
            #chunk_length= len(format_chunk) + 段与段之间的换行符长度2
            chunk_length = len(format_chunk) + len("\n\n")
            #判断是否会超出最大限制，
            if used_length + chunk_length > max_context_chars:
                break
            final_content.append(format_chunk)
            #更新已使用长度
            used_length += chunk_length

        return "\n\n".join(final_content),used_length


    def _llm_invoke(self, llm_client:ChatOpenAI, prompt:str) -> str:
        try:
            llm_res = llm_client.invoke(prompt)
        except Exception as e:
            return "LLM暂无输出内容"
        #获取回复内容
        llm_content = llm_res.content
        if not llm_content:
            return "LLM暂无输出内容"
        return llm_content


    def _llm_stream(self,state:QueryGraphState, llm_client:ChatOpenAI, prompt:str) -> str:
        final_answer = ""
        for chunk in llm_client.stream(prompt):
            #1.1 每得到一个内容块，将其推送到SSE队列中
            content = getattr(chunk, "content","")
            if content:
                push_sse_event(task_id=state.get("task_id"),event=SSEEvent.DELTA,data={'answer':content})

            #将所有chunk拼接并返回
            final_answer += content
        return final_answer


    def _save_history(self, state:QueryGraphState):
        #保存用户的对话内容
        save_chat_message(
            session_id=state.get("session_id"),
            role="user",
            text=state.get("original_query"),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names")
        )
        # 保存AI的对话内容
        save_chat_message(
            session_id=state.get("session_id"),
            role="assistant",
            text=state.get("answer"),
            item_names=state.get("item_names")
        )

    #历史对话格式规整化
    def _format_history(self, history: List[Dict[str, Any]], char_budget: int) -> Tuple[str, int]:
        """
        格式化历史对话
        Args:
            history: 历史对话
            char_budget:

        Returns:

        """
        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in history:
            # 1.1 获取消息角色
            role = msg.get('role', '')

            # 1.2 获取消息内容
            text = msg.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            # 1.4 计算分割符长度
            seperator_usage = 1 if formatted_lines else 0

            # 1.5 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines), char_budget - used_chars


if __name__ == '__main__':
    print("=" * 60)
    print("开始测试: 答案输出节点 (AnswerOutputNode)")
    print("=" * 60)

    mock_state = QueryGraphState()
    mock_state["rewritten_query"] = "我刚刚问你什么了？"
    mock_state["original_query"] = "我刚刚问你什么了？"
    mock_state["session_id"] = "2"
    mock_state["item_names"] = ["主板"]
    mock_state["reranked_docs"] = [
        {"chunk_id": "local_1", "title": "主板维修手册","source":"local",
         "content": "主板通电后通常表现为通电后风扇转一下不停，可以使用万用表的蜂鸣档测量。",
         "score": 5.06},
        {"chunk_id": "local_2", "title": "闲聊","url":"https://example.com","source":"web",
         "content": "今天中午去吃鸡腿饭饭吧，鸡腿饭很流行。",
         "score": 3.96},
    ]
    mock_state["task_id"] = "1"
    mock_state["is_stream"] = False

    histories = get_recent_messages("2")
    mock_state["history"] = histories

    node = AnswerOutputNode()
    final_state = node(mock_state)
    # print(final_state)

