import asyncio
import json
from typing import Tuple, List
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import httpx2
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class WebSearchNode(BaseNode):
    name = "web_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)
        # 2. 调用mcp服务获取结果（使用mcp原生SDK，绕过openai-agents封装的兼容性bug）
        mcp_web_search = asyncio.run(self.mcp_web_search(rewritten_query))
        #3. 对mcp结果进行格式化
        #获取结果中的text，是一个json字符串
        result_text = mcp_web_search.content[0].text
        json_obj = json.loads(result_text)
        pages = json_obj["pages"]
        web_search_docs = []
        for page in pages:
            #获取snippet，title，url
            web_search_docs.append({
                "snippet": page["snippet"],
                "title": page["title"],
                "url": page["url"]
            })

        #返回结果（只返回本节点负责的字段，避免并行节点同时写同一 key 冲突）
        return {"web_search_docs": web_search_docs}

    async def mcp_web_search(self, rewritten_query: str):
        """使用mcp原生SDK直接连接DashScope WebSearch MCP服务"""
        # 1. 创建带认证信息的HTTP客户端
        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {self.config.openai_mcp_key}"},
            timeout=httpx2.Timeout(connect=30, read=300, write=300, pool=30),
        )
        try:
            # 2. 连接MCP服务
            mcp_url = self.config.mcp_dashscope_base_url
            async with streamable_http_client(
                url=mcp_url,
                http_client=http_client,
                terminate_on_close=True,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # 3. 初始化会话
                    await session.initialize()
                    # 4. 调用工具，获取联网内容
                    result = await session.call_tool(
                        "bailian_web_search",
                        arguments={"query": rewritten_query, "count": 5}
                    )
                    return result
        finally:
            # 5. 确保HTTP客户端被关闭
            await http_client.aclose()

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state["rewritten_query"]
        item_names = state["item_names"]
        # 校验属性
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query不能为空且类型必须为str,{rewritten_query}")
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"item_names不能为空且类型必须为list,{item_names}")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names


if __name__ == "__main__":
    # 1. 创建state
    state = {
        "rewritten_query": "RS PRO RS-12 数字万用表怎么使用？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }
    node = WebSearchNode()
    final_state = node(state)
    print(final_state)