from langgraph.constants import END
from langgraph.graph import StateGraph
from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.md_image_node import MdImageNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState


#langGraph 的具体流程
def import_router(state: ImportGraphState) :
    #1. 判断是不是pdf文件
    if state.get("is_pdf_read_enabled"):
        #是pdf文件
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled"):
        # 是md文件
        return "md_image_node"
    else:
        return END



def import_graph():
    #1. 创建StateGraph,将state传给StateGraph
    work_flow = StateGraph(state_schema=ImportGraphState)
    #2. 添加节点
    #2.1 声明节点列表
    node_list = {
        "entry_node" : EntryNode(),
        "pdf_to_md_node" : PdfToMdNode(),
        "md_image_node" : MdImageNode(),
        "document_split_node" : DocumentSplitNode(),
        "item_name_recognition_node" : ItemNameRecognitionNode(),
        "bge_embedding_chunks_node": BgeEmbeddingChunksNode(),
        "milvus_import_node":MilvusImportNode()
    }
    #2.2 遍历节点列表并添加
    for node_name,node in node_list.items():
        work_flow.add_node(node_name,node)

    #2.3 添加边
    #指定入口节点
    work_flow.set_entry_point("entry_node")

    """
    entry_node -> 条件边：是pdf还是md? -> pdf_to_md_node(是pdf) -> md_image_node(是md) ->
    
    -> document_split_node  -> item_name_recognition_node    ->
    
    -> bge_embedding_chunks_node  -> milvus_import_node  -> end
    """

    work_flow.add_conditional_edges("entry_node",import_router)
    work_flow.add_edge("pdf_to_md_node", "md_image_node")
    work_flow.add_edge("md_image_node", "document_split_node")
    work_flow.add_edge("document_split_node", "item_name_recognition_node")
    work_flow.add_edge("item_name_recognition_node", "bge_embedding_chunks_node")
    work_flow.add_edge("bge_embedding_chunks_node", "milvus_import_node")
    work_flow.add_edge("milvus_import_node", END)

    #编译
    compiled_graph = work_flow.compile()

    return compiled_graph


if __name__ == "__main__":
    #1. 创建graph
    state = {
        "import_file_path" : r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\input_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf" ,
        "file_dir" : r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir",
        "is_pdf_read_enabled" : True,
    }
    compiled_graph = import_graph()

    final_state = {}
    for event in  compiled_graph.stream(state):
        for node_name,state in event.items():
            print(f"{node_name} ")
            final_state = state
            print("="*30)
    print(final_state)