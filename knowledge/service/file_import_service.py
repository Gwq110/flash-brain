import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Tuple
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.main_graph import import_graph
from knowledge.processor.import_processor.state import create_default_state
from knowledge.utils.clients.storage_clients import StorageClients, logger
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, add_running_task, add_done_task, add_node_duration


class FileProcessService:
    def process_upload_file(self,file:UploadFile) -> Tuple[str, str, str]:
        #生成task_id
        task_id = uuid.uuid4().hex[:8]
        # 1. 任务开始，应该设置整个任务的状态为RUNNING
        start_time = time.time()
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        add_running_task(task_id,"upload_file")
        #2. 将上传的文件保存到本地临时文件中
        try:
            import_file_path,file_dir = self._save_file_to_local(file)
        except Exception as e:
            logger.error(e)
            update_task_status(task_id,TASK_STATUS_FAILED)
        #3. 将文件备份到minio
        self._save_file_to_remote(import_file_path,file.filename)

        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file",time.time()-start_time)
        return import_file_path,file_dir,task_id


    def run_main_graph(self,import_file_path: str, file_dir: str, task_id: str):
        state = create_default_state(
            import_file_path=import_file_path,
            file_dir=file_dir,
            task_id=task_id
        )
        compiled_graph = import_graph()
        try:
            for event in compiled_graph.stream(state):
                for node_name, state in event.items():
                    print(f"{node_name} 开始执行......")
                    print("=" * 30)
            #整个任务执行完成
            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.exception(e)
            #代表任务失败
            update_task_status(task_id,TASK_STATUS_FAILED)


    # 获取本地临时目录
    @staticmethod
    def _get_base_dir():
        local_base_dir = get_local_base_dir()
        return os.path.join(local_base_dir,datetime.now().strftime("%Y%m%d"))

    #将上传的文件保存到本地临时文件中
    def _save_file_to_local(self, file: UploadFile) -> Tuple[str, str]:
        #获取本地临时目录
        base_dir = self._get_base_dir()
        #判断base_dir是否存在，不存在则创建
        os.makedirs(base_dir,exist_ok=True)
        #构建import_file_path
        import_file_path = os.path.join(base_dir,file.filename)
        #往import_file_path写入file
        try:
            with open(import_file_path,"wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.error(f"保存文件到目录：{import_file_path} 失败，详情：{e}")

        return import_file_path,base_dir


    def _save_file_to_remote(self, import_file_path: str, filename: str):
        #创建MinIo客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e :
            logger.warning(f"获取MinIo客户端失败，不影响主流程，详情{e}")
            return
        #2. 保存文件
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.warning(f"MinIo客户端备份源文件失败，不影响主流程，详情{e}")
