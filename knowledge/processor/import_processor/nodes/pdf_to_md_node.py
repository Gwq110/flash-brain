import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any
import requests
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import ConfigurationError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    node_name = "pdf_to_md_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1.上传pdf并轮询pdf解析结果
        pdf_path = state.get("pdf_path")
        pdf_path_obj = Path(pdf_path)
        zip_url = self._upload_pdf_and_query_result(self.config,pdf_path_obj)
        #2.下载zip，提取md文件
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        md_path = self._extract_md(zip_url,file_dir_obj,pdf_path_obj)

        #3.将md文件地址存入到state，方便之后分片
        state["md_path"] = md_path
        return state

    def _upload_pdf_and_query_result(self, config: ImportConfig, pdf_path_obj: Path) -> Any | None:
        #1.检查minerU的配置
        mineru_api_token = config.mineru_api_token
        mineru_base_url = config.mineru_base_url
        if not mineru_api_token or not mineru_base_url:
            self.logger.error(f"mineru_api_token or mineru_base_url must be set")
            raise ConfigurationError(f"mineru_api_token or mineru_base_url must be set")

        #2.获取文件上传链接
        #2.1 构建请求地址
        url = mineru_base_url + "/file-urls/batch"
        # 2.2 构建请求头
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mineru_api_token}"
        }
        #2.3 构建请求体
        data = {
            "files": [
                {"name": f"{pdf_path_obj.name}", "data_id": "abcd"}
            ],
            "model_version": "vlm"
        }
        # 2.4 发送请求并判断请求是否成功
        try:
            response = requests.post(url, headers=header, json=data)
        except Exception as e:
            self.logger.error(f"【获取上传链接】接口调用失败，异常信息：{str(e)}")
            raise RuntimeError(f"【获取上传链接】接口调用失败，异常信息：{str(e)}")

        if response.status_code != 200:
            self.logger.error(f"【获取上传链接】接口调用失败，状态码: {response.status_code}，详情: {response.text}")
            raise RuntimeError(f"【获取上传链接】接口调用失败，状态码: {response.status_code}，详情: {response.text}")
        # 2.5 判断业务状态码
        result = response.json()
        if result["code"] != 0:
            #获取上传链接失败
            self.logger.error(f"【获取上传链接】获取链接失败，状态码: {response.status_code}，详情: {response.text}")
            raise RuntimeError(f"【获取上传链接】获取链接失败，状态码: {response.status_code}，详情: {response.text}")
        #2.6 获取上传链接和batch_id
        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0] #单个上传所以只拿一个

        #3.上传pdf文件到MinerU
        try:
            with open(str(pdf_path_obj), "rb") as f:
                file_content = f.read()
                upload_res = requests.put(upload_url, data=file_content)
        except Exception as e:
            self.logger.error(f"调用pdf上传接口失败,失败信息是{str(e)}")
            raise RuntimeError(f"调用pdf上传接口失败,失败信息是{str(e)}")

        if upload_res.status_code != 200:
            self.logger.error(f"【上传pdf】上传pdf失败，状态码: {upload_res.status_code}，详情: {upload_res.text}")
            raise RuntimeError(f"【上传pdf】上传pdf失败，状态码: {upload_res.status_code}，详情: {upload_res.text}")

        #4.轮询结果，直到成功，失败或者超时
        #4.1 设置最大等待时间
        max_wait_time = 600
        #4.2 设置轮询间隔时长
        pull_interval = 3
        # 开始时间
        start_time = time.time()
        while True:
            #等待一会，避免请求太频繁
            time.sleep(pull_interval)
            # 判断轮询是否超时
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > max_wait_time:
                # 轮询超时
                self.logger.error(f"【轮询结果】轮询超时，轮询时间：{elapsed_time:.2f}s，batch_id: {batch_id}")
                raise TimeoutError(f"【轮询结果】轮询超时,轮询时间：{elapsed_time:.2f}s，batch_id: {batch_id}")

            pull_url = mineru_base_url + f"/extract-results/batch/{batch_id}"
            res = requests.get(pull_url, headers=header)
            #判断是否成功连接
            if res.status_code != 200:
                self.logger.error(f"【轮询请求结果】连接失败，状态码：{res.status_code}，响应内容：{res.text}")
                # raise RuntimeError(f"【轮询请求结果】连接失败，状态码：{res.status_code}，响应内容：{res.text}")
                continue

            pull_data = res.json()
            if pull_data["code"] != 0:
                self.logger.error(f"【轮询结果】获取文件转换状态失败，业务码: {pull_data['code']},响应内容: {pull_data}")
                #raise RuntimeError(f"【轮询结果】获取文件转换状态失败，业务码: {pull_data['code']},响应内容: {pull_data}")
                continue

            extract_item = pull_data["data"]["extract_result"][0]
            if extract_item["state"] == "done":
                # 已经完成了
                self.logger.info(f"【轮询结果】轮询已完成,耗时为{elapsed_time:.2f}s")
                full_zip_url = extract_item["full_zip_url"]
                return full_zip_url
            elif extract_item["state"] == "failed":
                self.logger.error(f"【轮询结果】轮询失败，解析任务失败，batch_id: {batch_id}")
                raise RuntimeError(f"【轮询结果】轮询失败，解析任务失败，batch_id: {batch_id}")
            else:
                continue

    #从解析结果中提取md
    def _extract_md(self, zip_url, file_dir_obj: Path, file_pdf_obj: Path) -> str:
        #1. 发送get请求下载zip包
        try:
            res = requests.get(zip_url, timeout=20)
        except Exception as e:
            self.logger.error(f"【下载ZIP】下载失败，异常信息：{str(e)}")
            raise RuntimeError(f"【下载ZIP】下载失败，异常信息：{str(e)}")

        if res.status_code != 200:
            self.logger.error(f"【下载ZIP】下载失败，状态码：{res.status_code}，响应内容：{res.text}")
            raise RuntimeError(f"【下载ZIP】下载失败，状态码：{res.status_code}，响应内容：{res.text}")

        self.logger.info("zip下载成功")

        #2. 指定zip包的存储路径，将下载下来的zip包写到该路径下
        #2.1 构建zip包的存储路径：filedir/文件名_result.zip
        zip_path = file_dir_obj/f"{file_pdf_obj.stem}_result.zip"
        #2.2 将response中的内容写入到zip包存储路径下
        try:
            with open(str(zip_path), "wb") as f:
                f.write(res.content)
        except Exception as e:
            self.logger.error(f"【保存ZIP】保存失败，异常信息：{str(e)}")
            raise RuntimeError(f"【保存ZIP】保存失败，异常信息：{str(e)}")

        self.logger.info("zip保存成功")
        #3. 解压zip包
        #3.1 构建解压路径
        extract_path = file_dir_obj/f"{file_pdf_obj.stem}"
        #3.2 如果该目录下已经有内容，要先清空文件夹
        try:
            if extract_path.exists():
                shutil.rmtree(str(extract_path))
                self.logger.info(f"【解压ZIP】目标目录已存在，已清空目录：{extract_path}")
        except Exception as e:
            self.logger.warning(f"【解压ZIP】目标目录已存在，但清空失败, 不影响后续解压")
        #3.3 将zip的内容解压到解压路径下
        try:
            with zipfile.ZipFile(str(zip_path)) as zip_ref:
                zip_ref.extractall(str(extract_path))
        except Exception as e:
            self.logger.info(f"【解压ZIP】解压到{extract_path}失败")
            raise RuntimeError(F"解压zip包失败，失败详情{str(e)}")
        self.logger.info("zip解压成功")
        #3.4 将md文件进行重命名
        md_file_path = extract_path/ "full.md"
        new_md_file_path = md_file_path.with_name(f"{file_pdf_obj.stem}.md")
        md_file_path.rename(new_md_file_path)

        self.logger.info("zip改名成功")
        #4. 返回md文件的路径
        return str(md_file_path)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = PdfToMdNode()
    state ={
        "pdf_path" : r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\input_dir\万用表RS-12的使用.pdf",
        "file_dir" : r"D:\pythonCode\PythonProject\shopkeeper-brain\knowledge\processor\import_processor\output_dir"

    }
    node(state)
