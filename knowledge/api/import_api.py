import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from knowledge.core.dependences import get_file_process_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.file_import_service import FileProcessService
from knowledge.utils.task_util import get_task_info


def register_router(app: FastAPI):

    @app.get("/hello")
    def hello():
        return "Hello World!"

    @app.post("/upload" ,response_model= UploadResponse)
    def uoload_file(file: UploadFile,background_tasks: BackgroundTasks ,file_process_srvice: FileProcessService = Depends(get_file_process_service)):
        #将上传的文件保存到本地临时文件中,并备份到minio中
        import_file_path,file_dir,task_id = file_process_srvice.process_upload_file(file)
        #执行导入的主流程
        background_tasks.add_task(file_process_srvice.run_main_graph,import_file_path,file_dir,task_id)

        return UploadResponse(message="上传成功",task_id=task_id)

    @app.get("/status/{task_id}" , response_model= TaskStatusResponse)
    def get_task_status(task_id: str):
        #获取当前任务的信息
        task_info = get_task_info(task_id)
        #一个*代表展开列表。两个*代表展开字典
        return TaskStatusResponse(**task_info)



def create_app():
    #创建FastAPI
    app = FastAPI(
        description="Knowledge_import_API",
        version="1.0"
    )

    #处理跨域问题: 这个项目不需要考虑
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #挂载静态资源
    front_dir = get_front_page_dir()
    #用户访问 URL 中 /front 开头的请求，都交给这个静态文件（front_dir）目录处理。
    app.mount("/front", StaticFiles(directory=front_dir))

    #注册路由
    register_router(app)

    return app

if __name__ == "__main__":
    #运行fastapi
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)


