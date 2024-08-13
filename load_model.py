# from modelscope import snapshot_download
# model_dir = snapshot_download('qwen/Qwen-7B-Chat',cache_dir='model')

# from modelscope import snapshot_download
# model_dir = snapshot_download('qwen/Qwen2-7B', cache_dir='Qwen2_7B_model')

#验证SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('3ebe706c-d1ea-40d8-a25d-41d8d224137d')

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('OpenGVLab/InternVL2-8B', cache_dir='InternVL2-8B')