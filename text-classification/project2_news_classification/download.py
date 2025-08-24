import os
from huggingface_hub import snapshot_download


# 设置国内镜像（关键：绕过境外网络）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 模型仓库与保存路径
repo_id = "google-bert/bert-base-chinese"
cache_dir = "/root/myproject/pretrained_bert" 

# 下载指定文件（确保为真实文件，非软链接）
snapshot_download(
    repo_id=repo_id,
    local_dir=cache_dir,
    allow_patterns=["config.json", "pytorch_model.bin", "vocab.txt"],  # 只下载需要的三个文件
    local_files_only=False,  # 强制从镜像重新下载
    local_dir_use_symlinks=False,  # 禁用软链接，保存真实文件
    use_auth_token=False  
)

# 验证文件是否为真实文件（非软链接）
print("\n下载文件验证：")
for file in ["config.json", "pytorch_model.bin", "vocab.txt"]:
    file_path = os.path.join(cache_dir, file)
    if os.path.islink(file_path):
        print(f"{file} 是软链接，请检查下载")
    else:
        print(f"{file} 是真实文件")
    

# unset HTTP_PROXY HTTPS_PROXY
# export HF_ENDPOINT="https://hf-mirror.com"  # 强制指定国内镜像
# echo $HF_ENDPOINT  # 确认输出：https://hf-mirror.com
# python download.py

# (以上下载会有冗余文件夹生成, 可直接删除）)


#命令行直接下载
# wget -P ./pretrained_bert https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/config.json
# wget -P ./pretrained_bert https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin
# wget -P ./pretrained_bert https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/vocab.txt