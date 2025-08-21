from datasets import load_dataset as hf_load_dataset

def data_load(dataset_name, local_path=None, data_files=None, cache_dir=None, file_format=None):
    """
    智能加载数据集：先尝试在线加载，如果失败则尝试从本地加载
    
    参数:
        dataset_name: 需下载数据集名称
        local_path (str, optional): 本地数据集路径
        data_files: 本地数据集文件（字典格式，如 {"train": "path/to/train.parquet"}）
        cache_dir (str, optional): 在线加载的缓存目录
        file_format (str, optional): 文件格式（如 "parquet", "csv", "json"）
        
    返回:
        datasets.DatasetDict: 加载的数据集
    """   
    
    try:
        # 尝试在线加载数据集，设置缓存路径
        dataset = hf_load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"在线加载成功(缓存路径: {cache_dir})")
        return dataset
    
    except Exception as e:
        print(f"在线加载失败: {e}")
        
        # 尝试从本地加载
        print(f"尝试从本地路径加载: {local_path}")       
        
        try:  
            # 根据文件格式选择正确的加载方式
            if file_format == "parquet":
                dataset = hf_load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
            else:
                dataset = hf_load_dataset(file_format, data_dir=local_path, cache_dir=cache_dir)
            return dataset
        
        except Exception as e:
            print(f"本地加载失败: {e}")
            return None




