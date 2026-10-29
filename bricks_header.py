import torch

# 共享的系统头部 (作为全局 Attention Sink)
SHARED_HEADER = (
    "System: You are an expert AI assistant. "
    "Please read the following documents carefully to answer the question.\n"
    "---------------------\n"
)

# 缓存机制，避免在批量推理时重复执行 Tokenize
_CACHED_HEADER_IDS = None

def get_header_input_ids(tokenizer, device):
    """获取序列化后的头部 Token 张量（带缓存）"""
    global _CACHED_HEADER_IDS
    if _CACHED_HEADER_IDS is None:
        _CACHED_HEADER_IDS = tokenizer(
            SHARED_HEADER, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids
    return _CACHED_HEADER_IDS.to(device)

def prepend_shared_header_tokens(input_ids, tokenizer):
    """
    在 Token 层面将共享头部拼接到文档张量的最前方。
    避免了纯文本拼接导致的 BPE 分词器边界污染。
    """
    header_ids = get_header_input_ids(tokenizer, input_ids.device)
    # 沿 seq_len 维度 (dim=-1) 拼接到原文档前面
    new_input_ids = torch.cat([header_ids, input_ids], dim=-1)
    return new_input_ids

def get_header_length(tokenizer):
    """动态获取头部在分词后的 Token 物理长度"""
    header_ids = get_header_input_ids(tokenizer, device="cpu")
    return header_ids.shape[1]

def trim_redundant_header(k_tensor, v_tensor, header_len):
    """
    在 KV Cache 在线拼接时，裁掉非首篇文档前方的共享头部冗余张量。
    输入张量形状标准为: [batch, num_heads, seq_len, head_dim]
    沿着 seq_len 维度 (dim=-2) 进行物理切片。
    """
    k_trimmed = k_tensor[..., header_len:, :]
    v_trimmed = v_tensor[..., header_len:, :]
    return k_trimmed, v_trimmed