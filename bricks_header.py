import torch

# 共享的系统头部 (作为全局 Attention Sink)
SHARED_HEADER = (
    "System: You are an expert AI assistant. "
    "Please read the following documents carefully to answer the question.\n"
    "---------------------\n"
)

def inject_shared_header(text):
    """在预计算前，将头部强行推入文档最前方"""
    return SHARED_HEADER + text

def get_header_length(tokenizer):
    """计算头部在分词后的 Token 长度 (包含 Llama-3 默认的 <BOS>)"""
    return len(tokenizer.encode(SHARED_HEADER, add_special_tokens=False))

def trim_redundant_header(k_tensor, v_tensor, header_len):
    """
    在拼接时，裁掉非首篇文档的头部张量
    张量形状: [batch, num_heads, seq_len, head_dim]
    在 seq_len 维度 (dim=-2) 切片
    """
    k_trimmed = k_tensor[..., header_len:, :]
    v_trimmed = v_tensor[..., header_len:, :]
    return k_trimmed, v_trimmed