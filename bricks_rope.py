import torch

def rotate_half(x):
    """旋转一半的维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_offset(k_tensor, offset, rotary_emb, device):
    """
    使用模型原生的 rotary_emb 层，获取完美适配 Llama-3.1 缩放机制的偏移角度，并叠加。
    """
    # 1. 构造一个形状为 [1, 1] 的位置张量
    position_ids = torch.tensor([[offset]], device=device, dtype=torch.long)
    
    # 2. 构造 dummy 变量骗出原生的 cos 和 sin
    dummy = torch.empty((1, 1, 1, k_tensor.shape[-1]), device=device, dtype=k_tensor.dtype)
    cos_delta, sin_delta = rotary_emb(dummy, position_ids)
    
    # 3. 兼容 HuggingFace 的维度规范
    # 返回的可能是 [batch, seq_len, head_dim]，我们将其扩展为可广播的形状
    if cos_delta.dim() == 3:
        cos_delta = cos_delta.unsqueeze(1)
        sin_delta = sin_delta.unsqueeze(1)
        
    cos_delta = cos_delta.to(k_tensor.dtype)
    sin_delta = sin_delta.to(k_tensor.dtype)
    
    # 4. 完美二次旋转叠加 (三角函数加法定理)
    k_shifted = (k_tensor * cos_delta) + (rotate_half(k_tensor) * sin_delta)
    
    return k_shifted