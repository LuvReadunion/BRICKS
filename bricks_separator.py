# bricks_separator.py
import torch

def append_separator_tokens(input_ids, tokenizer):
    """
    在文档完成 Token 化之后，在张量层面追加隔断 Token。
    彻底避免 BPE 分词器由于字符串相连产生的跨词合并现象。
    """
    # 独立序列化隔断符（必须 add_special_tokens=False，防止混入额外的 BOS）
    sep_ids = tokenizer.encode("\n\n\n", add_special_tokens=False, return_tensors="pt")
    print(sep_ids)
    
    # 沿 seq_len 维度拼接到原文档尾部
    new_input_ids = torch.cat([input_ids, sep_ids.to(input_ids.device)], dim=-1)
    
    return new_input_ids
