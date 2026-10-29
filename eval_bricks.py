import os
import json
import torch
import re
import string
import csv
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from tqdm import tqdm

# ================= 0. 导入 BRICKS 三大独立组件 =================
try:
    from bricks_header import prepend_shared_header_tokens, get_header_length, trim_redundant_header
except ImportError:
    print("Warning: bricks_header module not found.")

try:
    from bricks_separator import append_separator_tokens
except ImportError:
    print("Warning: bricks_separator module not found.")

try:
    from bricks_rope import apply_rope_offset
except ImportError:
    print("Warning: bricks_rope module not found.")

# ================= 1. 核心控制开关 (全局字典，在批量循环中被动态修改) =================
BRICKS_CONFIG = {
    "use_shared_header": False,  
    "use_separator": False,      
    "use_rope_repair": False     
}

# ================= 2. 基础配置与模型加载 =================
MODEL_PATH = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct" 
DATASET_PATH = "/root/autodl-tmp/datasets/2wikimqa_test.jsonl"
CSV_OUTPUT_PATH = "/root/autodl-tmp/ablation_results.csv"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    use_cache=True
)
model.eval()

# ================= 3. 核心算子 (接入三大组件) =================
@torch.no_grad()
def prefill_chunk(text):
    # 1. 独立进行严格的文档级 Tokenize
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs.input_ids
        
    # 2. 调用shared_header的组件，在头部拼接 Token
    if BRICKS_CONFIG["use_shared_header"]:
        input_ids = prepend_shared_header_tokens(input_ids, tokenizer)
        
    # 3. 调用 Separator 组件，在尾部追加 Token
    if BRICKS_CONFIG["use_separator"]:
        input_ids = append_separator_tokens(input_ids, tokenizer)
        
    outputs = model(input_ids=input_ids)
    kv = outputs.past_key_values
    
    extracted_layers = []
    for item in kv:
        extracted_layers.append((item[0], item[1]))
        
    return tuple(extracted_layers)

def concat_kv_caches(cache_list):
    if not cache_list: return None
        
    num_layers = len(cache_list[0])
    merged_layers = []
    header_len = get_header_length(tokenizer) if BRICKS_CONFIG["use_shared_header"] else 0
    
    for layer_idx in range(num_layers):
        k_tensors = []
        v_tensors = []
        current_seq_len_accumulated = 0
        
        for doc_idx, cache in enumerate(cache_list):
            k_tensor = cache[layer_idx][0]
            v_tensor = cache[layer_idx][1]
            
            if BRICKS_CONFIG["use_shared_header"] and doc_idx > 0 and header_len > 0:
                k_tensor, v_tensor = trim_redundant_header(k_tensor, v_tensor, header_len)
                
            doc_seq_len = k_tensor.shape[-2]
            
            if BRICKS_CONFIG["use_rope_repair"] and doc_idx > 0:
                actual_offset = current_seq_len_accumulated
                if BRICKS_CONFIG["use_shared_header"] and header_len > 0:
                    actual_offset = current_seq_len_accumulated - header_len
                
                k_tensor = apply_rope_offset(
                    k_tensor=k_tensor, 
                    offset=actual_offset, 
                    rotary_emb=model.model.rotary_emb, 
                    device=model.device
                )
                
            k_tensors.append(k_tensor)
            v_tensors.append(v_tensor)
            current_seq_len_accumulated += doc_seq_len
            
        merged_layers.append((torch.cat(k_tensors, dim=-2), torch.cat(v_tensors, dim=-2)))
        
    return DynamicCache(ddp_cache_data=merged_layers)

# ================= 4. 测评逻辑 (F1, EM, Inclusion) =================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if not common: return 0
    prec = len(common) / len(prediction_tokens)
    rec = len(common) / len(ground_truth_tokens)
    return (2 * prec * rec) / (prec + rec)

def exact_match_score(pred, gt): return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0
def inclusion_score(pred, gt): return 1.0 if normalize_answer(gt) in normalize_answer(pred) else 0.0

# ================= 5. 执行评测 Pipeline =================
print(f"Loading dataset from {DATASET_PATH}...")
dataset = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))

MAX_SAMPLES = 500
dataset_slice = dataset[:MAX_SAMPLES]

# 自动生成 8 种布尔值组合
options = [False, True]
combinations = list(itertools.product(options, options, options))

results_list = []

print(f"\n🚀 Starting Ablation Study: 8 Configurations, {MAX_SAMPLES} samples per run.\n")

# 批量循环开始
for run_idx, combo in enumerate(combinations, 1):
    header_opt, sep_opt, rope_opt = combo
    
    # 动态覆盖全局配置
    BRICKS_CONFIG["use_shared_header"] = header_opt
    BRICKS_CONFIG["use_separator"] = sep_opt
    BRICKS_CONFIG["use_rope_repair"] = rope_opt
    
    print(f"=========================================")
    print(f"Run {run_idx}/8 | Header: {header_opt} | Sep: {sep_opt} | RoPE: {rope_opt}")
    
    total_f1, total_em, total_inc, num_samples = 0, 0, 0, 0
    
    for i, data in enumerate(tqdm(dataset_slice, desc=f"Testing Run {run_idx}")):
        question = data['question']
        answer = data['answer'] 
        context_list = json.loads(data['context'])
        
        native_docs = []
        for item in context_list:
            title = item[0]
            sents = item[1]
            doc_text = f"Title: {title}\n" + " ".join(sents) + "\n"
            native_docs.append(doc_text)
        
        kv_caches = []
        for doc in native_docs:
            kv = prefill_chunk(doc)
            kv_caches.append(kv)
            
        merged_kv = concat_kv_caches(kv_caches)
        total_context_len = next(iter(merged_kv))[0].shape[-2]
        
        prompt = f"Answer the question based on the given passages.\nQuestion: {question}\nAnswer:"
        query_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        query_len = query_inputs.input_ids.shape[1]
        
        attention_mask = torch.ones((1, total_context_len + query_len), device=model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=query_inputs.input_ids,
                attention_mask=attention_mask,
                past_key_values=merged_kv,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(output_ids[0][query_len:], skip_special_tokens=True)
        answer_list = [answer] if isinstance(answer, str) else answer
        
        if not answer_list or all(not gt for gt in answer_list):
            s_f1, s_em, s_inc = 0.0, 0.0, 0.0
        else:
            s_f1 = max([f1_score(prediction, gt) for gt in answer_list])
            s_em = max([exact_match_score(prediction, gt) for gt in answer_list])
            s_inc = max([inclusion_score(prediction, gt) for gt in answer_list])
            
        total_f1 += s_f1
        total_em += s_em
        total_inc += s_inc
        num_samples += 1

    avg_f1 = total_f1 / num_samples
    avg_em = total_em / num_samples
    avg_inc = total_inc / num_samples
    
    # 记录结果
    results_list.append({
        "Shared_Header": header_opt,
        "Separator": sep_opt,
        "RoPE_Repair": rope_opt,
        "F1_Score": round(avg_f1, 4),
        "Exact_Match": round(avg_em, 4),
        "Inclusion": round(avg_inc, 4)
    })
    
    print(f"Metrics -> F1: {avg_f1:.4f} | EM: {avg_em:.4f} | Inclusion: {avg_inc:.4f}")
    print(f"=========================================\n")

# ================= 6. 导出 CSV =================
print(f"Writing results to {CSV_OUTPUT_PATH}...")
with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ["Shared_Header", "Separator", "RoPE_Repair", "F1_Score", "Exact_Match", "Inclusion"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(results_list)

print("✅ Batch ablation study completed successfully!")