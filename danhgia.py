import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def compute_cosine_similarity(reference, candidate, model):
    """Tính Cosine Similarity giữa 2 câu sử dụng vector nhúng từ model."""
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    cand_embedding = model.encode(candidate, convert_to_tensor=True)
    similarity = util.cos_sim(ref_embedding, cand_embedding)
    return similarity.item()

def evaluate_cosine(reference_file, generated_file):
    """Đánh giá Cosine Similarity cho toàn bộ bộ dữ liệu."""
    reference_data = load_json(reference_file)
    generated_data = load_json(generated_file)
    
    # Sử dụng mô hình nhúng văn bản, ví dụ: 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    total_sim = 0
    count = 0
    similarities = []
    
    for ref_item, gen_item in zip(reference_data, generated_data):
        # Lấy danh sách câu trả lời chuẩn từ file output (reference)
        ref_answers = [qa.get("answer", qa.get("correct_answer", ""))
                       for qa in ref_item.get("qa_pairs", [])]
        # Lấy danh sách câu trả lời sinh ra từ generated_answers (candidate)
        gen_answers = [qa.get("answer", qa.get("correct_answer", ""))
                       for qa in gen_item.get("qa_pairs", [])]
        for ref_ans, gen_ans in zip(ref_answers, gen_answers):
            sim = compute_cosine_similarity(ref_ans, gen_ans, model)
            similarities.append(sim)
            total_sim += sim
            count += 1
    
    avg_similarity = total_sim / count if count > 0 else 0
    return similarities, avg_similarity

if __name__ == "__main__":
    # reference_file = "output.json"                # File chứa câu trả lời chuẩn
    reference_file = "generated_answers.json"                # File chứa câu trả lời chuẩn
    # generated_file = "generated_answers.json"       # File chứa câu trả lời sinh ra từ mô hình
    generated_file = "output.json"       # File chứa câu trả lời sinh ra từ mô hình
    sims, avg_sim = evaluate_cosine(reference_file, generated_file)
    
    print(f"🔹 Cosine Similarity trung bình: {avg_sim:.4f}")
    print(f"📊 Cosine Similarity của từng cặp câu trả lời: {sims}")
