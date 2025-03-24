import json
from rouge_score import rouge_scorer

def load_json(file_path):
    """Hàm đọc dữ liệu từ tệp JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def calculate_avg_rouge_precision(reference_file, generated_file):
    """Hàm tính toán trung bình độ chính xác (Precision) của ROUGE-1, ROUGE-2, ROUGE-L."""
    reference_data = load_json(reference_file)
    generated_data = load_json(generated_file)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    count = 0

    for ref, gen in zip(reference_data, generated_data):
        for ref_pair, gen_pair in zip(ref.get("qa_pairs", []), gen.get("qa_pairs", [])):
            ref_answer = ref_pair.get("answer", "")
            gen_answer = gen_pair.get("answer", "")
            score = scorer.score(ref_answer, gen_answer)

            total_rouge1 += score["rouge1"].precision
            total_rouge2 += score["rouge2"].precision
            total_rougeL += score["rougeL"].precision
            count += 1

    avg_rouge1 = total_rouge1 / count if count > 0 else 0
    avg_rouge2 = total_rouge2 / count if count > 0 else 0
    avg_rougeL = total_rougeL / count if count > 0 else 0

    return avg_rouge1, avg_rouge2, avg_rougeL

if __name__ == "__main__":
    reference_file = "generated_answers.json"
    generated_file = "output.json"

    avg_rouge1, avg_rouge2, avg_rougeL = calculate_avg_rouge_precision(reference_file, generated_file)

    print(f"📊 Trung bình độ chính xác (Precision) ROUGE Scores:")
    print(f"✅ ROUGE-1 Precision: {avg_rouge1:.4f}")
    print(f"✅ ROUGE-2 Precision: {avg_rouge2:.4f}")
    print(f"✅ ROUGE-L Precision: {avg_rougeL:.4f}")
