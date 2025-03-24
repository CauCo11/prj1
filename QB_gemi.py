import json
import google.generativeai as genai
import time

# Cấu hình API Key
genai.configure(api_key="AIzaSyCKUM9cSVWi7va5F2L7qeqEcvg4VRrTjic")

def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, "r", encoding="utf-8") as file: return json.load(file)

def generate_answers(text, question, temperature=0.7):
    """Sinh câu trả lời từ nội dung văn bản và câu hỏi."""
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    prompt = f"Dựa vào nội dung sau, hãy trả lời câu hỏi: {question}\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "Không có câu trả lời"

def process_questions_and_answers(input_file, output_file):
    """Sinh câu trả lời từ câu hỏi và lưu vào file JSON từng câu."""
    data = load_json(input_file)
    results = []
    
    for item in data:
        content = item.get("content", "")
        questions = item.get("qa_pairs", [])

        if not content.strip() or not questions:
            continue

        qa_pairs = []
        
        for q in questions:
            if "options" in q:
                qa_pairs.append({
                    "question": q["question"],
                    "options": q["options"],
                    "correct_answer": q.get("correct_answer", "Không có đáp án hợp lệ")
                })
            else:
                answer = generate_answers(content, q["question"])
                qa_pairs.append({"question": q["question"], "answer": answer})
                time.sleep(5)  # Chờ 5 giây giữa các request tránh quá tải API
                
                # Lưu ngay vào file
                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=4, ensure_ascii=False)
        
        results.append({"content": content, "qa_pairs": qa_pairs})
        print(f"✔️ Xử lý xong đoạn văn bản: {content[:30]}...")

if __name__ == "__main__":
    input_file = "output.json"
    output_file = "generated_answers.json"
    process_questions_and_answers(input_file, output_file)
    print(f"✅ Kết quả đã lưu vào {output_file}")
