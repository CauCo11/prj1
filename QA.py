import json
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Thay bằng API key hợp lệ
os.environ["OPENAI_API_KEY"] = "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"  

def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_questions_and_answers(text):
    """Tạo câu hỏi và câu trả lời từ nội dung văn bản."""
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    prompt_template = """
        🎯 **Nhiệm vụ của bạn**:
        Hãy tạo một danh sách câu hỏi đa dạng từ đoạn văn bản sau:
        - Câu hỏi Ai (Who), Cái gì (What), Ở đâu (Where), Như thế nào (How).
        - Câu hỏi Có/Không (Yes/No).
        - Câu hỏi trắc nghiệm (MCQ - Multiple Choice Questions) với 4 đáp án, có một đáp án đúng.

        📌 **Nội dung**:
        -------------------
        {text}
        -------------------

        📝 **Định dạng JSON:**
        {{
            "qa_pairs": [
                {{"question": "Ai ...?", "answer": "..."}},
                {{"question": "Cái gì ...?", "answer": "..."}},
                {{"question": "Ở đâu ...?", "answer": "..."}},
                {{"question": "Như thế nào ...?", "answer": "..."}},
                {{"question": "Câu hỏi Có/Không?", "answer": "Có/Không"}},
                {{
                    "question": "Câu hỏi trắc nghiệm?",
                    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
                    "correct_answer": "B"
                }}
            ]
        }}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    response = llm.predict(PROMPT.format(text=text))

    # Kiểm tra xem có đúng JSON không
    try:
        qa_pairs = json.loads(response)["qa_pairs"]
    except json.JSONDecodeError:
        qa_pairs = [{"question": "Lỗi", "answer": "Không thể sinh câu hỏi"}]

    return qa_pairs

def save_to_json(json_path, output_file):
    """Tạo danh sách câu hỏi & câu trả lời và lưu ngay vào JSON sau mỗi lần request."""
    data = load_json(json_path)

    # Mở file để ghi từng phần tử ngay sau khi lấy được kết quả
    with open(output_file, "a", encoding="utf-8") as jsonfile:
        jsonfile.write("[\n")  # Bắt đầu danh sách JSON

        first_item = True  # Để kiểm tra phần tử đầu tiên (tránh dấu phẩy thừa)
        for item in data:
            content = item["content"]
            qa_pairs = generate_questions_and_answers(content)

            result = {
                "content": content,
                "qa_pairs": qa_pairs
            }

            # Ghi kết quả vào file ngay lập tức
            if not first_item:
                jsonfile.write(",\n")  # Thêm dấu phẩy trước phần tử tiếp theo
            json.dump(result, jsonfile, indent=4, ensure_ascii=False)
            first_item = False

            print(f"✔️ Đã xử lý xong đoạn văn bản: {content[:30]}...")

        jsonfile.write("\n]")  # Kết thúc danh sách JSON

if __name__ == "__main__":
    json_path = r"D:\thaythedoanpj1\data\data.json"
    output_json = "output.json"
    save_to_json(json_path, output_json)
    print(f"✅ Kết quả đã được lưu vào {output_json}")
