import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Thử import underthesea
try:
    from underthesea import word_tokenize # Sử dụng thư viện tokenize của Underthesea
    USE_UNDERTHESEA = True
except ImportError:
    USE_UNDERTHESEA = False
    print("Thư viện underthesea không được tìm thấy. Sử dụng phương pháp tokenize đơn giản.")

class DepartmentClassifier:
    def __init__(self, keywords_file):
        # Khởi tạo PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_auth_token=False)
        self.model = AutoModel.from_pretrained("vinai/phobert-base", use_auth_token=False)
        
        # Đọc keywords từ file Excel
        self.departments = self._load_keywords(keywords_file)
        
        # Tạo embedding cho các từ khóa phòng ban
        self.department_embeddings = {}
        for dept, keywords in self.departments.items():
            dept_emb = self._get_embeddings(" ".join(keywords))
            self.department_embeddings[dept] = dept_emb.mean(axis=0)

    def _load_keywords(self, excel_file):
        try:
            # Đọc file Excel
            df = pd.read_excel(excel_file)
            # Lấy tên các phòng ban từ hàng đầu tiên
            departments = {}
            # Duyệt qua từng cột
            for column in df.columns:
                # Lọc các giá trị không null trong cột
                keywords = df[column].dropna().tolist()
                 # Loại bỏ tên phòng ban khỏi danh sách keywords
                keywords = keywords[1:] # Bỏ qua hàng đầu tiên (tên phòng ban)
                # Thêm vào dictionary
                departments[column.lower()] = keywords
            
            return departments
            
        except Exception as e:
            print(f"Lỗi khi đọc file keywords: {str(e)}")
            return {}

    def _tokenize_text(self, text):
        # Tokenize văn bản tiếng Việt
        if USE_UNDERTHESEA:
            words = word_tokenize(text)
            return " ".join(words)
        else:
             # Phương pháp tokenize đơn giản nếu không có underthesea
            words = text.lower().split()
            return " ".join(words)

    def _get_embeddings(self, text):
          # Lấy embedding từ PhoBERT
        tokenized_text = self._tokenize_text(text) # Chuyển văn bản
        # Mã hóa văn bản
        encoded = self.tokenizer(tokenized_text, return_tensors='pt', padding=True, truncation=True)
         # Tính toán embedding không cần gradient
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state.numpy()
        
        return embeddings[0]
# threshold
    def classify_text(self, text, threshold=0.3): # Ngưỡng phân loại ( lọc ra phòng ban có số điểm từ 0.3 trở lên )
        # Phân loại văn bản vào các phòng ban
        text_emb = self._get_embeddings(text)
        text_emb_mean = text_emb.mean(axis=0)
          #Tính độ tương đồng cosine
        results = {}
        for dept, dept_emb in self.department_embeddings.items():
            similarity = cosine_similarity([text_emb_mean], [dept_emb])[0][0]
            if similarity >= threshold:
                results[dept] = float(similarity)
        # Sắp xếp kết quả theo độ tương đồng
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True)) # results nơi chứa kết quả tính độ tương đồng
    
# Xét điểm và các trạng thái thuộc về phòng ban

    def get_similarity_categories(self, classifications):
        very_high = []
        high = []
        medium = []

        for dept, score in classifications.items():
            similarity_percentage = score * 100
            if similarity_percentage > 70:
                very_high.append(dept.upper())
            elif similarity_percentage > 60:
                high.append(dept.upper())
            elif similarity_percentage > 45:
                medium.append(dept.upper())

        return very_high, high, medium

def process_texts_file(texts_file, classifier):
    try:
        texts_df = pd.read_excel(texts_file)
        
        if 'content' not in texts_df.columns:
            raise ValueError("Không tìm thấy cột 'content' trong file texts")

        results = []
          # Duyệt qua từng dòng văn bản
        for idx, row in texts_df.iterrows():
            text = row['content']
              # Phân loại văn bản
            classifications = classifier.classify_text(text)
            # Xác định nhãn phòng ban
            very_high, high, medium = classifier.get_similarity_categories(classifications)
            
            if very_high:
                department_label = f"Rất cao: {', '.join(very_high)}"
            elif high:
                department_label = f"Cao: {', '.join(high)}"
            elif medium:
                department_label = f"Trung bình: {', '.join(medium)}"
            else:
                department_label = "Đoạn văn không thuộc phòng ban nào"
            
            result_row = {
                'content': text
            }
            # Thêm cột cảm xúc nếu tồn tại
            if 'emotion' in texts_df.columns:
                result_row['emotion'] = row['emotion'] 
            # Tỷ lệ phần trăm
            for dept, score in classifications.items():
                result_row[f'{dept}_score'] = round(score * 100, 2)
            
            result_row['departments'] = department_label
            results.append(result_row)
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame(results)
        
        # Sắp xếp lại thứ tự cột
        columns_order = ['content', 'emotion'] if 'emotion' in texts_df.columns else ['content']
        columns_order.extend([col for col in results_df.columns if col.endswith('_score')])
        columns_order.append('departments')
        results_df = results_df[columns_order]
         # Lưu kết quả ra file Excel
        output_file = 'result1.xlsx'
        results_df.to_excel(output_file, index=False)
        print(f"Đã lưu kết quả phân loại vào file: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"Lỗi khi xử lý file texts: {str(e)}")
        return None

def main():
    try:
        keywords_file = 'knowledgebase.xlsx'
        texts_file = 'Hưngdataset.xlsx'
         # Tải model PhoBERT
        print("Đang tải model PhoBERT...")
        classifier = DepartmentClassifier(keywords_file)
        print("Đã tải xong model!")
         # Xử lý và phân loại văn bản
        print("\n=== PHÂN LOẠI VĂN BẢN THEO PHÒNG BAN ===\n")
        results_df = process_texts_file(texts_file, classifier)
         # In mẫu kết quả
        if results_df is not None:
            print("\nMẫu kết quả phân loại:")
            print(results_df.head())
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
        print("\nHãy đảm bảo đã cài đặt các thư viện cần thiết:")
        print("pip install torch transformers underthesea scikit-learn pandas openpyxl")

if __name__ == "__main__":
    main()