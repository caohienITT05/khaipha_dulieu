# KHAI PHÁ DỮ LIỆU – BÀI TẬP LỚN

## Nghiên cứu thuật toán Decision Tree và Random Forest trong bài toán dự đoán bệnh tim

## 1. Giới thiệu dự án

### 1.1. Thông tin đề tài

- **Tên đề tài:** Nghiên cứu nhóm thuật toán dựa trên cây (Cây quyết định và Rừng ngẫu nhiên) trong Khai phá dữ liệu và ứng dụng vào bài toán Dự đoán bệnh tim.
- **Môn học:** Khai phá dữ liệu (Data Mining).
- **Ngôn ngữ lập trình:** Python.
- **Môi trường thực nghiệm:** Google Colab.
- **Thư viện chính:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Bộ dữ liệu:** UCI Cleveland Heart Disease.
- **Tệp dữ liệu dự kiến:** `heart_disease_uci.csv`.

### 1.2. Mục đích dự án

**Mục đích học thuật**

- Tìm hiểu bản chất toán học và cơ chế hoạt động của Decision Tree và Random Forest.
- Hiểu cách lựa chọn thuộc tính phân nhánh bằng Entropy, Information Gain và Gini Impurity.
- Phân tích hiện tượng Overfitting và phương pháp hạn chế quá khớp.
- Nắm vững các chỉ số đánh giá mô hình để giải thích kết quả thực nghiệm và trả lời câu hỏi vấn đáp.

**Mục đích thực tiễn**

- Ứng dụng học máy để xây dựng mô hình dự đoán nguy cơ mắc bệnh tim từ dữ liệu lâm sàng.
- Xác định các đặc trưng có ảnh hưởng lớn đến kết quả dự đoán.
- Minh họa khả năng khai phá tri thức từ dữ liệu y tế.

> **Lưu ý:** Mô hình trong dự án phục vụ mục đích học tập và nghiên cứu, không thay thế chẩn đoán hoặc quyết định điều trị của bác sĩ.

## 2. Mục tiêu cụ thể

### Mục tiêu 1: Tổng quan và công trình liên quan

- Trình bày khái niệm Machine Learning trong Khai phá dữ liệu.
- Giới thiệu nhóm thuật toán Tree-based.
- Tổng hợp các ứng dụng thực tế của thuật toán dựa trên cây.
- Tìm hiểu các công trình liên quan đến ID3, C4.5, CART, Bagging và Random Forest.

### Mục tiêu 2: Cơ sở lý thuyết thuật toán

- Trình bày công thức Entropy, Information Gain và Gini Impurity.
- Giải thích cách lựa chọn thuộc tính phân nhánh.
- Làm rõ cơ chế Bootstrap Sampling, Bagging và Random Subspace.
- Trình bày Majority Voting và Out-of-Bag (OOB).
- Mô tả các bước thực hiện Decision Tree và Random Forest bằng sơ đồ khối.

### Mục tiêu 3: Thực nghiệm

- Thu thập và khám phá bộ dữ liệu bệnh tim.
- Kiểm tra dữ liệu khuyết và tiền xử lý dữ liệu.
- Thực hiện phân tích khám phá dữ liệu (EDA).
- Xây dựng và tối ưu mô hình Decision Tree.
- Xây dựng mô hình Random Forest.
- Đánh giá và so sánh kết quả của các mô hình.

### Mục tiêu 4: Phân tích kết quả

- So sánh Decision Tree chưa cắt tỉa, Decision Tree có cắt tỉa và Random Forest.
- Đánh giá bằng Accuracy, Precision, Recall, F1-Score, ROC-AUC và Confusion Matrix.
- Phân tích Feature Importance.
- Tổng hợp điểm mạnh, điểm yếu của từng thuật toán.
- Đưa ra kết luận và tài liệu tham khảo theo chuẩn IEEE.

## 3. Bộ dữ liệu

### 3.1. Nguồn dữ liệu

Bộ dữ liệu dự kiến sử dụng là Cleveland Heart Disease, thuộc UCI Machine Learning Repository.

- **Tên tệp:** `heart_disease_uci.csv`
- **Số lượng mẫu dự kiến:** 303 bệnh nhân.
- **Số đặc trưng đầu vào:** 13.
- **Nhãn dự đoán:** `target`.
- **Bài toán:** Phân loại nhị phân.

Nguồn tham khảo: https://archive.ics.uci.edu/dataset/45/heart+disease

**Lưu ý về dữ liệu:** Cần kiểm tra cấu trúc và phiên bản CSV thực tế trước khi chạy mô hình. Một số phiên bản Cleveland có thể sử dụng cột `num` với các giá trị từ 0 đến 4 thay vì cột `target` nhị phân. Nếu cần chuyển thành hai lớp, phải xác định rõ quy tắc chuyển đổi: `num = 0` thành lớp 0 và `num > 0` thành lớp 1.

### 3.2. Các thuộc tính dữ liệu

| Thuộc tính | Ý nghĩa |
|---|---|
| `age` | Tuổi bệnh nhân |
| `sex` | Giới tính |
| `cp` | Loại đau ngực |
| `trestbps` | Huyết áp lúc nghỉ |
| `chol` | Cholesterol huyết thanh |
| `fbs` | Đường huyết lúc đói |
| `restecg` | Kết quả điện tâm đồ lúc nghỉ |
| `thalach` | Nhịp tim tối đa đạt được |
| `exang` | Đau thắt ngực do gắng sức |
| `oldpeak` | Mức chênh ST khi gắng sức so với lúc nghỉ |
| `slope` | Độ dốc đoạn ST khi gắng sức |
| `ca` | Số mạch máu chính được quan sát bằng phương pháp huỳnh quang |
| `thal` | Kết quả thăm dò thalassemia theo mã hóa của bộ dữ liệu |
| `target` | Nhãn bệnh tim sau khi chuẩn hóa về hai lớp |

**Ý nghĩa nhãn:**

- `target = 0`: Không ghi nhận bệnh tim theo quy ước nhãn của bộ dữ liệu.
- `target = 1`: Có bệnh tim theo quy ước nhãn của bộ dữ liệu.

Phân bố 164 mẫu lớp 0 và 139 mẫu lớp 1 chỉ được sử dụng nếu kiểm tra CSV thực tế xác nhận đúng các số lượng này.

## 4. Cấu trúc báo cáo

Báo cáo được chia thành 7 chương:

### Chương 1: Giới thiệu

- Tổng quan về Khai phá dữ liệu và Machine Learning.
- Giới thiệu nhóm thuật toán Tree-based.
- Lý do lựa chọn đề tài.
- Mục đích và mục tiêu nghiên cứu.

### Chương 2: Tổng quan ứng dụng và công trình liên quan

- Ứng dụng của Decision Tree và Random Forest.
- Các công trình liên quan đến ID3, C4.5, CART, Bagging và Random Forest.
- So sánh và nhận xét các nghiên cứu liên quan.

### Chương 3: Cơ sở lý thuyết

- Khái niệm và cơ chế hoạt động của Decision Tree.
- Entropy, Information Gain và Gini Impurity.
- Hiện tượng Overfitting và phương pháp Pruning.
- Cơ chế Bootstrap, Bagging và Random Subspace.
- Majority Voting và Out-of-Bag.
- Sơ đồ khối mô tả hai thuật toán.

### Chương 4: Thực nghiệm

- Mô tả bộ dữ liệu.
- Phân tích khám phá dữ liệu (EDA).
- Tiền xử lý dữ liệu.
- Xây dựng mô hình Decision Tree.
- Xây dựng mô hình Random Forest.
- Tối ưu tham số và đánh giá mô hình.

### Chương 5: Phân tích kết quả

- Bảng so sánh các chỉ số đánh giá.
- Phân tích Confusion Matrix.
- Phân tích hiện tượng Overfitting.
- Đánh giá Feature Importance.
- Giải thích và thảo luận kết quả thực nghiệm.

### Chương 6: Kết luận

- Tổng kết nội dung nghiên cứu.
- So sánh điểm mạnh và điểm yếu của các thuật toán.
- Hạn chế của dự án.
- Hướng phát triển trong tương lai.

### Chương 7: Tài liệu tham khảo

- Các bài báo khoa học.
- Tài liệu về thuật toán học máy.
- Nguồn dữ liệu UCI.
- Tài liệu thư viện Python và Scikit-learn.

## 5. Cơ sở lý thuyết cần nắm vững

### 5.1. Entropy

Entropy đo mức độ hỗn loạn của các nhãn tại một nút.

Công thức:

`Entropy(S) = -Σ p_i log₂(p_i)`

Trong đó `p_i` là tỷ lệ mẫu thuộc lớp thứ `i`.

### 5.2. Information Gain

Information Gain đo mức giảm Entropy sau khi phân chia dữ liệu theo một thuộc tính.

`IG(S, A) = Entropy(S) - Σ (|S_v| / |S|) × Entropy(S_v)`

Thuộc tính có Information Gain lớn hơn sẽ đem lại mức giảm độ hỗn loạn lớn hơn theo tiêu chí này.

### 5.3. Gini Impurity

Gini Impurity đo mức độ không thuần nhất của các lớp trong một nút.

`Gini(S) = 1 - Σ p_i²`

Với dữ liệu gồm hai lớp, tỷ lệ lớp lần lượt là `p₀` và `p₁`:

`Gini(S) = 1 - (p₀² + p₁²)`

Nếu CSV thực tế có 164 mẫu lớp 0 và 139 mẫu lớp 1, cần tính lại các giá trị Entropy và Gini trực tiếp từ dữ liệu để tránh sai số do làm tròn.

### 5.4. Decision Tree

Decision Tree phân chia dữ liệu thành các nhánh dựa trên điều kiện của thuộc tính. Quá trình tiếp tục cho đến khi đạt điều kiện dừng.

Các nội dung cần tìm hiểu:

- Lựa chọn thuộc tính phân nhánh.
- Độ sâu của cây (`max_depth`).
- Số mẫu tối thiểu để chia nút (`min_samples_split`).
- Số mẫu tối thiểu tại nút lá (`min_samples_leaf`).
- Cắt tỉa cây (`Pruning`).
- Hiện tượng Overfitting.

### 5.5. Random Forest

Random Forest kết hợp nhiều cây quyết định để cải thiện khả năng tổng quát hóa.

Các cơ chế chính:

1. **Bootstrap Sampling:** Lấy mẫu có hoàn lại để tạo tập huấn luyện cho từng cây.
2. **Random Subspace:** Chọn ngẫu nhiên một tập con thuộc tính khi xem xét phân nhánh.
3. **Bagging:** Huấn luyện nhiều cây trên các tập mẫu khác nhau.
4. **Majority Voting:** Tổng hợp dự đoán của các cây bằng bỏ phiếu đa số trong bài toán phân loại.
5. **Out-of-Bag:** Sử dụng những mẫu không được chọn vào bootstrap của một cây để ước lượng hiệu quả mô hình.

## 6. Kế hoạch thực hiện trong 7 ngày

| Ngày | Công việc | Sản phẩm |
|---|---|---|
| Ngày 1 | Thu thập dữ liệu, viết Chương 1 và Chương 2 | CSV và nội dung tổng quan |
| Ngày 2 | Tìm hiểu công thức toán, thuật toán và viết Chương 3 | Cơ sở lý thuyết và sơ đồ khối |
| Ngày 3 | EDA và tiền xử lý dữ liệu | Biểu đồ EDA, dữ liệu sạch |
| Ngày 4 | Huấn luyện và cắt tỉa Decision Tree | Mô hình cây, biểu đồ Overfitting |
| Ngày 5 | Huấn luyện Random Forest và so sánh mô hình | Bảng kết quả, Confusion Matrix, Feature Importance |
| Ngày 6 | Phân tích kết quả, viết Chương 5–7 | Báo cáo hoàn chỉnh |
| Ngày 7 | Định dạng báo cáo, làm slide và ôn vấn đáp | Word/PDF, slide và notebook |

## 7. Quy trình thực nghiệm

### Bước 1: Thu thập dữ liệu

- Tải dữ liệu từ nguồn đáng tin cậy.
- Lưu tệp `heart_disease_uci.csv`.
- Kiểm tra số dòng, số cột, tên thuộc tính và nhãn mục tiêu.

### Bước 2: Khám phá dữ liệu (EDA)

- Xem cấu trúc dữ liệu bằng `head()`, `info()` và `describe()`.
- Kiểm tra phân bố nhãn.
- Kiểm tra dữ liệu khuyết và dữ liệu trùng lặp.
- Vẽ biểu đồ phân bố bệnh tim theo tuổi và loại đau ngực.
- Vẽ ma trận tương quan cho các thuộc tính số phù hợp.

### Bước 3: Tiền xử lý

- Chuẩn hóa tên cột nếu cần.
- Xác định nhãn mục tiêu và quy tắc mã hóa nhãn.
- Xử lý giá trị khuyết bằng phương pháp phù hợp.
- Phân biệt dữ liệu phân loại và dữ liệu số.
- Chia tập huấn luyện và kiểm thử theo tỷ lệ dự kiến 80:20, có thể dùng `stratify=y`.

Với 303 mẫu, phép chia 80:20 thường tạo ra khoảng 242 mẫu huấn luyện và 61 mẫu kiểm thử. Số lượng chính xác phụ thuộc cách chia dữ liệu.

**Nguyên tắc quan trọng:** Các bước học tham số tiền xử lý phải được thực hiện trên tập huấn luyện để tránh rò rỉ dữ liệu. Nên sử dụng `Pipeline` và `ColumnTransformer` của Scikit-learn khi phù hợp.

### Bước 4: Huấn luyện Decision Tree

- Xây dựng cây chưa giới hạn độ sâu.
- Huấn luyện cây có cắt tỉa hoặc giới hạn độ sâu.
- Tối ưu tham số bằng Cross-Validation trên tập huấn luyện.
- So sánh Accuracy của tập Train và Test.
- Vẽ cấu trúc cây quyết định.

### Bước 5: Huấn luyện Random Forest

- Khởi tạo `RandomForestClassifier`.
- Thử nghiệm số lượng cây và các tham số quan trọng.
- Có thể bật `oob_score=True` để đánh giá Out-of-Bag khi cấu hình lấy mẫu phù hợp.
- Trích xuất Feature Importance.

### Bước 6: Đánh giá và so sánh

Các chỉ số đánh giá:

- **Accuracy:** Tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision:** Tỷ lệ dự đoán dương tính đúng trên tổng số dự đoán dương tính.
- **Recall:** Tỷ lệ phát hiện đúng các mẫu dương tính thực tế.
- **F1-Score:** Trung bình điều hòa của Precision và Recall.
- **ROC-AUC:** Đánh giá khả năng phân biệt hai lớp qua các ngưỡng dự đoán.
- **Confusion Matrix:** Thể hiện số mẫu dự đoán đúng và sai theo từng lớp.

Trong bài toán liên quan đến sức khỏe, cần chú ý Recall của lớp bệnh tim và số trường hợp dương tính giả, âm tính giả. Không nên chỉ dựa vào Accuracy để kết luận mô hình tốt hơn.

## 8. Danh sách biểu đồ cần có

- [ ] Biểu đồ phân bố nhãn bệnh tim.
- [ ] Biểu đồ phân bố bệnh tim theo tuổi.
- [ ] Biểu đồ phân bố bệnh tim theo loại đau ngực.
- [ ] Heatmap tương quan.
- [ ] Biểu đồ Train Accuracy và Test Accuracy theo độ sâu cây.
- [ ] Hình trực quan hóa Decision Tree.
- [ ] Confusion Matrix của các mô hình.
- [ ] Biểu đồ so sánh các chỉ số đánh giá.
- [ ] Biểu đồ Top 10 Feature Importance của Random Forest.

## 9. Cấu trúc thư mục dự án đề xuất

```text
khaipha_dulieu/
├── README.md
├── data/
│   └── heart_disease_uci.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── figures/
│   ├── target_distribution.png
│   ├── age_chest_pain_distribution.png
│   ├── correlation_heatmap.png
│   ├── decision_tree.png
│   ├── overfitting_analysis.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── report/
│   ├── bao_cao.docx
│   └── bao_cao.pdf
└── slides/
    └── thuyet_trinh.pptx
```

Đây là cấu trúc đề xuất. Có thể điều chỉnh tên thư mục và tệp theo cách tổ chức thực tế trên Google Colab hoặc GitHub.

## 10. Checklist tiến độ

### Giai đoạn 1: Lý thuyết

- [ ] Hoàn thành Chương 1.
- [ ] Hoàn thành Chương 2.
- [ ] Hoàn thành Chương 3.
- [ ] Kiểm tra công thức Entropy, Information Gain và Gini.
- [ ] Hiểu cơ chế hoạt động của Decision Tree và Random Forest.

### Giai đoạn 2: Thực nghiệm

- [ ] Tải và kiểm tra dữ liệu.
- [ ] Hoàn thành EDA.
- [ ] Xử lý dữ liệu khuyết.
- [ ] Chia tập Train/Test.
- [ ] Huấn luyện Decision Tree chưa cắt tỉa.
- [ ] Huấn luyện Decision Tree có cắt tỉa.
- [ ] Huấn luyện Random Forest.
- [ ] Tối ưu tham số.
- [ ] Xuất bảng đánh giá và biểu đồ.

### Giai đoạn 3: Báo cáo và thuyết trình

- [ ] Hoàn thành Chương 4.
- [ ] Hoàn thành Chương 5.
- [ ] Hoàn thành Chương 6.
- [ ] Hoàn thành Chương 7.
- [ ] Kiểm tra tài liệu tham khảo theo chuẩn IEEE.
- [ ] Định dạng báo cáo Word và xuất PDF.
- [ ] Hoàn thiện slide thuyết trình.
- [ ] Ôn tập câu hỏi vấn đáp.

## 11. Tài liệu tham khảo ban đầu

1. J. R. Quinlan, “Induction of Decision Trees,” *Machine Learning*, vol. 1, pp. 81–106, 1986.
2. L. Breiman, “Bagging Predictors,” *Machine Learning*, vol. 24, pp. 123–140, 1996.
3. L. Breiman, “Random Forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001.
4. UCI Machine Learning Repository, “Heart Disease,” https://archive.ics.uci.edu/dataset/45/heart+disease
5. Scikit-learn, “Decision Trees,” https://scikit-learn.org/stable/modules/tree.html
6. Scikit-learn, “Ensembles: Random Forests,” https://scikit-learn.org/stable/modules/ensemble.html#forest

Các tài liệu trên là danh mục khởi đầu. Cần kiểm tra và bổ sung thông tin xuất bản chính xác cho toàn bộ tài liệu được trích dẫn trong báo cáo.

## 12. Kết quả mong đợi

Sau khi hoàn thành dự án, nhóm cần có:

- Báo cáo đầy đủ 7 chương.
- Notebook Python thực nghiệm trên Google Colab.
- Bộ dữ liệu CSV đã kiểm tra và tiền xử lý.
- Các biểu đồ EDA và trực quan hóa mô hình.
- Bảng so sánh Decision Tree chưa cắt tỉa, Decision Tree có cắt tỉa và Random Forest.
- Phân tích kết quả, ưu điểm, nhược điểm và hạn chế.
- Slide thuyết trình và bộ câu hỏi ôn tập vấn đáp.

**Nguyên tắc thực hiện:** Mọi số liệu, biểu đồ và nhận xét về độ chính xác mô hình phải lấy từ kết quả chạy thực tế. Không tự đặt số liệu hoặc khẳng định Random Forest luôn tốt hơn Decision Tree trước khi có kết quả thực nghiệm.
