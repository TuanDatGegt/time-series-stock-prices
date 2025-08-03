## Time Series Forecasting

Dự đoán chuỗi thời gian là một chuỗi các điểm dữ liệu theo thứ tự thời gian được các doanh nghiệp sử dụng để phân tích dữ liệu trong quá khứ và đưa ra dự đoán trong tương lai. Các điểm dữ liệu này là một tập hợp các quan sát tại các thời điểm cụ thể và các khoảng thời gian bằng nhau, thường có chỉ số ngày giờ và giá trị tương ứng.

---

## Dữ liệu chuỗi thời gian bao gồm bốn thành phần:

- **Thành phần xu hướng (Trend)**: Biến động tăng hoặc giảm theo một mô hình có thể dự đoán được trong một khoảng thời gian dài.
- **Thành phần theo mùa (Seasonality)**: Biến động đều đặn tuần hoàn và lặp lại trong một khoảng thời gian cụ thể như ngày, tuần, tháng, mùa,...
- **Thành phần theo chu kỳ (Cyclical)**: Biến động tương ứng với các chu kỳ "boom-bust" của doanh nghiệp hoặc nền kinh tế, hoặc tuân theo các chu kỳ riêng biệt.
- **Thành phần ngẫu nhiên (Irregular/Noise)**: Biến động bất thường hoặc dư thừa, không thuộc bất kỳ phân loại nào trong ba loại trên.

---

## 🎯 Mục tiêu:

Dự đoán giá cổ phiếu trong **10 ngày tiếp theo**. Đưa ra đánh giá về xu hướng giá cổ phiếu Intel (INTC) trong bối cảnh thị trường biến động bởi suy thoái toàn cầu, cuộc đua công nghệ phần cứng, và ảnh hưởng của các yếu tố chính trị như chiến tranh Trung Đông - Mỹ. 

Mục đích của báo cáo là **đề xuất chiến lược đầu tư** phù hợp cho mã cổ phiếu Intel bằng cách phân tích các yếu tố định lượng và định tính, từ đó hỗ trợ nhà đầu tư ra quyết định mua/bán cổ phiếu một cách chính xác.

---

## Câu hỏi được đặt ra:

- Khi nào cổ phiếu đạt **giá cao nhất/thấp nhất**?
- **Thời điểm nên mua hoặc bán** cổ phiếu là khi nào?
- Tại sao công ty lại bán một lượng lớn cổ phiếu vào một thời điểm nhất định?
- **Yếu tố truyền thông và chính trị** ảnh hưởng như thế nào đến biến động giá cổ phiếu?

---

## 📥 Thu thập dữ liệu:

- Dữ liệu được thu thập từ [Yahoo! Finance](https://finance.yahoo.com) thông qua thư viện **`yfinance`** của Python.
- Hàm sử dụng: `yf.Ticker("INTC").history()`

---

## 🧹 Data Preprocessing & Exploratory Data Analysis (EDA)

Dữ liệu được lấy từ `yfinance` bằng hàm `history()` cho mã cổ phiếu Intel (INTC), từ ngày **01/08/1995** đến **01/08/2025**.

- Kích thước dữ liệu: **7551 dòng × 7 cột**
- Mỗi dòng tương ứng với **dữ liệu giao dịch của một ngày**

### 🧾 Các cột dữ liệu bao gồm:

- `Open`: Giá mở cửa hằng ngày
- `High`: Giá cao nhất trong ngày
- `Low`: Giá thấp nhất trong ngày
- `Close`: Giá đóng cửa
- `Volume`: Khối lượng giao dịch
- `Dividends`: Cổ tức được chia (nếu có)
- `Stock Splits`: Thông tin chia tách cổ phiếu (nếu có)

---

### 📋 Ví dụ về dữ liệu

| Date                | Open      | High      | Low       | Close     | Volume     | Dividends | Stock Splits |
|---------------------|-----------|-----------|-----------|-----------|------------|-----------|---------------|
| 1995-08-01 00:00:00 | 4.579953  | 4.579953  | 4.403801  | 4.456646  | 94556800   | 0.0       | 0.0           |
| 1995-08-02 00:00:00 | 4.535913  | 4.579951  | 4.298107  | 4.333338  | 135620800  | 0.0       | 0.0           |
| 1995-08-03 00:00:00 | 4.210034  | 4.421416  | 4.165996  | 4.377378  | 117961600  | 0.0       | 0.0           |
| ...                 | ...       | ...       | ...       | ...       | ...        | ...       | ...           |
| 2025-07-31 00:00:00 | 20.170000 | 20.230000 | 19.660000 | 19.799999 | 90665200   | 0.0       | 0.0           |
| 2025-08-01 00:00:00 | 19.500000 | 19.549999 | 18.969999 | 19.309999 | 86320300   | 0.0       | 0.0           |

---

> ✅ Tiền xử lý và phân tích dữ liệu sẽ giúp phát hiện xu hướng, mùa vụ, hoặc biến động bất thường của giá cổ phiếu, từ đó hỗ trợ xây dựng mô hình dự báo chính xác hơn.


