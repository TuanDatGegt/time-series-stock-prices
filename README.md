## Time Series Forecasting
Dự đoán chuỗi thời gian này là một chỗi các điểm dữ liệu theo thứ tự thời gian được các doanh nghiệp sử dụng để phân tích dữ liệu trong quá khứ và đưa ra dự đoán trong tương lai. Các điểm dữ liệu này là một tập hợp các quan sát tại các thời điểm cụ thể và các khoảng thời gian bằng nhau, thường có chỉ số ngày giờ và giá trị tương ứng.



## Dữ liệu chuỗi thời gian bao gồm bốn thành phần:

- Thành phần xu hướng: Đây là biến động tăng hoặc giảm theo một mô hình có thể dự đoán được trong một khoảng thời gian dài.
- Thành phần theo mùa: Là biến động đều đặn tuần hoàn và lặp lại trong một khoảng thời gian cụ thể như ngày, tuần, tháng, mùa,...
- Thành phần theo chu kỳ: Là biến động tương ứng với các chu kỳ "boom-bust" của doanh nghiệp hoặc nền kinh tế, hoặc tuân theo các chu kỳ riêng biệt của chúng.
- Thành phần ngẫu nhiên: Là biến động bất thường hoặc dư thừa và không thuộc bất kỳ phân loại nào trong ba phân loại trên.


## Mục tiêu hướng đến: 
Dự đoán ra kết quả của giá cổ phiếu trong 10 ngày tiếp theo. Đưa ra đánh giá về dự đoán về giá cả của mã cổ phiếu. Trong bối cảnh thị trường biến động suy thoái toàn cầu, cuộc đua về công nghệ phần cứng. Tác động của chiến tranh của các nước trung đông với Mỹ góp phần ảnh hưởng đến thị trường chứng khoáng tài chính. Vì lẽ đó nên mục đích của bài báo cáo này nhằm đưa ra chiến lược phù hợp cho nhà đầu dự đoán xu hướng của mã cổ phiếu Intel (INTC) nhằm mục đích đưa ra phán đoán chính xác về cách nhìn từ những con số và thông tin chính trị mang lại. Điều này giúp nhà đầu tư có thể đưa ra các quyết định nên làm gì với mã cổ phiếu này.

Trong đề tài này, Tôi sử dụng mã cổ phiếu của Intel (INTC). Câu hỏi trong dự án nhỏ này là các giá trị Open, High, Low, Close, Volume, Khi nào có được giá cao nhất, khi nào có giá thấp nhất, nhà đầu tư bán cổ phiếu khi nào? Nhà đầu tư muốn mua cổ phiếu khi nào? Vì sao công ty phát hành cổ phiếu bán đi một lượng lớn cổ phiếu vào một khoảng thời gian đó? Truyền thông và chính trị của đất nước đó ảnh hưởng như thế nào đến giá cổ phiếu?

## Thu thập dữ liệu:
Dữ liệu được thu thập từ trang web Yahoo!Finance. Sử dụng một thư viện yfinance có sẵn trong Python, đây là một thư viện được xây dựng dựa trên việc thu thập dữ liệu từ trang web của Yahoo!Finance.

## Data Preprocessing & Exploratory Data Analysis
Dữ liệu được lấy từ thư viện yfinance bằng hàm history() dùng để lấy mã lịch sử giao dịch của cổ phiếu Intel(INTC) từ ngày 1/8/1995 đến thời gian thực hiện tại.

Dữ liệu bao gồm 7550 dòng và 7 cột. Mỗi dòng tương ứng với dữ liệu giao dịch của 1 ngày:
    * Open: Giá mở cửa hằng ngày.
    * High: Giá cao nhất trong ngày.
    * Low: Giá thấp nhất trong ngày.
    * Close: Giá đóng cửa.
    * Volume: Khối lượng giao dịch.
    * Dividends: Cổ tức được chia(nếu có).
    * Stock Splits: Thông tin chia tách cổ phiếu(nếu có).

    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1995-08-01 00:00:00-04:00</th>
      <td>4.579953</td>
      <td>4.579953</td>
      <td>4.403801</td>
      <td>4.456646</td>
      <td>94556800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1995-08-02 00:00:00-04:00</th>
      <td>4.535913</td>
      <td>4.579951</td>
      <td>4.298107</td>
      <td>4.333338</td>
      <td>135620800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1995-08-03 00:00:00-04:00</th>
      <td>4.210034</td>
      <td>4.421416</td>
      <td>4.165996</td>
      <td>4.377378</td>
      <td>117961600</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1995-08-04 00:00:00-04:00</th>
      <td>4.386187</td>
      <td>4.439033</td>
      <td>4.350957</td>
      <td>4.368572</td>
      <td>68723200</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1995-08-07 00:00:00-04:00</th>
      <td>4.403798</td>
      <td>4.500682</td>
      <td>4.386183</td>
      <td>4.474259</td>
      <td>51580000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-07-28 00:00:00-04:00</th>
      <td>20.820000</td>
      <td>21.290001</td>
      <td>20.650000</td>
      <td>20.680000</td>
      <td>86105600</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2025-07-29 00:00:00-04:00</th>
      <td>20.690001</td>
      <td>20.850000</td>
      <td>20.340000</td>
      <td>20.410000</td>
      <td>100831500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2025-07-30 00:00:00-04:00</th>
      <td>20.430000</td>
      <td>20.620001</td>
      <td>20.080000</td>
      <td>20.340000</td>
      <td>67420300</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2025-07-31 00:00:00-04:00</th>
      <td>20.170000</td>
      <td>20.230000</td>
      <td>19.660000</td>
      <td>19.799999</td>
      <td>90665200</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2025-08-01 00:00:00-04:00</th>
      <td>19.500000</td>
      <td>19.549999</td>
      <td>18.969999</td>
      <td>19.309999</td>
      <td>86320300</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7551 rows × 7 columns</p>
</div>


