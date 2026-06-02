# Transportation Algorithm Visualizer

App Streamlit minh họa và so sánh các thuật toán giải bài toán vận tải (Transportation Problem).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
streamlit run app.py
```

## Tính năng

| Tính năng | Chi tiết |
|-----------|----------|
| **Quy trình giải** | Giai đoạn 1: NW/LCM/VAM tìm điểm xuất phát; giai đoạn 2: MODI tối ưu; LP làm mốc tối ưu; Hungarian cho bài phân công |
| **Input** | 7 ví dụ có sẵn, nhập tay m×n, upload CSV |
| **Xử lý** | Balanced/Unbalanced (dummy), Forbidden cells (Big-M), Max→Min transform |
| **Visualization** | Heatmap chi phí, heatmap phân bổ, heatmap Delta MODI, bảng làm bài từng bước, network graph, bar chart khoảng cách tối ưu |
| **Export** | PNG heatmap, PNG network, CSV allocation, JSON result |

## Lộ trình ví dụ chiến thuật

Hệ thống bao gồm 12 ví dụ được thiết kế theo lộ trình giáo trình:

1.  **Nhóm Cơ bản:** Bài toán cân bằng chuẩn.
2.  **Nhóm Kỹ thuật:** Tranh chấp phí (Tie-breaking), Suy biến (Degeneracy) giai đoạn đầu và trong lúc tối ưu.
3.  **Nhóm Chiến thuật:** So sánh VAM vs LCM (Cái bẫy của sự tham lam), Đa nghiệm tối ưu.
4.  **Nhóm Thực tế:** Cung > Cầu (Phí lưu kho), Cầu > Cung (Phí phạt), Tuyến đường bị cấm.
5.  **Nhóm Đặc biệt:** Vô nghiệm (Infeasible), Phân công (Assignment), Logistics Việt Nam (Quy mô lớn).

## Quy ước

- `Delta_ij = u_i + v_j - c_ij` (dương = có thể cải thiện)
- Bài Min tối ưu khi mọi `Delta_ij ≤ 0`
- `u[0] = 0` khi tính potentials
- MODI đảm bảo cho balanced min, kích thước ≤ 6×6

## Chạy tests

```bash
python -m pytest tests/ -v
```

28 tests, 0 failures.
