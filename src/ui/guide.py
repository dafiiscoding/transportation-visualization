from __future__ import annotations
import streamlit as st


def render_guide() -> None:
    st.markdown("""
    <h2 style='color:#1e40af;margin-bottom:4px'>📖 Hướng dẫn sử dụng</h2>
    <p style='color:#64748b;margin-top:0'>Transportation Algorithm Visualizer — giải & trực quan hoá bài toán vận tải</p>
    """, unsafe_allow_html=True)

    # ── Quick start ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#eff6ff;border-left:4px solid #2563eb;border-radius:8px;padding:14px 18px;margin-bottom:16px'>
    <b>🚀 Bắt đầu nhanh</b><br>
    1. Chọn <b>Ví dụ có sẵn</b> ở sidebar → chọn ví dụ<br>
    2. Chọn phương pháp <b>tìm điểm xuất phát</b>, bật MODI và LP nếu cần<br>
    3. Bấm <b>▶ Chạy</b> → xem lần lượt điểm xuất phát, MODI và tổng hợp ở tab <b>📊 Kết quả</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Nhập dữ liệu ──────────────────────────────────────────────
    with st.expander("1️⃣  Nhập dữ liệu — 3 chế độ", expanded=True):
        tab_ex, tab_manual, tab_csv = st.tabs(["Ví dụ có sẵn", "Nhập tay", "Upload CSV"])

        with tab_ex:
            st.markdown("""
**Ví dụ có sẵn** là lộ trình được thiết kế để bao quát mọi trường hợp từ cơ bản đến nâng cao.

| Nhóm | Mục tiêu |
|---|---|
| **1. Cơ bản** | Làm quen giao diện, so sánh các thuật toán điểm xuất phát. |
| **2. Kỹ thuật** | Xem cách hệ thống xử lý **Suy biến (Degeneracy)** và **Tranh chấp phí**. |
| **3. Chiến thuật** | Chứng minh tại sao Vogel (VAM) lại thông minh hơn Least Cost (LCM). |
| **4. Thực tế** | Bài toán **không cân bằng** với phí lưu kho/phí phạt và tuyến đường bị cấm. |
| **5. Đặc biệt** | Bài toán **vô nghiệm**, bài toán **Phân công**, và logistics quy mô lớn. |

> **Tip:** Hãy bắt đầu từ ví dụ số 1 và đi dần xuống để nắm vững toàn bộ lý thuyết.
            """)

        with tab_manual:
            st.markdown("""
**Nhập tay** cho phép bạn tự định nghĩa bài toán.

**Các bước:**
1. Đặt số nguồn `m` và số đích `n`
2. Đặt tên cho từng nguồn (A1, A2…) và đích (B1, B2…)
3. Nhập **Cung (Supply)** cho mỗi nguồn
4. Nhập **Cầu (Demand)** cho mỗi đích
5. Điền **ma trận chi phí** — chi phí vận chuyển 1 đơn vị từ nguồn i đến đích j
6. (Tuỳ chọn) Đánh dấu **ô cấm** nếu có tuyến không sử dụng được

> **Lưu ý:** Cung & cầu không cần cân bằng — ứng dụng tự thêm nguồn/đích ảo.
            """)

        with tab_csv:
            st.markdown("""
**Upload CSV** cho phép nhập dữ liệu từ file.

**Định dạng CSV:**
```
,B1,B2,B3,B4,Supply
A1,2,3,1,5,120
A2,7,3,4,6,80
A3,2,4,3,2,80
Demand,150,70,100,60,
```
- Hàng đầu: tên cột đích + cột `Supply`
- Cột đầu: tên nguồn
- Hàng cuối: hàng `Demand`

> Bấm **⬇ Tải template CSV** để lấy file mẫu đúng định dạng.
            """)

    # ── Section 2: Cấu trúc bài toán ─────────────────────────────────────────
    with st.expander("2️⃣  Hiểu cấu trúc bài toán"):
        st.markdown("""
**Bài toán vận tải** tìm phương án phân bổ hàng hoá từ *m* nguồn đến *n* đích với tổng chi phí nhỏ nhất (hoặc lợi nhuận lớn nhất).

```
         B1   B2   B3  │ Supply
  A1  [  2    3    1  ] │   120
  A2  [  7    3    4  ] │    80
  ────────────────────────────
Demand  150   70  100
```

| Khái niệm | Ý nghĩa |
|---|---|
| **Supply (Cung)** | Lượng hàng nguồn i có thể cung cấp |
| **Demand (Cầu)** | Lượng hàng đích j cần nhận |
| **cᵢⱼ (Chi phí)** | Chi phí vận chuyển 1 đơn vị từ i → j |
| **xᵢⱼ (Phân bổ)** | Lượng hàng thực tế vận chuyển từ i → j |
| **Cân bằng** | Tổng cung = Tổng cầu (nếu không, ứng dụng tự cân bằng) |
| **Ô cấm** | Tuyến i→j không được sử dụng (chi phí = ∞) |

**Bài toán tối đa hoá:** Hệ thống đổi dấu ma trận chi phí (cᵢⱼ → −cᵢⱼ) để chuyển về bài toán tối thiểu hoá, sau đó hiển thị lại lợi nhuận thực.
        """)

    # ── Section 3: Thuật toán ─────────────────────────────────────────────────
    with st.expander("3️⃣  Quy trình giải — điểm xuất phát → MODI → LP"):
        st.markdown("""
Ứng dụng tách bài toán thành 3 phần:

1. **Tìm điểm xuất phát:** NW, LCM hoặc VAM tạo phương án cơ sở ban đầu.
2. **MODI tối ưu hóa:** bắt đầu từ từng phương án cơ sở và lặp đến khi không còn Δ dương.
3. **LP tối ưu:** giải trực tiếp bằng quy hoạch tuyến tính để làm mốc so sánh.

### Northwest Corner (NW)
- **Cách làm:** Bắt đầu từ ô trên trái, phân bổ min(cung, cầu), rồi dịch sang phải hoặc xuống.
- **Ưu điểm:** Cực kỳ đơn giản, dễ hiểu từng bước.
- **Nhược điểm:** Không xem xét chi phí → thường ra nghiệm xa tối ưu.
- **Dùng khi:** Học khái niệm cơ bản, làm điểm xuất phát cho MODI.

---

### Least Cost Method (LCM)
- **Cách làm:** Mỗi bước chọn ô chi phí nhỏ nhất còn dùng được, phân bổ tối đa vào ô đó.
- **Ưu điểm:** Xem xét chi phí → nghiệm tốt hơn NW đáng kể.
- **Tie-breaking:** (1) Tiềm năng phân bổ lớn hơn → (2) Chỉ số hàng nhỏ hơn → (3) Chỉ số cột nhỏ hơn.
- **Dùng khi:** Bài toán nhỏ, muốn nghiệm tốt mà không cần tối ưu hoàn toàn.

---

### Vogel's Approximation Method (VAM)
- **Cách làm:** Tính *penalty* = chênh lệch 2 chi phí nhỏ nhất cho mỗi hàng/cột. Chọn hàng/cột có penalty lớn nhất, phân bổ vào ô rẻ nhất.
- **Ưu điểm:** Thường cho nghiệm gần tối ưu hoặc bằng tối ưu.
- **Dùng khi:** Muốn nghiệm tốt nhanh mà không giải LP.

---

### MODI (Modified Distribution Method)
- **Cách làm:** Từ phương án cơ bản (NW/LCM/VAM), tính thế vị uᵢ, vⱼ sao cho uᵢ + vⱼ = cᵢⱼ với mọi ô cơ sở. Sau đó tính Δᵢⱼ = uᵢ + vⱼ − cᵢⱼ cho ô ngoài cơ sở.
- **Điều kiện tối ưu:** Tất cả Δᵢⱼ ≤ 0 (bài toán min).
- **Nếu chưa tối ưu:** Đưa ô có Δ > 0 lớn nhất vào cơ sở, tìm chu trình, điều chỉnh phân bổ.
- **Giới hạn:** Chỉ đảm bảo kết quả đúng với bài toán cân bằng, kích thước ≤ 6×6.

---

### LP Solver
- **Cách làm:** Giải bài toán quy hoạch tuyến tính bằng HiGHS solver (scipy).
- **Vai trò trong app:** Là mốc tối ưu toàn cục. LP không cần so với chính nó.
- **Dùng khi:** Cần nghiệm chuẩn, hoặc bài toán lớn (>6×6).
- **Cách đọc:** Các phương pháp khác được ghi rõ còn cách mốc LP bao nhiêu.
        """)

    # ── Section 4: Đọc kết quả ───────────────────────────────────────────────
    with st.expander("4️⃣  Đọc kết quả — tab 📊 Kết quả"):
        st.markdown("""
Sau khi bấm **▶ Chạy**, tab **📊 Kết quả** hiển thị:

**Hàng metrics trên cùng:**
- Điểm xuất phát tốt nhất.
- Kết quả MODI tốt nhất.
- Mốc LP tối ưu toàn cục.

**Các bảng kết quả:**

| Cột | Ý nghĩa |
|---|---|
| Phương pháp / Bắt đầu từ | Nguồn tạo phương án |
| Chi phí / Lợi nhuận | Giá trị của phương án |
| Cách mốc tối ưu | Khoảng cách tuyệt đối và % so với LP |
| Số bước | Số bước lập phương án hoặc số vòng MODI |
| Đường đi | Chuỗi giá trị qua từng vòng MODI |

**Biểu đồ thanh:** So sánh trực quan các phương án, đường đỏ đứt = mốc LP tối ưu.

> **Tip:** MODI không nằm chung danh sách với thuật toán điểm xuất phát; nó là giai đoạn tối ưu hóa sau NW/LCM/VAM.
        """)

    # ── Section 5: Từng bước ─────────────────────────────────────────────────
    with st.expander("5️⃣  Xem từng bước — tab 🔍 Từng bước"):
        st.markdown("""
Tab **🔍 Từng bước** cho phép bạn theo dõi từng thao tác của thuật toán.

**Cách sử dụng:**
1. Chọn giai đoạn: điểm xuất phát, MODI hoặc LP.
2. Chọn phương pháp trong giai đoạn đó.
3. Kéo thanh trượt hoặc bấm **◀ ▶** để di chuyển giữa các bước.
4. Mỗi bước hiển thị:
   - **Tiêu đề & mô tả** — ô nào được chọn, lý do, lượng phân bổ.
   - **Bảng làm bài** — mỗi ô có chi phí, phân bổ, Δ và dấu chu trình nếu có.
   - **Heatmap** — ô xanh lá = ô đang được chọn ở bước này.
   - **Bảng Penalty** (VAM) — giá trị penalty của từng hàng/cột.
   - **Heatmap Delta** (MODI) — ô có Δ dương là cơ hội cải thiện.
   - **Chu trình** (MODI) — chuỗi ô tạo thành vòng điều chỉnh.

> **Tip:** Dùng tab này để học cách hoạt động của từng thuật toán từng bước một.
        """)

    # ── Section 6: Visualisation ─────────────────────────────────────────────
    with st.expander("6️⃣  Visualisation — tab 🗺"):
        st.markdown("""
Tab **🗺 Visualisation** có 3 loại biểu đồ:

**Heatmap chi phí** (trái)
- Màu càng đậm (đỏ) → chi phí càng cao.
- Ô đỏ với ký hiệu ∞ = ô cấm.

**Heatmap phân bổ** (phải)
- Màu xanh dương càng đậm → lượng phân bổ càng nhiều.
- Chọn kết quả theo giai đoạn: điểm xuất phát, MODI hoặc LP.

**Bảng chi phí - phân bổ**
- Mỗi ô hiển thị chi phí và lượng phân bổ hiện tại.
- Với kết quả MODI, xem tab từng bước để có thêm Δ và chu trình.

**Network graph**
- Hình tròn trái = nguồn, hình tròn phải = đích.
- Đường nối = tuyến có phân bổ > 0.
- Độ đậm của đường = lượng phân bổ.
- Số trên đường = (lượng phân bổ / chi phí).
        """)

    # ── Section 7: Export ─────────────────────────────────────────────────────
    with st.expander("7️⃣  Export — tab 💾"):
        st.markdown("""
Tab **💾 Export** cho phép tải về kết quả dưới nhiều định dạng:

| Nút | Nội dung |
|---|---|
| 🖼 Heatmap chi phí (PNG) | Biểu đồ màu của ma trận chi phí |
| 🖼 Heatmap phân bổ (PNG) | Biểu đồ màu phân bổ hàng hoá |
| 🖼 Network graph (PNG) | Sơ đồ mạng lưới vận tải |
| 📄 Phân bổ (CSV) | Bảng phân bổ xuất file CSV |
| 📄 Kết quả (JSON) | Toàn bộ kết quả + từng bước dưới dạng JSON |

> Chọn thuật toán ở dropdown trước khi tải để xuất đúng kết quả mong muốn.
        """)

    # ── Section 8: Tips & edge cases ─────────────────────────────────────────
    with st.expander("8️⃣  Lưu ý & các trường hợp đặc biệt"):
        st.markdown("""
**Bài toán không cân bằng (unbalanced)**
- Tổng cung ≠ tổng cầu → ứng dụng tự thêm nguồn/đích ảo với chi phí = 0.
- Phần dư của nguồn/đích ảo trong kết quả = lượng hàng tồn kho hoặc thiếu hụt.

**Ô cấm (forbidden cells)**
- Ô cấm được gán chi phí rất lớn (M = 10⁷) trong quá trình tính toán.
- Hiển thị là ∞ trong bảng, màu đỏ trong heatmap.

**Bài toán tối đa hoá (max)**
- Hệ thống chuyển sang min bằng cách đổi dấu: cᵢⱼ → max(C) − cᵢⱼ.
- Kết quả hiển thị là lợi nhuận tối đa (đã đổi lại dấu).

**MODI giới hạn kích thước ≤ 6×6**
- Với bài toán lớn hơn, dùng **LP Solver** để có nghiệm tối ưu.
- NW/LC/VAM vẫn chạy bình thường với bài toán lớn nhưng không có MODI.

**Bài toán phân công (Assignment)**
- Dành riêng cho ma trận vuông n×n, giải bằng thuật toán Hungarian.
- Chọn ví dụ `assignment_4x4` để thử.

**Bấm ↺ Đặt lại**
- Xoá toàn bộ kết quả và trạng thái session, giữ nguyên chế độ nhập và loại bài toán.
        """)

    st.divider()
    st.caption("Transportation Algorithm Visualizer · Python · Streamlit · SciPy · NetworkX")
