from __future__ import annotations


def nw_explanation() -> str:
    return (
        "**Phương pháp Góc Tây Bắc (NW):** Bắt đầu từ ô góc trên trái. "
        "Mỗi bước phân bổ min(cung[i], cầu[j]) vào ô hiện tại, "
        "rồi chuyển sang hàng mới (nếu hết cung) hoặc cột mới (nếu hết cầu)."
    )


def lcm_explanation() -> str:
    return (
        "**Phương pháp Chi phí Nhỏ nhất (LCM):** Mỗi bước chọn ô có chi phí nhỏ nhất còn khả dụng. "
        "Phân bổ nhiều nhất có thể vào ô đó, loại hàng/cột đã thoả mãn. "
        "Tie-breaking: tiềm năng phân bổ lớn hơn → chỉ số hàng nhỏ hơn → chỉ số cột nhỏ hơn."
    )


def vogel_explanation() -> str:
    return (
        "**Phương pháp Xấp xỉ Vogel (VAM):** Mỗi bước tính penalty = chênh lệch 2 chi phí nhỏ nhất "
        "cho mỗi hàng và cột còn hoạt động. Chọn hàng/cột có penalty lớn nhất, "
        "phân bổ vào ô rẻ nhất trong hàng/cột đó."
    )


def modi_explanation() -> str:
    return (
        "**Phương pháp MODI:** Từ phương án cơ sở, tính thế vị u/v "
        "sao cho u[i] + v[j] = c[i][j] với mọi ô cơ sở. "
        "Tính Δᵢⱼ = uᵢ + vⱼ − cᵢⱼ cho ô ngoài cơ sở. "
        "Nếu mọi Δ ≤ 0: tối ưu. Nếu có Δ > 0: đưa ô đó vào cơ sở, "
        "tìm chu trình, điều chỉnh θ = min phân bổ ô âm trong chu trình."
    )


def lp_explanation() -> str:
    return (
        "**LP Solver:** Giải bài toán quy hoạch tuyến tính với ràng buộc cung/cầu "
        "bằng HiGHS solver (scipy.optimize.linprog). Đảm bảo nghiệm tối ưu toàn cục."
    )
