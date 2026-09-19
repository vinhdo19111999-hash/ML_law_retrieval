"""
src/evaluation_ext.py — ĐÁNH GIÁ MỞ RỘNG (bổ sung cho src/evaluation.py)

Vì sao cần file này
-------------------
Đề tài yêu cầu: các chỉ số đánh giá (precision, recall, F1), đánh giá câu hỏi
PHÙ HỢP / KHÔNG PHÙ HỢP với dữ liệu, lựa chọn ngưỡng quyết định và phân tích lỗi.
`src/evaluation.py` hiện chỉ có Hit@1/3/5 + MRR trên 60 câu hỏi TRONG MIỀN, nên
không thể trả lời được các mục đó.

File này bổ sung 3 thứ, KHÔNG sửa gì của code cũ:

1) Bộ chỉ số đầy đủ trên câu hỏi trong miền:
      Hit@k, MRR, Precision@k, Recall@k, NDCG@k
   (định nghĩa "liên quan" = mọi chunk thuộc đúng Điều/Khoản/Điểm trong đáp án)

2) Đánh giá khả năng TỪ CHỐI câu hỏi ngoài miền / rác:
      bài toán phân loại nhị phân (trả lời vs từ chối)
      -> Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC
   
3) Quét ngưỡng cosine similarity để CHỌN NGƯỠNG có căn cứ thực nghiệm
   (thay cho việc đặt cố định 0.78 / 0.70).

CÁCH CHẠY (sau khi đã cài requirements; cần model E5 nên chạy trên Colab/GPU)
-----------------------------------------------------------------------------
    python src/evaluation_ext.py \
        --test-set data/test_questions.csv \
        --ood      data/out_of_domain_questions.csv \
        --ra       reports/hinh-anh

Máy yếu / muốn chạy nhanh:
    MODEL_E5=intfloat/multilingual-e5-base python src/evaluation_ext.py --ra reports/hinh-anh

    (2) DÁN VÀO 1 Ô CELL COLAB:
        main([])                      # chạy như mặc định, ghi vào reports/hinh-anh/
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

# ============================================================================
# TƯƠNG THÍCH CẢ 2 CÁCH CHẠY:
#   (1) chạy bằng dòng lệnh :  python src/evaluation_ext.py [tuỳ chọn]
#   (2) DÁN VÀO 1 Ô CELL Colab:  main([])   hoặc   main(["--bo-qua-cum"])
# ============================================================================
def _trong_notebook() -> bool:
    """True nếu file này đang được DÁN vào ô cell Jupyter/Colab (không phải chạy bằng python)."""
    try:
        get_ipython()          # noqa: F821 — biến này CHỈ tồn tại trong IPython/Jupyter/Colab
        return True
    except NameError:
        return False


def _tim_goc_repo() -> str:
    """Tìm thư mục gốc repo — chạy được cả khi là script lẫn khi dán vào ô cell.

    Trong ô cell notebook KHÔNG có biến __file__ (đó là lý do lỗi
    "NameError: name '__file__' is not defined"), nên phải dò thư mục từ chỗ khác.
    """
    ung_vien = []
    try:                                              # 1) chạy như script bình thường
        ung_vien.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass
    ung_vien.append(os.environ.get("GOC_REPO", ""))
    # 2) Colab: thư mục clone nằm trong /content (tên thư mục = tên repo: ML_law_retrieval, ...)
    #    -> liệt kê con của /content VÀ của thư mục hiện tại, KHÔNG hard-code tên repo.
    p = os.getcwd()                                   # dò ngược lên từ thư mục hiện tại
    for _ in range(4):
        ung_vien.append(p)
        p = os.path.dirname(p)
    for goc in ("/content", os.getcwd(), "/workspace"):
        try:
            for ten in sorted(os.listdir(goc)):
                ung_vien.append(os.path.join(goc, ten))
        except Exception:
            pass
    for u in ung_vien:                     # ưu tiên thư mục có dấu hiệu CHẮC CHẮN là repo này
        if u and (os.path.exists(os.path.join(u, "src", "retrieval.py"))
                  or os.path.isdir(os.path.join(u, ".git"))):
            return os.path.abspath(u)
    for u in ung_vien:                     # nới lỏng: chỉ cần có data/ + src/
        if u and os.path.isdir(os.path.join(u, "data")) and os.path.isdir(os.path.join(u, "src")):
            return os.path.abspath(u)
    return os.path.abspath(os.getcwd())


GOC_REPO = _tim_goc_repo()

def _chuan_bi_import(goc_repo: str | None = None) -> None:
    """Thêm gốc repo + thư mục src/ vào sys.path để import được trong MỌI cách chạy:
    `python src/xxx.py` · `python -m src.xxx` · DÁN vào ô cell Colab.
    """
    import sys
    goc = goc_repo or GOC_REPO
    for d in (goc, os.path.join(goc, "src")):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def _hien_thi_anh(*duong_dan) -> None:
    """Hiện ảnh NGAY TRONG OUTPUT khi chạy trong ô cell Colab/Jupyter;
    khi chạy bằng `python ...` (không hiển thị được ảnh) thì in đường dẫn để mở file."""
    if _trong_notebook():
        try:
            from IPython.display import Image, display
            for d in duong_dan:
                if os.path.exists(d):
                    print(f"🖼️  {os.path.basename(d)}")
                    display(Image(filename=d))
        except Exception as e:                      # thiếu IPython -> chỉ in đường dẫn
            for d in duong_dan:
                print(f"🖼️  (mở file để xem) {d}   [{e}]")
    else:
        for d in duong_dan:
            print(f"🖼️  (chạy `!python scripts/xem_ket_qua.py` để xem trong Colab) {d}")


DUONG_DAN_TEST_MAC_DINH = os.path.join(GOC_REPO, "data", "test_questions.csv")
DUONG_DAN_OOD_MAC_DINH = os.path.join(GOC_REPO, "data", "out_of_domain_questions.csv")
DUONG_DAN_CHUNKS_MAC_DINH = os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv")
THU_MUC_RA_MAC_DINH = os.path.join(GOC_REPO, "reports", "hinh-anh")

CAC_CHI_SO = ["Hit@k", "MRR", "Precision@k", "Recall@k", "NDCG@k"]


# ============================================================================
# PHẦN 0 — TIỆN ÍCH
# ============================================================================
def _so(v):
    """Chuyển ô CSV ('3', '3.0', NaN) về số nguyên hoặc None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _chuoi(v):
    """Chuẩn hoá ô ký hiệu Điểm: NaN/'nan'/'' -> None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None") else s


def tap_lien_quan(df_chunks: pd.DataFrame, dieu_id, khoan_id, diem_id) -> set:
    """Trả về tập `full_citation` được coi là ĐÁP ÁN ĐÚNG của một câu hỏi.

    Quy ước:
      - có diem_id   -> đúng 1 chunk (Điều/Khoản/Điểm)
      - có khoan_id  -> mọi chunk của (Điều, Khoản) gồm cả các Điểm con
      - chỉ có dieu_id -> mọi chunk của Điều đó
    """
    d = df_chunks
    mask = pd.Series(True, index=d.index)
    if dieu_id is not None:
        mask &= d["dieu_id"] == dieu_id
    if khoan_id is not None:
        mask &= d["khoan_id"] == khoan_id
    if diem_id is not None:
        mask &= d["diem_id"] == diem_id
    return set(d.loc[mask, "full_citation"].astype(str))


def ndcg_at_k(hang_lien_quan: list, k: int) -> float:
    """NDCG@k với gain nhị phân (liên quan = 1)."""
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(hang_lien_quan[:k]) if r)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len([x for x in hang_lien_quan if x]), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# PHẦN 1 — BỘ CHỈ SỐ ĐẦY ĐỦ TRÊN CÂU HỎI TRONG MIỀN
# ============================================================================
def danh_gia_trong_mien(ham_truy_van, df_test, df_chunks, top_k=5) -> pd.DataFrame:
    """Chấm 1 hàm tra cứu (trả list dict có 'full_citation') trên bộ câu hỏi có đáp án."""
    rows = []
    for r in df_test.itertuples():
        if getattr(r, "loai_truy_van", "ngu_nghia") != "ngu_nghia":
            continue
        lien_quan = tap_lien_quan(df_chunks, _so(getattr(r, "dieu_id", None)),
                                  _so(getattr(r, "khoan_id", None)),
                                  _chuoi(getattr(r, "diem_id", None)))
        cau_hoi = str(r.cau_hoi)
        if cau_hoi.strip() and str(r.cau_hoi) != "nan":
            try:
                ds = ham_truy_van(cau_hoi, top_k=top_k, nguong=0.0)
            except TypeError:                      # hàm tra cứu không có tham số nguong
                ds = ham_truy_van(cau_hoi, top_k=top_k)
            citations = [x["full_citation"] for x in ds]
        else:
            citations = []

        co_lien_quan = [c in lien_quan for c in citations]
        hang = next((i for i, ok in enumerate(co_lien_quan, 1) if ok), 0)   # 0 = không có trong top-k
        so_dung = sum(co_lien_quan)

        rows.append({
            "cau_hoi": cau_hoi,
            "dap_an_dung": " | ".join(sorted(lien_quan))[:120],
            "top1_citation": citations[0] if citations else "N/A",
            "hang_dung": hang,
            "so_chunk_lien_quan": len(lien_quan),
            "so_chunk_dung_trong_topk": so_dung,
            "Hit@k": 1 if hang else 0,
            "MRR": (1.0 / hang) if hang else 0.0,
            "Precision@k": so_dung / top_k if citations else 0.0,
            "Recall@k": so_dung / len(lien_quan) if lien_quan else 0.0,
            "NDCG@k": ndcg_at_k(co_lien_quan, top_k),
        })
    return pd.DataFrame(rows)


# ============================================================================
# PHẦN 2 + 3 — TRONG MIỀN vs NGOÀI MIỀN, QUÉT NGƯỠNG, CONFUSION MATRIX
# ============================================================================
def _diem_top1(ham_truy_van, cau_hoi) -> float:
    """Điểm tương đồng cao nhất mà hệ thống tìm được cho 1 câu hỏi."""
    try:
        ds = ham_truy_van(cau_hoi, top_k=1, nguong=0.0)
    except Exception:
        return 0.0
    if not ds:
        return 0.0
    r = ds[0]
    return float(r.get("diem_tuong_dong", r.get("diem", 0.0)))


def danh_gia_ngoai_mien(ham_truy_van, df_test, df_ood, la_rac_ro_rang=None,
                        thu_muc_ra=THU_MUC_RA_MAC_DINH, ve_bieu_do=True):
    """Coi 'nên trả lời' vs 'nên từ chối' là bài toán phân loại nhị phân.

    - Nhãn dương (1) = câu hỏi TRONG miền (phải trả lời được)
    - Nhãn âm   (0) = câu hỏi NGOÀI miền / rác (phải từ chối)
    Điểm quyết định = cosine similarity của top-1.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    ds_diem, ds_nhan, ds_loai, ds_cau = [], [], [], []
    for r in df_test.itertuples():
        if getattr(r, "loai_truy_van", "ngu_nghia") != "ngu_nghia":
            continue
        cau = str(r.cau_hoi)
        if not cau.strip() or cau == "nan":
            continue
        ds_diem.append(_diem_top1(ham_truy_van, cau))
        ds_nhan.append(1)
        ds_loai.append("trong_mien")
        ds_cau.append(cau)

    for r in df_ood.itertuples():
        cau = str(getattr(r, "cau_hoi", "") or "")
        loai = str(getattr(r, "loai", "ngoai_mien"))
        if not cau.strip() or cau == "nan":       # dòng rỗng -> câu hỏi rỗng, vẫn là rác
            cau = "(câu hỏi rỗng)"
        if la_rac_ro_rang is not None and la_rac_ro_rang(cau):
            diem = 0.0                            # bị chặn ngay từ tuyến 1 -> không có điểm
        else:
            diem = _diem_top1(ham_truy_van, cau)
        ds_diem.append(diem); ds_nhan.append(0); ds_loai.append(loai); ds_cau.append(cau)

    df = pd.DataFrame({"cau_hoi": ds_cau, "loai": ds_loai,
                       "nhan_dung": ds_nhan, "diem_top1": ds_diem})

    print("\n" + "=" * 78)
    print("PHÂN BỐ ĐIỂM TƯƠNG ĐỒNG TOP-1")
    print("=" * 78)
    print(df.groupby("nhan_dung")["diem_top1"].describe()[["count", "mean", "50%", "min", "max"]].to_string())
    auc = roc_auc_score(df["nhan_dung"], df["diem_top1"])
    print(f"\nROC-AUC (dùng điểm top-1 để phân biệt trong miền / ngoài miền): {auc:.4f}")

    # ---- quét ngưỡng ----
    nguong = np.round(np.arange(0.50, 0.991, 0.01), 2)
    ket_qua = []
    for t in nguong:
        quyet_dinh_tra_loi = (df["diem_top1"] >= t).astype(int)   # 1 = trả lời
        ket_qua.append({
            "nguong": t,
            "Accuracy": accuracy_score(df["nhan_dung"], quyet_dinh_tra_loi),
            "Precision": precision_score(df["nhan_dung"], quyet_dinh_tra_loi, zero_division=0),
            "Recall": recall_score(df["nhan_dung"], quyet_dinh_tra_loi, zero_division=0),
            "F1": f1_score(df["nhan_dung"], quyet_dinh_tra_loi, zero_division=0),
            "so_cau_bi_tu_choi_oan": int(((df["nhan_dung"] == 1) & (quyet_dinh_tra_loi == 0)).sum()),
            "so_cau_ngoai_mien_lot_vao": int(((df["nhan_dung"] == 0) & (quyet_dinh_tra_loi == 1)).sum()),
        })
    bang = pd.DataFrame(ket_qua)
    tot_nhat = bang.loc[bang["F1"].idxmax()]
    print("\n" + "=" * 78)
    print("QUÉT NGƯỠNG (ngưỡng càng cao càng dễ từ chối)")
    print("=" * 78)
    print(bang.set_index("nguong").round(3).to_string())
    print(f"\n👉 Ngưỡng F1 cao nhất: {tot_nhat['nguong']:.2f}  (F1 = {tot_nhat['F1']:.3f}, "
          f"Recall = {tot_nhat['Recall']:.3f}, Precision = {tot_nhat['Precision']:.3f})")
    print(f"👉 Ngưỡng đang hard-code trong retrieval.py (0.78): "
          f"F1 = {bang.loc[bang['nguong'] == 0.78, 'F1'].squeeze():.3f}")

    os.makedirs(thu_muc_ra, exist_ok=True)
    bang.to_csv(os.path.join(thu_muc_ra, "quet-nguong-ngoai-mien.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(thu_muc_ra, "diem-trung-mien-ngoai-mien.csv"), index=False, encoding="utf-8-sig")

    # ---- confusion matrix tại 2 ngưỡng: đang dùng vs tốt nhất ----
    for ten, t in [("nguong-dang-dung-0.78", 0.78), (f"nguong-toi-uu-{tot_nhat['nguong']:.2f}", float(tot_nhat["nguong"]))]:
        cm = confusion_matrix(df["nhan_dung"], (df["diem_top1"] >= t).astype(int), labels=[1, 0])
        print(f"\nConfusion matrix @ {ten}  (hàng = thực tế, cột = hệ thống)")
        print(pd.DataFrame(cm,
                           index=["Thực tế: trong miền", "Thực tế: ngoài miền"],
                           columns=["Hệ thống: trả lời", "Hệ thống: từ chối"]).to_string())
        if ve_bieu_do:
            _ve_confusion_matrix(cm, ten, thu_muc_ra)

    if ve_bieu_do:
        _ve_bieu_do_quet_nguong(bang, thu_muc_ra)
    return df, bang


def _ve_confusion_matrix(cm, ten_file, thu_muc_ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "#20303F")
    ax.set_xticks([0, 1], ["Hệ thống: trả lời", "Hệ thống: từ chối"])
    ax.set_yticks([0, 1], ["Thực tế: trong miền", "Thực tế: ngoài miền"])
    ax.set_title(f"Ma trận nhầm lẫn — {ten_file}", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    duong_dan = os.path.join(thu_muc_ra, f"confusion-matrix-{ten_file}.png")
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    print(f"   💾 {duong_dan}")
    _hien_thi_anh(duong_dan)


def _ve_bieu_do_quet_nguong(bang, thu_muc_ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(bang["nguong"], bang["Precision"], label="Precision", color="#2E5AAC", lw=2)
    ax.plot(bang["nguong"], bang["Recall"], label="Recall", color="#C1554C", lw=2)
    ax.plot(bang["nguong"], bang["F1"], label="F1", color="#4C9A52", lw=2.4)
    i = bang["F1"].idxmax()
    ax.axvline(bang.loc[i, "nguong"], ls="--", color="#E0A33B")
    ax.annotate(f"ngưỡng tốt nhất = {bang.loc[i, 'nguong']:.2f}\nF1 = {bang.loc[i, 'F1']:.3f}",
                xy=(bang.loc[i, "nguong"], bang.loc[i, "F1"]),
                xytext=(bang.loc[i, "nguong"] + 0.03, 0.55),
                arrowprops=dict(arrowstyle="->", color="#E0A33B"),
                fontsize=10, color="#7A5B14")
    ax.axvline(0.78, ls=":", color="#666666")
    ax.text(0.781, 0.08, "ngưỡng đang dùng 0.78", rotation=90, fontsize=9, color="#666666")
    ax.set_xlabel("Ngưỡng cosine similarity (từ chối nếu top-1 < ngưỡng)")
    ax.set_ylabel("Giá trị (0 → 1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Chọn ngưỡng từ chối câu hỏi ngoài miền theo thực nghiệm")
    ax.legend(loc="center right")
    fig.tight_layout()
    duong_dan = os.path.join(thu_muc_ra, "bieu-do-6-quet-nguong-tu-choi.png")
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    print(f"   💾 {duong_dan}")
    _hien_thi_anh(duong_dan)


# ============================================================================
# CLI
# ============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Đánh giá mở rộng: P/R/F1 + trong-ngoài miền + quét ngưỡng")
    p.add_argument("--test-set", default=DUONG_DAN_TEST_MAC_DINH)
    p.add_argument("--ood", default=DUONG_DAN_OOD_MAC_DINH,
                   help="CSV câu hỏi ngoài miền: cột câu hỏi ở tên 'cau_hoi' hoặc 'cau_hoi' + 'loai'")
    p.add_argument("--chunks", default=DUONG_DAN_CHUNKS_MAC_DINH)
    p.add_argument("--ra", default=THU_MUC_RA_MAC_DINH, help="thư mục ghi CSV + PNG")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--chi-tfidf", action="store_true", help="chỉ chạy baseline TF-IDF (không cần tải model E5)")
    a = p.parse_known_args(argv)[0]

    os.environ["DUONG_DAN_CHUNKS"] = a.chunks
    _chuan_bi_import()                                # để import được cả khi dán vào ô cell Colab
    try:
        from .retrieval import (bai_toan_2_tra_cuu_ngu_nghia, khoi_tao, la_rac_ro_rang,
                                nap_du_lieu)
        from .evaluation import _ham_tfidf_builder
    except ImportError:
        try:
            from src.retrieval import (bai_toan_2_tra_cuu_ngu_nghia, khoi_tao, la_rac_ro_rang,
                                       nap_du_lieu)
            from src.evaluation import _ham_tfidf_builder
        except ImportError:
            from retrieval import (bai_toan_2_tra_cuu_ngu_nghia, khoi_tao, la_rac_ro_rang,
                                   nap_du_lieu)
            from evaluation import _ham_tfidf_builder

    df_chunks, _ = nap_du_lieu()
    ham = _ham_tfidf_builder(df_chunks)
    ten_ham = "TF-IDF"
    if not a.chi_tfidf:
        khoi_tao(a.chunks)
        ham = bai_toan_2_tra_cuu_ngu_nghia
        ten_ham = "E5 embedding"

    df_test = pd.read_csv(a.test_set, encoding="utf-8-sig")
    if "loai_truy_van" not in df_test.columns:
        df_test["loai_truy_van"] = "ngu_nghia"

    print("\n" + "=" * 78)
    print(f"1) BỘ CHỈ SỐ ĐẦY ĐỦ TRÊN CÂU HỎI TRONG MIỀN — {ten_ham} (top_k = {a.top_k})")
    print("=" * 78)
    df_tm = danh_gia_trong_mien(ham, df_test, df_chunks, top_k=a.top_k)
    for c in CAC_CHI_SO:
        print(f"  {c:<12} {df_tm[c].mean():.4f}")
    os.makedirs(a.ra, exist_ok=True)
    df_tm.to_csv(os.path.join(a.ra, "chi-so-day-du-trong-mien.csv"), index=False, encoding="utf-8-sig")
    print(f"   💾 {os.path.join(a.ra, 'chi-so-day-du-trong-mien.csv')}")

    if os.path.exists(a.ood):
        print("\n" + "=" * 78)
        print(f"2) TRONG MIỀN vs NGOÀI MIỀN/RÁC — {ten_ham}")
        print("=" * 78)
        df_ood = pd.read_csv(a.ood, encoding="utf-8-sig")
        col_cau = "cau_hoi" if "cau_hoi" in df_ood.columns else df_ood.columns[1]
        df_ood = df_ood.rename(columns={col_cau: "cau_hoi"})
        if "loai" not in df_ood.columns:
            df_ood["loai"] = "ngoai_mien"
        danh_gia_ngoai_mien(ham, df_test, df_ood, la_rac_ro_rang=la_rac_ro_rang,
                            thu_muc_ra=a.ra)
    else:
        print(f"\n⚠️  Không thấy {a.ood} -> bỏ qua phần đánh giá ngoài miền.")

    print("\n" + "-" * 78)
    print("GHI VÀO BÁO CÁO:")
    print("  • Mục 'chỉ số đánh giá': lấy Hit@k / MRR / Precision@k / Recall@k / NDCG@k.")
    print("  • Mục 'câu hỏi phù hợp hay không phù hợp': lấy bảng quét ngưỡng + ROC-AUC + confusion matrix.")
    print("  • Nêu rõ ngưỡng chọn được và vì sao chọn nó (đánh đổi Precision ↔ Recall).")
    print("-" * 78)


if __name__ == "__main__":
    if _trong_notebook():
        print("ℹ️  File này đang chạy trong Ô CELL notebook -> bỏ qua phần dòng lệnh.")
        print("    (Trong notebook, argparse sẽ đọc nhầm tham số của Colab và báo lỗi")
        print("     'unrecognized arguments: -f /root/.local/share/jupyter/runtime/kernel-....json')")
        print("    Hãy gọi hàm trực tiếp, ví dụ:")
        print("        main([])          # chạy như mặc định")
        print("        main(['--tuy-chon'])  # tuỳ chọn: xem danh sách ở phần CÁCH CHẠY đầu file")
    else:
        main()

