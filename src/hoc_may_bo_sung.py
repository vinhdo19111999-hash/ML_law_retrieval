"""
src/hoc_may_bo_sung.py — THÀNH PHẦN HỌC MÁY BỔ SUNG (Clustering + Phân loại + So sánh mô hình)

Vì sao cần file này
-------------------
Đề tài yêu cầu áp dụng thuật toán học máy (phân cụm / phân loại / hồi quy) và
đánh giá bằng precision, recall, F1, so sánh nhiều mô hình. Repo hiện tại chỉ có
TÌM KIẾM THÔNG TIN (TF-IDF + embedding) và gán nhãn bằng LUẬT TAY, chưa có phần
học máy đúng nghĩa. File này bổ sung 2 nhánh, dùng lại đúng dữ liệu 606 chunks:

A) HỌC KHÔNG GIÁM SÁT — KMeans trên embedding E5 của 606 chunks
   • chọn số cụm k bằng silhouette score (quét k = 2..12)
   • trực quan hoá 2D (t-SNE/PCA) tô màu theo cụm
   • đối chiếu cụm tìm được với Chương (nhãn có sẵn) và với nhãn luật tay
     dang_quy_dinh -> purity / Adjusted Rand Index

B) HỌC CÓ GIÁM SÁT — phân loại `dang_quy_dinh` (quyen / nghia_vu / dieu_kien /
   thu_tuc / thoi_han / khac) từ nội dung chunk
   • đặc trưng: TF-IDF word(1-2) + char_wb(3-5)  (tùy chọn ghép embedding E5)
   • mô hình so sánh: LogisticRegression vs LinearSVC vs MultinomialNB
   • đánh giá: precision / recall / F1 (macro + weighted), confusion matrix,
     cross-validation 5-fold, ROC-AUC one-vs-rest (LogReg)
   • xuất bảng so sánh mô hình -> dùng trực tiếp cho mục "So sánh mô hình" của báo cáo

CÁCH CHẠY
---------
    # chỉ cần CPU, chạy rất nhanh (phần B); phần A cần embedding E5 -> Colab/GPU
    python src/hoc_may_bo_sung.py --chunks data/processed/law_dataset_chunks.csv \
                                  --ra reports/hinh-anh

    # bỏ phần A (không cần torch):
    python src/hoc_may_bo_sung.py --bo-qua-cum

    (2) DÁN VÀO 1 Ô CELL COLAB (không cần dòng lệnh):
        main([])                      # chạy như mặc định
        main(["--bo-qua-cum"])        # chạy nhanh, không cần torch
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ============================================================================
# TƯƠNG THÍCH CẢ 2 CÁCH CHẠY:
#   (1) chạy bằng dòng lệnh :  python src/hoc_may_bo_sung.py [tuỳ chọn]
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


CHUNKS_MAC_DINH = os.path.join(GOC_REPO, "data", "processed", "law_dataset_chunks.csv")
THU_MUC_RA_MAC_DINH = os.path.join(GOC_REPO, "reports", "hinh-anh")

# Các nhãn quá ít mẫu sẽ được gộp vào 'khac' để CV không vỡ
NGUONG_GOP_NHAN = 8


# ============================================================================
# A) HỌC KHÔNG GIÁM SÁT — KMeans + silhouette + t-SNE
# ============================================================================
def phan_cum(embeddings: np.ndarray, df_chunks: pd.DataFrame, thu_muc_ra: str,
             k_min=2, k_max=12, ve_hinh=True):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                                 davies_bouldin_score, silhouette_score)

    print("=" * 78)
    print("A) HỌC KHÔNG GIÁM SÁT — PHÂN CỤM 606 CHUNK LUẬT BẰNG KMEANS")
    print("=" * 78)

    X = np.asarray(embeddings, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # chuẩn hoá L2 như khi tính cosine

    ket_qua = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        nhan = km.fit_predict(X)
        ket_qua.append({
            "k": k,
            "silhouette": silhouette_score(X, nhan, metric="cosine"),
            "davies_bouldin": davies_bouldin_score(X, nhan),
            "calinski_harabasz": calinski_harabasz_score(X, nhan),
            "inertia": km.inertia_,
        })
    bang_k = pd.DataFrame(ket_qua).set_index("k").round(4)
    print(bang_k.to_string())
    k_tot = int(bang_k["silhouette"].idxmax())
    print(f"\n👉 k tốt nhất theo silhouette = {k_tot} (silhouette = {bang_k.loc[k_tot, 'silhouette']:.4f})")

    km = KMeans(n_clusters=k_tot, n_init=10, random_state=42)
    nhan = km.fit_predict(X)
    df = df_chunks.copy()
    df["cum"] = nhan

    # --- đối chiếu cụm với các nhãn CÓ SẴN (đánh giá ngoài, không giám sát) ---
    cot_chuong = "chuong_id" if "chuong_id" in df.columns else "chuong_ten"   # tên Chương quá dài khi in
    if cot_chuong in df.columns:
        ct = pd.crosstab(df["cum"], df[cot_chuong])
        purity = ct.max(axis=1).sum() / len(df)
        ari = adjusted_rand_score(df["chuong_ten"].astype(str), nhan)
        print(f"\nĐối chiếu cụm ↔ Chương: purity = {purity:.3f} | ARI = {ari:.3f}")
        print(ct.to_string())
        ct.to_csv(os.path.join(thu_muc_ra, "cum-theo-chuong.csv"), encoding="utf-8-sig")

    if "dang_quy_dinh" in df.columns:
        ct2 = pd.crosstab(df["cum"], df["dang_quy_dinh"])
        purity2 = ct2.max(axis=1).sum() / len(df)
        ari2 = adjusted_rand_score(df["dang_quy_dinh"].astype(str), nhan)
        print(f"\nĐối chiếu cụm ↔ nhãn luật tay (dang_quy_dinh): purity = {purity2:.3f} | ARI = {ari2:.3f}")
        print(ct2.to_string())
        ct2.to_csv(os.path.join(thu_muc_ra, "cum-theo-dang-quy-dinh.csv"), encoding="utf-8-sig")

    # --- 3 chunk gần tâm cụm nhất, để đặt tên cụm bằng mắt ---
    print("\nTên gợi ý cho từng cụm (3 chunk gần tâm nhất):")
    for c in range(k_tot):
        idx = np.where(nhan == c)[0]
        tam = km.cluster_centers_[c]
        gan = idx[np.argsort(-(X[idx] @ tam))][:3]
        print(f"\n  Cụm {c} (n={len(idx)}):")
        for i in gan:
            print(f"    - {df.iloc[i]['full_citation']}: {str(df.iloc[i]['noi_dung'])[:90]}...")

    df[["id", "full_citation", "cap_do", "cum"]].to_csv(
        os.path.join(thu_muc_ra, "cum-cua-tung-chunk.csv"), index=False, encoding="utf-8-sig")

    if ve_hinh:
        _ve_cum(X, df, k_tot, thu_muc_ra)
    return k_tot, bang_k, df


def _ve_cum(X, df, k_tot, thu_muc_ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    try:                                    # t-SNE đẹp hơn nhưng chậm hơn
        toa_do = TSNE(n_components=2, perplexity=30, init="pca",
                      learning_rate="auto", random_state=42).fit_transform(X)
        ten_phuong_phap = "t-SNE"
    except Exception as e:                  # máy yếu / lỗi -> dùng PCA
        print(f"   (t-SNE lỗi: {e} -> dùng PCA)")
        toa_do = PCA(n_components=2, random_state=42).fit_transform(X)
        ten_phuong_phap = "PCA"

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    cmap_cum = plt.get_cmap("tab10", k_tot)
    sc = axes[0].scatter(toa_do[:, 0], toa_do[:, 1], c=df["cum"], cmap=cmap_cum, s=16,
                         vmin=-0.5, vmax=k_tot - 0.5)
    axes[0].set_title(f"(a) {k_tot} cụm do KMeans tìm ra ({ten_phuong_phap})")
    cb = fig.colorbar(sc, ax=axes[0], ticks=range(k_tot), label="cụm")
    cb.ax.set_yticklabels([f"Cụm {i}" for i in range(k_tot)])

    if "chuong_ten" in df.columns:
        # rút gọn tên Chương cho vừa chú thích
        def _ngan(t):
            t = str(t)
            return t if len(t) <= 28 else t[:26].rstrip() + "…"

        cac_chuong = sorted(df["chuong_ten"].astype(str).unique())
        mau_chuong = plt.get_cmap("tab10", max(len(cac_chuong), 3))
        for i, t in enumerate(cac_chuong):
            mask = df["chuong_ten"].astype(str) == t
            axes[1].scatter(toa_do[mask, 0], toa_do[mask, 1], s=16, color=mau_chuong(i),
                            label=f"Chương {df.loc[mask, 'chuong_id'].iloc[0]}: {_ngan(t)}")
        axes[1].set_title("(b) Nhãn Chương (cấu trúc luật có sẵn)")
        axes[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.suptitle("Cấu trúc ngữ nghĩa của 606 chunk Luật ATVSLĐ 2015", fontsize=13)
    fig.tight_layout()
    duong_dan = os.path.join(thu_muc_ra, "bieu-do-7-phan-cum-kmeans.png")
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    print(f"   💾 {duong_dan}")
    _hien_thi_anh(duong_dan)


# ============================================================================
# B) HỌC CÓ GIÁM SÁT — phân loại dạng quy định, so sánh 3 mô hình
# ============================================================================
def _gop_nhan_nho(y: pd.Series, nguong=NGUONG_GOP_NHAN) -> pd.Series:
    dem = y.value_counts()
    return y.where(y.map(dem) >= nguong, "khac")


def phan_loai(df_chunks: pd.DataFrame, thu_muc_ra: str, dung_embedding=None, ve_hinh=True):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    print("\n" + "=" * 78)
    print("B) HỌC CÓ GIÁM SÁT — PHÂN LOẠI 'dang_quy_dinh' CỦA CHUNK")
    print("=" * 78)

    if "dang_quy_dinh" not in df_chunks.columns:
        print("!")
        print("!" * 78)
        print("⚠️  KHÔNG có cột 'dang_quy_dinh' trong file chunks -> phần PHÂN LOẠI bị bỏ qua.")
        print("    Nguyên nhân: `src/preprocess.py` GHI ĐÈ file chunks và làm mất 2 cột do")
        print("    `src/keyphrase.py` tạo ra (keyphrase, dang_quy_dinh).")
        print("    ➜ SỬA: chạy lệnh sau rồi chạy lại file này:")
        print("            !python src/keyphrase.py")
        print("      (hoặc chạy tất cả đúng thứ tự:  !python scripts/run_all.py)")
        print("!" * 78)
        return None

    X_text = df_chunks["noi_dung"].astype(str)
    if "van_ban_tim_kiem" in df_chunks.columns:                  # ngữ cảnh Khoản cha -> tín hiệu mạnh hơn
        X_text = df_chunks["van_ban_tim_kiem"].astype(str)
    y = _gop_nhan_nho(df_chunks["dang_quy_dinh"].astype(str))
    print(f"Số mẫu: {len(y)}   |   Phân bố nhãn:\n{y.value_counts().to_string()}")

    cac_mo_hinh = {
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.5), cv=3)),
        ]),
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", MultinomialNB(alpha=0.3)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bang, du_doan = [], {}
    for ten, mo_hinh in cac_mo_hinh.items():
        y_hat = cross_val_predict(mo_hinh, X_text, y, cv=cv)
        du_doan[ten] = y_hat
        bang.append({
            "mo_hinh": ten,
            "accuracy": accuracy_score(y, y_hat),
            "precision_macro": precision_score(y, y_hat, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_hat, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_hat, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_hat, average="weighted", zero_division=0),
            "so_nhan": y.nunique(),
        })
    bang = pd.DataFrame(bang).set_index("mo_hinh").round(4).sort_values("f1_macro", ascending=False)
    print("\n--- SO SÁNH MÔ HÌNH (cross-validation 5-fold, dự đoán out-of-fold) ---")
    print(bang.to_string())
    bang.to_csv(os.path.join(thu_muc_ra, "so-sanh-mo-hinh-phan-loai.csv"), encoding="utf-8-sig")

    tot_nhat = bang.index[0]
    print(f"\n--- BÁO CÁO CHI TIẾT CỦA MÔ HÌNH TỐT NHẤT: {tot_nhat} ---")
    bao_cao = classification_report(y, du_doan[tot_nhat], zero_division=0, digits=3)
    print(bao_cao)
    with open(os.path.join(thu_muc_ra, "classification-report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Mô hình tốt nhất: {tot_nhat}\n\n{bao_cao}\n\n{bang.to_string()}\n")

    # --- ROC-AUC macro (one-vs-rest) cho mô hình xác suất tốt nhất ---
    try:
        nhan_sap_xep = sorted(y.unique())                    # sklearn sắp nhãn theo alphabet
        xac_suat = cross_val_predict(cac_mo_hinh[tot_nhat], X_text, y, cv=cv, method="predict_proba")
        auc = roc_auc_score(pd.get_dummies(y)[nhan_sap_xep].to_numpy(), xac_suat,
                            multi_class="ovr", average="macro")
        print(f"ROC-AUC (macro, one-vs-rest) của {tot_nhat}: {auc:.4f}")
    except Exception as e:
        print(f"(bỏ qua ROC-AUC: {e})")

    if ve_hinh:
        nhan = sorted(y.unique())
        cm = confusion_matrix(y, du_doan[tot_nhat], labels=nhan, normalize="true")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(nhan)), nhan, rotation=40, ha="right")
        ax.set_yticks(range(len(nhan)), nhan)
        for i in range(len(nhan)):
            for j in range(len(nhan)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 else "#20303F")
        ax.set_xlabel("Mô hình dự đoán")
        ax.set_ylabel("Nhãn thật (luật tay)")
        ax.set_title(f"Ma trận nhầm lẫn (chuẩn hoá theo hàng) — {tot_nhat}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        duong_dan = os.path.join(thu_muc_ra, "bieu-do-8-confusion-matrix-phan-loai.png")
        fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
        print(f"   💾 {duong_dan}")

        # các từ khoá quan trọng nhất của lớp (giải thích mô hình)
        try:
            mo_hinh.fit(X_text, y)
            ten_dac_trung = mo_hinh.named_steps["tfidf"].get_feature_names_out()
            he_so = mo_hinh.named_steps["clf"].coef_
            print("\n--- 8 từ khoá đặc trưng nhất của mỗi lớp (LogReg/SVM) ---")
            for i, c in enumerate(mo_hinh.classes_):
                top = np.argsort(-he_so[i])[:8]
                print(f"  {c:<11}: " + ", ".join(ten_dac_trung[j] for j in top))
        except Exception as e:
            print(f"(bỏ qua phần từ khoá: {e})")

    return bang


def main(argv=None):
    p = argparse.ArgumentParser(description="Bổ sung phần học máy: phân cụm + phân loại + so sánh mô hình")
    p.add_argument("--chunks", default=CHUNKS_MAC_DINH)
    p.add_argument("--ra", default=THU_MUC_RA_MAC_DINH)
    p.add_argument("--bo-qua-cum", action="store_true", help="chỉ chạy phần phân loại (không cần embedding)")
    a = p.parse_known_args(argv)[0]

    os.environ["DUONG_DAN_CHUNKS"] = a.chunks
    os.makedirs(a.ra, exist_ok=True)
    df_chunks = pd.read_csv(a.chunks, encoding="utf-8-sig")

    if not a.bo_qua_cum:
        try:
            _chuan_bi_import()                            # để import được cả khi dán vào ô cell Colab
            try:
                from . import retrieval as R
            except ImportError:
                try:
                    from src import retrieval as R
                except ImportError:
                    import retrieval as R
            R.khoi_tao(a.chunks)                                 # nạp model E5 + embedding corpus
            phan_cum(R.corpus_embeds, df_chunks, a.ra)
        except Exception as e:
            print("\n" + "!" * 78)
            print("⚠️  KHÔNG chạy được phần A (phân cụm) -> sẽ KHÔNG có hình phân cụm + bảng silhouette.")
            print(f"    Lỗi: {type(e).__name__}: {e}")
            print("    ➜ SỬA (chọn 1):")
            print("        !pip install -q sentence-transformers        # thiếu thư viện / thiếu torch")
            print("        !python src/keyphrase.py                     # đảm bảo có cột keyphrase/dang_quy_dinh")
            print("        !MODEL_E5=intfloat/multilingual-e5-base python src/hoc_may_bo_sung.py   # máy yếu")
            print("    ➜ Chỉ cần phần PHÂN LOẠI (không cần hình phân cụm): thêm cờ --bo-qua-cum")
            print("!" * 78)

    phan_loai(df_chunks, a.ra)

    # ── TỔNG KẾT: liệt kê MỌI file đã xuất (để không bị 'chạy xong mà không thấy gì') ──
    CAN_XUAT = [
        ("bieu-do-7-phan-cum-kmeans.png", "hình phân cụm (t-SNE/PCA) — CHỈ có khi KHÔNG dùng --bo-qua-cum"),
        ("cum-theo-chuong.csv", "bảng đối chiếu cụm ↔ Chương (purity / ARI)"),
        ("cum-theo-dang-quy-dinh.csv", "bảng đối chiếu cụm ↔ nhãn luật tay"),
        ("cum-cua-tung-chunk.csv", "cụm của từng chunk (để tra cứu lại)"),
        ("so-sanh-mo-hinh-phan-loai.csv", "BẢNG SO SÁNH 3 MÔ HÌNH (accuracy / P / R / F1)"),
        ("classification-report.txt", "báo cáo chi tiết precision-recall-F1 từng lớp"),
        ("bieu-do-8-confusion-matrix-phan-loai.png", "hình confusion matrix + F1 từng lớp"),
    ]
    print("\n" + "=" * 78)
    print("📦 FILE ĐÃ XUẤT (thư mục:", os.path.abspath(a.ra) + ")")
    print("=" * 78)
    thieu = []
    for ten, mo_ta in CAN_XUAT:
        d = os.path.join(a.ra, ten)
        if os.path.exists(d):
            print(f"   ✅ {ten:<45} {os.path.getsize(d)/1024:>7.1f} KB   ({mo_ta})")
        else:
            print(f"   ⬜ {ten:<45} {'':>7}      ({mo_ta})")
            thieu.append(ten)
    if a.bo_qua_cum and "bieu-do-7-phan-cum-kmeans.png" in thieu:
        print("   ↳ muốn có hình phân cụm: chạy lại KHÔNG kèm --bo-qua-cum (cần tải model E5).")
    if "so-sanh-mo-hinh-phan-loai.csv" in thieu:
        print("   ↳ thiếu bảng phân loại: chạy `!python src/keyphrase.py` rồi chạy lại.")
    print("-" * 78)
    print("👀 CÁCH XEM HÌNH + SỐ:")
    print("   • Colab : chạy ô `!python scripts/xem_ket_qua.py`  -> hiện TẤT CẢ ảnh ngay trong output")
    print("   • Colab : bấm biểu tượng 📁 (thư mục) bên trái -> mở", a.ra)
    print("   • Máy bạn: mở thư mục", a.ra)
    print("-" * 78)
    print("GHI VÀO BÁO CÁO:")
    print("  • Mục 'thuật toán học máy': nêu KMeans + silhouette + ARI/purity (học không giám sát)")
    print("    và LogisticRegression / LinearSVC / MultinomialNB (học có giám sát).")
    print("  • Mục 'precision, recall, F1': lấy bảng so-sanh-mo-hinh-phan-loai.csv + classification-report.txt.")
    print("  • Mục 'ma trận nhầm lẫn / ROC-AUC': lấy bieu-do-8 + con số ROC-AUC macro in ở trên.")
    print("  • Mục 'lỗi phổ biến': các lớp bị nhầm nhiều nhất trong confusion matrix.")
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
