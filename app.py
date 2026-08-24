import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

# ---------------------------------------------------------------------
# DATABASE KOEFISIEN PENGALIRAN (C KONSERVATIF/MAKSIMUM) DAN MANNING (n)
# ---------------------------------------------------------------------
DATABASE_C = {
    "Atap / Bangunan Kedap Air (0.75 - 0.95)": 0.95,
    "Jalan Aspal / Perkerasan Beton (0.70 - 0.95)": 0.95,
    "Jalan Kerikil / Tanah Keras (0.35 - 0.70)": 0.70,
    "Perumahan Padat / Kawasan Komersial (0.60 - 0.80)": 0.80,
    "Perumahan Sedang / Pemukiman (0.30 - 0.50)": 0.50,
    "Taman / Halaman / Rerumputan (0.10 - 0.25)": 0.25,
    "Hutan / Lahan Perkebunan (0.10 - 0.20)": 0.20
}

DATABASE_MANNING = {
    "Beton Halus / Smooth Concrete (n = 0.011 - 0.013)": (0.011, 0.013),
    "Beton Kasar / Rough Concrete (n = 0.014 - 0.017)": (0.014, 0.017),
    "Pasangan Batu / Masonry (n = 0.017 - 0.025)": (0.017, 0.025),
    "Saluran Tanah Bersih & Lurus (n = 0.018 - 0.022)": (0.018, 0.022),
    "Saluran Tanah Berrumput & Berbatu (n = 0.025 - 0.035)": (0.025, 0.035)
}

st.set_page_config(page_title="Analisa Drainase", layout="wide")

st.title("=== PROGRAM DRAINASE: ANALISA FREKUENSI & EVALUASI SALURAN ===")
st.markdown("Petunjuk: Silakan periksa/sesuaikan parameter input, lalu klik tombol untuk mengkalkulasi.")

# ---------------------------------------------------------------------
# INPUT KOSONG (SUDAH DIPERBAIKI)
# ---------------------------------------------------------------------
st.header("1. Data Hujan Harian Maksimum Tahunan (HHMT)")
col1, col2 = st.columns(2)
with col1:
    w_tahun = st.text_area("Kolom Tahun:", "", height=200)
with col2:
    w_hujan = st.text_area("Kolom Hujan (mm):", "", height=200)

st.header("2. Parameter Daerah Tangkapan Air (Catchment Area)")
w_c = st.selectbox("Penggunaan Lahan (C):", list(DATABASE_C.keys()))
w_area = st.text_input("Luas DTA / Catchment Area A (km²):", "")
w_tc = st.text_input("Waktu Konsentrasi tc (menit):", "")

st.header("3. Parameter & Dimensi Saluran Drainase")
w_shape = st.selectbox("Bentuk Saluran:", ['Persegi', 'Trapesium', 'Segitiga'])
w_b = st.text_input("Lebar Dasar Saluran b (m):", "")
w_h = st.text_input("Tinggi Muka Air h (m):", "")
w_m = st.text_input("Kemiringan Tebing m (1:m):", "")
w_material = st.selectbox("Bahan / Lapisan Saluran:", list(DATABASE_MANNING.keys()))
w_slope = st.text_input("Kemiringan Saluran S (m/m):", "")

def parse_float(val):
    return float(str(val).replace(',', '.').strip())

if st.button("Hitung & Analisa Keamanan Saluran", type="primary"):
    try:
        th_lines = [int(x.strip()) for x in w_tahun.strip().split('\n') if x.strip()]
        hj_lines = [float(x.strip().replace(',', '.')) for x in w_hujan.strip().split('\n') if x.strip()]
        
        A_val = parse_float(w_area)
        tc_menit = parse_float(w_tc)
        b = parse_float(w_b)
        h = parse_float(w_h)
        m_side = parse_float(w_m)
        S_val = parse_float(w_slope)
    except Exception:
        st.error("Pastikan seluruh angka pada kolom input diisi dengan benar dan tidak dibiarkan kosong!")
        st.stop()

    if len(th_lines) != len(hj_lines):
        st.error("Jumlah data Tahun dan Hujan tidak sama!")
        st.stop()

    df = pd.DataFrame({'Tahun': th_lines, 'Hujan': hj_lines}).sort_values(by='Hujan').reset_index(drop=True)
    n = len(df)

    if n < 3:
        st.error("Minimal butuh 3 baris data hujan!")
        st.stop()

    # --- A. ANALISA FREKUENSI ---
    i_arr = np.arange(1, n + 1)
    y_i = -np.log(-np.log(i_arr / (n + 1)))
    Yn, Sn = np.mean(y_i), np.std(y_i, ddof=0)

    mean, std_dev, skew = df['Hujan'].mean(), df['Hujan'].std(), df['Hujan'].skew()
    df['Log_X'] = np.log10(df['Hujan'])
    log_mean, log_std, log_skew = df['Log_X'].mean(), df['Log_X'].std(), df['Log_X'].skew()

    T_list = [2, 5, 10, 20, 50, 100, 1000]
    alpha_g = std_dev / Sn
    u_g = mean - (Yn * alpha_g)

    def factor_k(skew_val, T):
        if abs(skew_val) < 0.001: return stats.norm.ppf(1 - 1/T)
        p = 1 / T
        z = stats.norm.ppf(1 - p)
        return (2 / skew_val) * (((z - skew_val / 6) * (skew_val / 6) + 1)**3 - 1)

    dict_rencana = {
        'Normal': [mean + stats.norm.ppf(1 - (1/t)) * std_dev for t in T_list],
        'Log Normal': [10**(log_mean + stats.norm.ppf(1 - (1/t)) * log_std) for t in T_list],
        'Gumbel': [u_g + (alpha_g * (-np.log(-np.log(1 - 1/t)))) for t in T_list],
        'Pearson III': [mean + factor_k(skew, t) * std_dev for t in T_list],
        'Log Pearson III': [10**(log_mean + factor_k(log_skew, t) * log_std) for t in T_list]
    }

    def uji_ks(data, cdf_func, args):
        p_empiris = (np.arange(1, len(data) + 1)) / (len(data) + 1)
        return np.max(np.abs(p_empiris - cdf_func(data, *args))) * 100

    results_ks = {
        'Normal': uji_ks(df['Hujan'], stats.norm.cdf, (mean, std_dev)),
        'Log Normal': uji_ks(df['Log_X'], stats.norm.cdf, (log_mean, log_std)),
        'Gumbel': uji_ks(df['Hujan'], stats.gumbel_r.cdf, (u_g, alpha_g)),
        'Pearson III': uji_ks(df['Hujan'], stats.pearson3.cdf, (skew, mean, std_dev)),
        'Log Pearson III': uji_ks(df['Log_X'], stats.pearson3.cdf, (log_skew, log_mean, log_std))
    }

    critical_ks_pct = (1.36 / np.sqrt(n)) * 100
    best_metode = min(results_ks, key=results_ks.get)
    r_24_best = dict_rencana[best_metode]

    # --- B. HITUNG INTENSITAS HUJAN (MONONOBE) & Q RENCANA (RASIONAL) ---
    C_val = DATABASE_C[w_c]
    tc_jam = tc_menit / 60.0

    q_rencana_list = []
    i_list = []
    for r24 in r_24_best:
        I_mmhr = (r24 / 24.0) * ((24.0 / tc_jam) ** (2.0 / 3.0))
        i_list.append(I_mmhr)
        Q_rec = (C_val * I_mmhr * A_val) / 3.6
        q_rencana_list.append(Q_rec)

    # --- C. HITUNG Q SALURAN (MANNING) ---
    if w_shape == 'Persegi':
        A_sal = b * h
        P_sal = b + (2 * h)
    elif w_shape == 'Trapesium':
        A_sal = (b + (m_side * h)) * h
        P_sal = b + (2 * h * np.sqrt(1 + (m_side**2)))
    else: # Segitiga
        A_sal = m_side * (h**2)
        P_sal = 2 * h * np.sqrt(1 + (m_side**2))

    R_sal = A_sal / P_sal if P_sal > 0 else 0
    n_min, n_max = DATABASE_MANNING[w_material]

    v_min_n = (1.0 / n_max) * (R_sal ** (2.0/3.0)) * np.sqrt(S_val)
    v_max_n = (1.0 / n_min) * (R_sal ** (2.0/3.0)) * np.sqrt(S_val)

    Q_sal_max_n = v_min_n * A_sal  # n max (Kondisi Kasar)
    Q_sal_min_n = v_max_n * A_sal  # n min (Kondisi Licin)

    # --- D. TAMPILAN OUTPUT HTML ---
    html_code = """
    <style>
        .custom-table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 13px; margin-bottom: 20px; color: black; background-color: white;}
        .custom-table th { background-color: #f2f2f2; border: 1px solid #333; padding: 6px 10px; text-align: center; font-weight: bold; color: black;}
        .custom-table td { border: 1px solid #333; padding: 6px 10px; text-align: center; color: black;}
        .row-safe { background-color: #d4edda !important; }
        .row-unsafe { background-color: #f8d7da !important; }
        .status-safe { color: green; font-weight: bold; }
        .status-unsafe { color: red; font-weight: bold; }
        .info-box { background-color: #e9ecef; border-left: 4px solid #0d6efd; padding: 10px; font-family: Arial; font-size: 13px; margin-bottom: 15px; color: black;}
    </style>
    """

    html_code += f"""
    <div class='info-box'>
        <b>1. PARAMETER STATISTIK DATA (n = {n} Tahun):</b><br>
        • <b>Yn:</b> {Yn:.4f} | <b>Sn:</b> {Sn:.4f} | <b>Rata-Rata (X):</b> {mean:.2f} mm | <b>Std Deviasi (S):</b> {std_dev:.2f} | <b>Cs:</b> {skew:.4f}<br>
        • <b>Rata-Rata Log X:</b> {log_mean:.4f} | <b>Std Deviasi Log X:</b> {log_std:.4f} | <b>Cs Log X:</b> {log_skew:.4f}
    </div>
    """

    html_code += "<h4>1. Hasil Analisa Frekuensi Hujan Rencana (Semua Distribusi)</h4>"
    html_code += "<table class='custom-table'>"
    html_code += "<tr><th>Periode Ulang (T)</th><th>Probabilitas</th><th>Normal (mm)</th><th>Log Normal (mm)</th><th>Gumbel (mm)</th><th>Pearson III (mm)</th><th>Log Pearson III (mm)</th></tr>"
    
    prob_labels = ["50.0%", "20.0%", "10.0%", "5.0%", "2.0%", "1.0%", "0.1%"]
    for idx_t, t_val in enumerate(T_list):
        html_code += f"<tr>"
        html_code += f"<td><b>T = {t_val} Tahun</b></td><td>{prob_labels[idx_t]}</td>"
        html_code += f"<td>{dict_rencana['Normal'][idx_t]:.3f}</td>"
        html_code += f"<td>{dict_rencana['Log Normal'][idx_t]:.3f}</td>"
        html_code += f"<td>{dict_rencana['Gumbel'][idx_t]:.3f}</td>"
        html_code += f"<td>{dict_rencana['Pearson III'][idx_t]:.3f}</td>"
        html_code += f"<td>{dict_rencana['Log Pearson III'][idx_t]:.3f}</td>"
        html_code += f"</tr>"
    html_code += "</table>"

    html_code += f"<h4>2. Hasil Uji Kesesuaian Kolmogorov-Smirnov (Batas Kritis D<sub>tabel</sub> = {critical_ks_pct:.2f}%)</h4>"
    html_code += "<table class='custom-table'>"
    html_code += "<tr><th>Metode Distribusi</th><th>D<sub>max</sub> (%)</th><th>D<sub>tabel</sub> (α = 5%)</th><th>Status Kesesuaian</th></tr>"
    
    for met, dmax_val in results_ks.items():
        is_safe = dmax_val <= critical_ks_pct
        bg_class = "row-safe" if is_safe else "row-unsafe"
        st_text = "<span class='status-safe'>AMAN / DITERIMA</span>" if is_safe else "<span class='status-unsafe'>TIDAK AMAN / DITOLAK</span>"
        
        html_code += f"<tr class='{bg_class}'>"
        html_code += f"<td><b>{met}</b></td><td>{dmax_val:.2f}%</td><td>{critical_ks_pct:.2f}%</td><td>{st_text}</td>"
        html_code += "</tr>"
    html_code += "</table>"

    html_code += f"""
    <div class='info-box'>
        <b>METODE TERPILIH UNTUK DEBIT RENCANA:</b><br>
        • <b>Metode Terpilih:</b> <b>{best_metode}</b> (D<sub>max</sub> = {results_ks[best_metode]:.2f}% &le; {critical_ks_pct:.2f}%).<br>
        • <b>Parameter Rasional:</b> C = <b>{C_val}</b> | A = <b>{A_val} km²</b> | t<sub>c</sub> = {tc_menit} mnt (<b>{tc_jam:.4f} jam</b>).<br>
        • <b>Geometri Saluran ({w_shape}):</b> b = {b} m | h = {h} m &rarr; <b>A = {A_sal:.2f} m²</b> | <b>P = {P_sal:.2f} m</b> | <b>R = {R_sal:.6f} m</b>.<br>
        • <b>Kecepatan Aliran (v):</b> v<sub>(n={n_min})</sub> = <b>{v_max_n:.3f} m/s</b> | v<sub>(n={n_max})</sub> = <b>{v_min_n:.3f} m/s</b>.
    </div>
    """

    html_code += f"<h4>3. Tabel Hujan & Debit Rencana (Q<sub>rencana</sub> - {best_metode})</h4>"
    html_code += "<table class='custom-table'>"
    html_code += "<tr><th>Periode Ulang (T)</th><th>Probabilitas</th><th>R<sub>24</sub> Terpilih (mm)</th><th>Intensitas Hujan I (mm/jam)</th><th>Debit Rencana Q<sub>rencana</sub> (m³/s)</th></tr>"
    for idx_t, t_val in enumerate(T_list):
        html_code += f"<tr><td><b>T = {t_val} Tahun</b></td><td>{prob_labels[idx_t]}</td><td>{r_24_best[idx_t]:.3f}</td><td>{i_list[idx_t]:.3f}</td><td><b>{q_rencana_list[idx_t]:.3f} m³/s</b></td></tr>"
    html_code += "</table>"

    html_code += f"<h4>4. Evaluasi Keamanan Saluran (Q<sub>saluran</sub> vs Q<sub>rencana</sub>)</h4>"
    html_code += "<table class='custom-table'>"
    html_code += f"<tr><th rowspan='2'>Periode Ulang</th><th rowspan='2'>Q<sub>rencana</sub> (m³/s)</th><th colspan='2'>Kekasaran n = {n_min} (Kondisi Licin)<br><i>Q<sub>saluran</sub> = {Q_sal_min_n:.3f} m³/s ({Q_sal_min_n*1000:.1f} L/s)</i></th><th colspan='2'>Kekasaran n = {n_max} (Kondisi Kasar)<br><i>Q<sub>saluran</sub> = {Q_sal_max_n:.3f} m³/s ({Q_sal_max_n*1000:.1f} L/s)</i></th></tr>"
    html_code += "<tr><th>Q<sub>saluran</sub> (m³/s)</th><th>Status</th><th>Q<sub>saluran</sub> (m³/s)</th><th>Status</th></tr>"

    for idx_t, t_val in enumerate(T_list):
        q_rec = q_rencana_list[idx_t]
        
        safe_min_n = Q_sal_min_n >= q_rec
        safe_max_n = Q_sal_max_n >= q_rec

        st_min_n = "<span class='status-safe'>OK / AMAN</span>" if safe_min_n else "<span class='status-unsafe'>NOT OK / MELUAP</span>"
        st_max_n = "<span class='status-safe'>OK / AMAN</span>" if safe_max_n else "<span class='status-unsafe'>NOT OK / MELUAP</span>"

        bg_min_n = "row-safe" if safe_min_n else "row-unsafe"
        bg_max_n = "row-safe" if safe_max_n else "row-unsafe"

        html_code += f"<tr>"
        html_code += f"<td><b>T = {t_val} Thn</b></td><td><b>{q_rec:.3f}</b></td>"
        html_code += f"<td class='{bg_min_n}'>{Q_sal_min_n:.3f}</td><td class='{bg_min_n}'>{st_min_n}</td>"
        html_code += f"<td class='{bg_max_n}'>{Q_sal_max_n:.3f}</td><td class='{bg_max_n}'>{st_max_n}</td>"
        html_code += f"</tr>"

    html_code += "</table>"
    
    st.markdown(html_code, unsafe_allow_html=True)
