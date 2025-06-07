# ☕ Coffee Shop Customer Type Prediction App

Aplikasi ini dibuat sebagai bagian dari proyek machine learning oleh **Kelompok 1**, bertujuan untuk mengklasifikasikan tipe pelanggan coffee shop berdasarkan perilaku kunjungan, pengeluaran, waktu kunjungan, serta status membership.

---

## 🔍 Deskripsi Singkat

Aplikasi ini menggunakan model **Random Forest** untuk memprediksi tipe pelanggan menjadi salah satu dari tiga kategori:

* 🏆 **Royal**
* 💤 **Inactive**
* 🌱 **New**

Prediksi ini didasarkan pada data seperti frekuensi kunjungan, pengeluaran, durasi di kafe, jenis kelamin, dan status membership.

---

## 🚀 Fitur Utama

* 🔁 **Prediksi Individu**: Input data pelanggan satu per satu melalui form.
* 📂 **Prediksi Massal**: Unggah file CSV untuk memproses banyak data sekaligus.
* 📈 **Skor Loyalitas**: Memberikan penilaian skor dari 0–100 berdasarkan kebiasaan pelanggan.
* 📊 **Visualisasi Probabilitas**: Menampilkan grafik probabilitas tiap kategori.
* 💡 **Rekomendasi**: Menyediakan tips berdasarkan hasil klasifikasi.
* 🖼️ **Avatar Pelanggan**: Tampilan visual untuk tiap tipe pelanggan.
* 🌓 **Mode Gelap**: Tampilan modern dan nyaman di mata.
* 🌐 **Dukungan Bahasa**: Indonesia 🇮🇩 dan Inggris 🇬🇧.

---

## 🧠 Teknologi yang Digunakan

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn (RandomForestClassifier)
* Pickle (untuk menyimpan model dan encoder)

---

## 📁 Struktur File

```
├── app.py                      # Source code utama Streamlit
├── model.pkl                  # Model Random Forest yang telah dilatih
├── label_encoders.pkl        # Encoder Label untuk Gender dan Membership
├── image/
│   ├── kiki.jpg               # Avatar Royal
│   ├── Amu.jpg                # Avatar Inactive
│   └── upi.jpg                # Avatar New
├── hasil_prediksi.csv        # Contoh hasil prediksi massal (opsional)
└── README.md
```

---

## 🛠️ Cara Menjalankan

1. Clone repositori ini:

   ```bash
   git clone https://github.com/namakamu/prediksi-tipe-pelanggan.git
   cd prediksi-tipe-pelanggan
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

---

## 📌 Catatan

* Pastikan file `model.pkl` dan `label_encoders.pkl` tersedia di direktori yang sama.
* Format CSV untuk prediksi massal harus memuat kolom: `Age`, `Gender`, `Visit_Frequency`, `Spending_per_Visit`, `Time_Spent_in_Cafe`, `Membership_Status`.

---

## 👨‍💻 Kontributor

**Kelompok 1 - Proyek Machine Learning**
Anggota:

* Ranggis
* Fatima
* Egi
* Syahid
* Bila

---

## 📜 Lisensi

Proyek ini dibuat untuk tujuan edukasi. Silakan gunakan, modifikasi, dan distribusikan dengan mencantumkan atribusi yang sesuai.

---

Terima kasih telah menggunakan aplikasi kami! ☕✨
