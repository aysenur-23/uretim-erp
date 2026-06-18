# Taş Yünü Kalite Kontrol

## MEGA Next.js Ana Arayuz

Yeni ana arayuz Next.js + FastAPI ile calisir. Streamlit prototip olarak kalmistir.

```powershell
.\start_mega_qc.ps1
```

- Frontend: http://localhost:3000
- Backend: http://127.0.0.1:8000

Backend mevcut OpenCV kalite kontrol akisindakileri kullanir: `process_frame`, ROI/bbox, catlak, renk, kenar ve yerel anomali kurallari. Frontend `/api/*` isteklerini Next.js proxy uzerinden FastAPI backend'e yollar.
Endüstriyel taş yünü paneller için görüntü tabanlı kalite kontrol uygulaması.

Uygulama iki giriş kaynağını aynı denetim hattından geçirir:

- Görüntü yükleme
- Canlı kamera ve tek kare alma

Her denetimde ürün bölgesi bulunur, dikdörtgen panel geometrisi ve ürün rengi kullanılarak ROI/maske rafine edilir, OpenCV tabanlı hata kuralları çalışır ve sonuç operatör onayıyla kayıt altına alınır.

## Mevcut Durum

Sistem kurumsal demo seviyesinde çalışır durumdadır:

- Tek sayfa Streamlit arayüzü
- Görüntü yükleme ve canlı kamera akışı
- Ürün ROI, maske ve şekle göre bbox üretimi
- Kenar hasarı, renk sapması, koyu çizgi/çatlak ve yerel anomali skorları
- Karar motoru: `SAĞLAM`, `ŞÜPHELİ`, `HATALI`
- Operatör etiketi, not, model yanlış işareti
- Silinebilir denetim kayıtları
- CSV ve veri seti dışa aktarımı
- Kayıtları güncel algoritmayla yeniden tarama
- Oturum içi HSV renk kalibrasyonu

## Kurulum

Python 3.10+ önerilir.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Arayüz:

```text
http://localhost:8501/
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall app.py src tests
```

## Operasyon Akışı

1. Görüntü yükle veya canlı kameradan kare al.
2. Sistem ürünü bulur ve hata adaylarını işaretler.
3. Karar ve skor denetim kartında görünür.
4. Operatör etiketi ve not girilerek kayıt oluşturulur.
5. Hatalı model kararları `Model yanlış` ile işaretlenir.
6. Gerekirse `Kayıtları yeniden tara` ile eski kayıtlar güncel algoritmayla yeniden işlenir.

## Veri Klasörleri

- `data/raw`: Kaydedilen ham görüntüler
- `data/database`: SQLite veritabanı
- `outputs/overlays`: İşaretlenmiş denetim çıktıları
- `outputs/reports`: Rapor çıktıları
- `outputs/heatmaps`: Anomali çıktıları

## Config

Temel ayarlar `config.yaml` içindedir.

Önemli eşikler:

- `product_hsv_lower`
- `product_hsv_upper`
- `product_color_profile_threshold`
- `crack_darkness_threshold`
- `local_anomaly_threshold`
- `anomaly_score_suspicious`
- `anomaly_score_defect`

Ortam ışığı veya ürün tonu değişirse önce arayüzdeki `Kalibrasyon` alanından ürün rengini kalibre etmek önerilir.

## Sonraki Faz

Kurumsal pilot için sıradaki teknik adımlar:

- Daha geniş gerçek saha veri setiyle eşik doğrulama
- Operatör geri bildiriminden otomatik eşik önerisi
- PatchCore veya benzeri öğrenilmiş modelin adapter üzerinden devreye alınması
- Rol bazlı kullanıcı akışı ve denetim raporu çıktısı
