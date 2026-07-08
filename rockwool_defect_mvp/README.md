# Taş Yünü Kalite Kontrol

## MEGA Next.js Ana Arayuz

Yeni ana arayuz Next.js + FastAPI ile calisir. Streamlit prototip olarak kalmistir.

```powershell
.\start_mega_qc.ps1
```

- Frontend: http://localhost:3000
- Backend: http://127.0.0.1:8000

Backend mevcut OpenCV kalite kontrol akisindakileri kullanir: `process_frame`, ROI/bbox ve 8 bagimsiz hata siniflandirmasi. Frontend `/api/*` isteklerini Next.js proxy uzerinden FastAPI backend'e yollar.

## Hata Türleri (8 sınıf)

Her tür kendi bağımsız dedektörüyle ayrı ayrı çalışır:

1. **Çatlak** (`dark_crack`) — ince, çizgisel, uzun sürekli koyu yapı (çok yönlü black-hat).
2. **Cam yanığı / koyu leke** (`glass_burn`) — kompakt koyu kahverengi-siyah bölge.
3. **Renk / leke farklılığı** (`color_anomaly`) — panelden kromatik sapan bölge (L kanalı yarı ağırlık).
4. **Çiğ elyaf** (`raw_fiber`) — açık renkli, düşük doygunluklu, lifsi/homojen olmayan yüzey.
5. **Kenar bozukluğu** (`edge_damage`) — kenarda çentik, kopma, düzensiz sınır.
6. **Boyut / gönye hatası** (`size_tolerance`) — sabit kamera px/mm kalibrasyonuyla ölçü ve köşe dikliği.
7. **Deformasyon** (`deformation`) — global form bozulması (eğilme/ezilme, kenar yayı).
8. **Sağlam** — hata bulunmayan ürün (KABUL kararı).

Ek olarak `local_anomaly` destekleyici bir genel-tarama sinyalidir (tek başına RED veremez).

Sınıf ayrımı her dedektörün kendi şekil/renk mantığıyla sağlanır; yalnızca çatlak–yanık
çakışmasında tek bir açık hakem kuralı (`arbitrate_overlaps`) devreye girer.

## Denetim Modu (telefon / sabit kamera)

`config.yaml` içindeki `inspection_mode` iki senaryoyu ayırır:

- **`phone`** (öntanımlı) — telefonla çekilip yüklenen görüntüler. Mesafe/açı
  değişken olduğundan **boyut/gönye ve deformasyon kararı (RED) etkilemez**
  (perspektif yanlış RED üretmesin); karar yüzey/çizgi/kenar hatalarına dayanır.
  Bu sınıflar yine hesaplanır ve raporlanır, ama tek başlarına ürünü reddedemez.
- **`fixed_camera`** — bant üstü sabit/üstten kamera. Tüm sınıflar (boyut/gönye
  ve deformasyon dahil) kararda tam etkilidir.

## Boyut/Gönye ve Arka Plan Kalibrasyonu (yalnız sabit kamera)

> Not: Boyut/gönye ve arka plan referansı yalnız `fixed_camera` modunda anlamlıdır;
> telefon senaryosunda kullanılmaz.


Boyut kontrolü öntanımlı kapalıdır. Sabit kamera kurulumunda:

- `POST /api/calibration/size` (`{recordId, knownWidthMm, knownHeightMm}`) — bilinen ölçülü bir
  panelden px/mm öğrenir, `data/calibration/size_calibration.json` sidecar'ına yazar ve boyut
  kontrolünü etkinleştirir. `config.yaml` yorumlarına dokunulmaz.
- `POST /api/calibration/background` (görüntü) — boş bant referansı; düşük kontrastlı panellerde
  en güvenilir ürün ayrımı için kullanılır. `DELETE` ile kapatılır.
- Beklenen ölçü/tolerans `config.yaml`: `expected_width_mm`, `expected_height_mm`,
  `size_tolerance_mm`, `squareness_tolerance_deg`.
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

- Daha geniş gerçek saha veri setiyle eşik doğrulama (`decision_profile` ile sınıf-bazlı ayar)
- Operatör geri bildiriminden otomatik eşik önerisi
- PatchCore veya benzeri öğrenilmiş modelin adapter üzerinden devreye alınması
- Rol bazlı kullanıcı akışı ve denetim raporu çıktısı

## Karar Motoru Ayarı

Sınıf-bazlı eşikler `src/decision/decision_engine.py` içindeki `DEFAULT_PROFILE`'da tanımlıdır
(her sınıf için `warn`/`reject`/`weight`). Saha verisiyle ayar için `config.yaml` içindeki
`decision_profile` ile yalnızca istenen alanlar ezilebilir, örn:

```yaml
decision_profile:
  raw_fiber:
    reject: 0.5
```
