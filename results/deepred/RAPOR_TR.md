# DeepRED: İlk Channel Estimation deneyi

## Uygulama

Özgün dosyalar değiştirilmedi. `deepred/ce.py` içinde ayrı PnP ve RED
solver'ları, ortak ölçüm operatörü ve aynı FFDNet ağırlıklarını kullanan inference
sarmalayıcısı eklendi. RED, sabit sigma=0.1 ve sabit lambda ile
`x <- x - eta [A*(Ax-b) + lambda(x-D(x))]` denge iterasyonunu uyguluyor.
Jacobian koşulları doğrulanmadığından bu yönün klasik RED enerjisinin gerçek
gradyanı olduğu iddia edilmiyor. Yeni eğitim yapılmadı.

NumPy FFT ölçeği için S*=8T; A=MS/sqrt(8), b=y/sqrt(8).
Veri residual'ı ||Ax-b||, göreli denge residual'ı ||g||/||x|| olarak raporlandı.
Bu veri ölçeği lambda'nın anlamını belirler; gürültü varyansıyla ayrıca bölünmedi.

## Deney ve doğrulama

- Tek mevcut CE kanal örneği, seed=2026, nominal SNR=20 dB, 256 pilot.
- Özgün programın fiilen kullandığı mode=2 maskesi korundu. CLI mode davranışı değiştirilmedi.
- Kontrollü PnP ve RED için gözlem, gürültü, maske ve projekte RBF başlangıcı aynı.
- İki kontrollü çıktı da denoiser sonrası sentezlenen tam kanal üzerinden değerlendirildi.
- Özgün PnP ayrıca değişmeden subprocess içinde çalıştı: exit=0.
- Yeniden uygulanan özgün akış ile log NMSE'si 1e-6 dB toleransında eşleşti.
- Adjoint bağıl hatası: 2.23e-16.
- Yedi birim testi geçti: adjoint, dönüşüm/ölçek, yönlü veri türevi, analitik lineer
  denoiser çözümü, büyüyen iterasyon tespiti, NaN tespiti ve sıfır maskede denge.
- FFDNet doğrudan inference çıktısı `predict` ile eşleşti; tekrarlı inference aynı çıktı verdi.

## Ölçülen sonuçlar

Daha negatif NMSE daha iyidir.

| Referans | NMSE (dB) | Çıktı |
|---|---:|---|
| Özgün PnP | -23.1930 | Son veri adımı z |
| Özgün PnP, denoiser sonrası | -23.3493 | Son h |
| Kontrollü PnP | -23.4075 | Son h, ortak başlangıç |

Kontrollü PnP: 0.370 saniye, 10 denoiser çağrısı,
son veri residual'ı 0.00762727.
Özgün subprocess 8.604 saniye sürdü; model yükleme ve
Python başlangıcını içerdiği için sıcak solver süreleriyle doğrudan karşılaştırılmaz.

| Lambda | Adım | 100 güncelleme NMSE | 10 çağrıda NMSE | Veri residual | Göreli denge residual | Süre (s) | Çağrı | Durum |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 0.25 | -19.4709 | -18.7593 | 0.000633039 | 0.000637009 | 3.531 | 101 | max_iterations |
| 0.01 | 0.75 | -20.6137 | -19.0003 | 0.000507301 | 0.000386889 | 2.953 | 101 | max_iterations |
| 0.1 | 0.25 | -22.7534 | -19.5581 | 0.00313696 | 0.00113152 | 2.443 | 101 | max_iterations |
| 0.1 | 0.75 | -24.0488 | -20.9372 | 0.00311874 | 0.000388864 | 2.215 | 101 | max_iterations |
| 1.0 | 0.25 | -20.7881 | -22.1965 | 0.00775732 | 0.00261442 | 2.497 | 101 | max_iterations |
| 1.0 | 0.75 | -15.7638 | -22.6279 | 0.010811 | 0.00222699 | 2.185 | 101 | max_iterations |

100 güncelleme 101 denoiser çağrısı gerektirir: son denge residual'ı için D(x)
hesaplaması da sayıldı. Eşit 10 çağrı bütçesinde tüm RED ayarları kontrollü
PnP'den kötü kaldı. En düşük 100-güncelleme NMSE'si lambda=.1, eta=.75
ayarında gözlendi; bu ayar test örneği üzerinde gözlendiği için doğrulanmış
hiperparametre seçimi veya yöntem üstünlüğü sayılmaz.

## RED yakınsadı mı?

İlk altı koşunun hiçbiri göreli denge residual <=1e-5 eşiğine ulaşmadı.
Hepsi `max_iterations`; NaN/Inf veya büyüme eşiği aşımı görülmedi.
Bu, hepsinin yakınsadığı veya monoton iyileştiği anlamına gelmez.
Özellikle lambda=1, eta=.75 için 10 çağrıdan 100 güncellemeye NMSE kötüleşti.

En düşük 100-güncelleme NMSE'sini veren ayar ayrıca en fazla 1000 güncelleme
çalıştırıldı: durum **converged**, son iterasyon 665,
NMSE **-24.1495 dB**, göreli denge residual **9.94219e-06**,
veri residual **0.00319747**, süre **14.784 s**,
çağrı sayısı **666**.
Bu uzatma aynı başlangıçtan bağımsız yeniden çalıştırıldı; ara örnekten devam edilmedi.

## Olası matematiksel açıklamalar ve sınırlar

- Küçük lambda ölçümlere daha sıkı uyum sağlarken gözlenmeyen bileşenleri yavaş düzeltebilir.
- Büyük lambda denoiser yanlılığını güçlendirebilir; küçük veri residual'ı tek başına iyi NMSE değildir.
- Küçük adımlar pratik önlemdir. FFDNet Jacobian'ının simetrisi, yerel homojenliği ve
  spektral özellikleri ölçülmedi; global yakınsama garantisi verilemez.
- Sabit RED sigma ile PnP'nin azalan sigma çizelgesi farklıdır. Sonuç farkı yalnızca
  solver değişikliğine atfedilemez; sabit-sigma PnP ablasyonu sonraki aşamadır.
- Her iki kontrollü nihai çıktı ilk 32 gecikme bileşeninin altuzayındadır; gerçek
  kanalın bu altuzay dışındaki enerjisi yeniden oluşturulamaz.
- Tek kanal ve tek gürültü örneği güven aralığı veya genelleme çıkarımı sağlamaz.
- Eşit çağrı kıyası verildi; süreler tek CPU koşusudur, tekrarlı zamanlama ölçümü değildir.
- RED denge residual'ının küçülmesi, gerçek kanal hatasının küçülmesini garanti etmez.

## Sonraki araştırma

Prototip çalışıyor; kısa bütçede RED daha kötü, daha uzun bütçede bazı ayarlar
rekabetçi sonuç verdi. Bu gözlem daha kapsamlı araştırmayı hak ediyor fakat
üstünlük kanıtı değildir. Çok kanallı bağımsız doğrulama/test ayrımı, eşleştirilmiş
çoklu gürültü tekrarları, sabit-sigma PnP ablasyonu, SNR/pilot/dağılım taraması ve
Jacobian ölçümleri önceliklidir. Yakınsama eşiği ile NMSE ayrı izlenmelidir.

## Yeniden üretim ve ham kayıtlar

Komutlar `deepred/README.md` içindedir. `metrics.json` her iterasyon için NMSE,
veri/denge residual'ı, süre ve çağrı sayısını içerir; `iterations.csv` aynı
kayıtların tablo biçimidir. `arrays.npz` gözlemi, gürültüyü, başlangıcı ve
100-güncelleme tahminlerini saklar. Model/veri SHA256 değerleri JSON'dadır.
`original_pnp.log` değiştirilmemiş programın gerçek çıktısıdır.
