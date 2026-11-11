"""Render Turkish report and iteration CSV from measured JSON (no new experiment)."""
import csv
import json
from pathlib import Path


def main():
    out = Path('results/deepred')
    m = json.loads((out/'metrics.json').read_text())
    p = m['pnp']; ph = p['history'][-1]
    rows = []
    for run in m['red']:
        h = run['history'][-1]
        rows.append(f"| {run['lam']} | {run['step']} | {h['nmse_db']:.4f} | {run['history'][9]['nmse_db']:.4f} | {h['data_residual']:.6g} | {h['equilibrium_relative']:.6g} | {run['seconds']:.3f} | {run['denoiser_calls']} | {run['status']} |")
    e = m['extended_red']; eh = e['history'][-1]
    report = f'''# DeepRED: İlk Channel Estimation deneyi

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
- Özgün PnP ayrıca değişmeden subprocess içinde çalıştı: exit={m['original_exit_code']}.
- Yeniden uygulanan özgün akış ile log NMSE'si 1e-6 dB toleransında eşleşti.
- Adjoint bağıl hatası: {m['adjoint_relative_error']:.3g}.
- Yedi birim testi geçti: adjoint, dönüşüm/ölçek, yönlü veri türevi, analitik lineer
  denoiser çözümü, büyüyen iterasyon tespiti, NaN tespiti ve sıfır maskede denge.
- FFDNet doğrudan inference çıktısı `predict` ile eşleşti; tekrarlı inference aynı çıktı verdi.

## Ölçülen sonuçlar

Daha negatif NMSE daha iyidir.

| Referans | NMSE (dB) | Çıktı |
|---|---:|---|
| Özgün PnP | {m['original_nmse_db']:.4f} | Son veri adımı z |
| Özgün PnP, denoiser sonrası | {m['original_reproduction']['history'][-1]['nmse_db']:.4f} | Son h |
| Kontrollü PnP | {ph['nmse_db']:.4f} | Son h, ortak başlangıç |

Kontrollü PnP: {p['seconds']:.3f} saniye, {p['denoiser_calls']} denoiser çağrısı,
son veri residual'ı {ph['data_residual']:.6g}.
Özgün subprocess {m['original_process_seconds']:.3f} saniye sürdü; model yükleme ve
Python başlangıcını içerdiği için sıcak solver süreleriyle doğrudan karşılaştırılmaz.

| Lambda | Adım | 100 güncelleme NMSE | 10 çağrıda NMSE | Veri residual | Göreli denge residual | Süre (s) | Çağrı | Durum |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
'''+'\n'.join(rows)+f'''

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
çalıştırıldı: durum **{e['status']}**, son iterasyon {eh['iteration']},
NMSE **{eh['nmse_db']:.4f} dB**, göreli denge residual **{eh['equilibrium_relative']:.6g}**,
veri residual **{eh['data_residual']:.6g}**, süre **{e['seconds']:.3f} s**,
çağrı sayısı **{e['denoiser_calls']}**.
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
'''
    (out/'RAPOR_TR.md').write_text(report)
    records=[]
    for label,run in [('pnp',m['pnp'])]+[(f"red_l{r['lam']}_s{r['step']}",r) for r in m['red']]+[('red_extended',e)]:
        records.extend(dict(method=label,**h) for h in run['history'])
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (out/'iterations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(records)


if __name__ == '__main__':
    main()
