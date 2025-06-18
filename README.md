 ![Demo](demo.png)
 
# Akıllı Trafik Sinyal Optimizasyon Sistemi

Bu proje, trafik sinyal optimizasyonu için derin öğrenme ve pekiştirmeli öğrenme yaklaşımlarını karşılaştıran bir sistemdir.

## Özellikler

- SUMO trafik simülasyonu entegrasyonu
- PyTorch tabanlı derin öğrenme modeli
- Pekiştirmeli öğrenme ajanları
- OpenRouter LLM entegrasyonu
- Gerçek zamanlı performans görselleştirme
- Streamlit tabanlı kullanıcı arayüzü

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. SUMO'yu yükleyin:
- macOS: `brew install sumo`
- Linux: `sudo apt-get install sumo sumo-tools sumo-doc`

3. Çevre değişkenlerini ayarlayın:
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

## Kullanım

```bash
streamlit run src/app.py
```

2000 saniyelik 4 yol ağzı ışık simülasyonunda trafik yoğunluğu 0.50 iken

Pekiştirmeli öğrenme - 
{
  "waiting_time": 77.75,
  "queue_length": 0,
  "average_speed": 6.945,
  "reward": -4.997
}

Derin Öğrenme - 
{
"waiting_time":433
"queue_length":0
"average_speed":6.945
"reward":-40.522000000000006
}

