import requests

samples = [
    ("حکومت نے عوام کے لیے بجٹ میں اضافہ کیا جس کی تصدیق وزارت خزانہ نے کی", "Real"),
    ("یہ خبر بالکل جھوٹی ہے پروپیگنڈا کے سوا کچھ نہیں دشمن ملک کی سازش ہے", "Fake"),
    ("حکومت کی پالیسی صرف امیروں کے حق میں ہے غریب عوام کو نظرانداز کیا جارہا ہے", "Bias"),
]

for text, expected in samples:
    for model in ["nb", "cnn"]:
        r = requests.post("http://127.0.0.1:5000/predict",
                          json={"text": text, "model": model})
        d = r.json()
        probs = d["probabilities"]
        print(
            f"[{model.upper()}] Expected={expected:<4}  "
            f"Got={d['label']:<4}  Conf={d['confidence']:5.1f}%  "
            f"Real={probs['Real']}%  Bias={probs['Bias']}%  Fake={probs['Fake']}%"
        )
    print()
