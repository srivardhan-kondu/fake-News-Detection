import sys, json, math
sys.path.insert(0, '.')
import joblib, numpy as np
from app.services.preprocessing import preprocess_text

artifact_dir = 'app/models_artifacts'
vectorizer = joblib.load(f'{artifact_dir}/vectorizer.joblib')
ml_models = joblib.load(f'{artifact_dir}/ml_models.joblib')

samples = [
    (
        'REAL',
        'Bobby Jindal pastor dinner conversion story',
        'A dozen politically active pastors came here for a private dinner Friday night to hear a conversion story unique in the context of presidential politics: how Louisiana Gov. Bobby Jindal traveled from Hinduism to Protestant Christianity and ultimately became what he calls an evangelical Catholic.',
    ),
    (
        'FAKE',
        'SATAN 2 Russia Sarmat missile SUPERNUKE',
        'The RS-28 Sarmat missile dubbed Satan 2 will replace the SS-18. It flies at 4.3 miles per second with a range of 6213 miles. It could deliver a warhead of 40 megatons, 2000 times as powerful as the atom bombs dropped on Hiroshima and Nagasaki in 1945.',
    ),
    (
        'REAL',
        'Schumer Puerto Rico relief',
        'WASHINGTON Reuters Charles Schumer the top Democrat in the US Senate called on President Donald Trump on Sunday to name a single official to oversee and coordinate relief efforts in hurricane-ravaged Puerto Rico. Schumer along with Representatives said a CEO of response and recovery is needed to manage the complex and ongoing federal response in the territory.',
    ),
    (
        'REAL',
        'Odebrecht Brazil corruption house arrest',
        'RIO DE JANEIRO Reuters Billionaire Marcelo Odebrecht the highest-profile executive imprisoned in Brazils massive graft scandal was released from jail on Tuesday to continue his sentence for corruption under house arrest according to a federal court. The former chief executive officer of Odebrecht Latin Americas largest construction firm was arrested in 2015.',
    ),
    (
        'REAL',
        'Brexit EU citizens rights parliament',
        'BRUSSELS Reuters British Prime Minister Theresa May offer of settled status for EU residents is flawed and will leave them with fewer rights after Brexit the European Parliament Brexit coordinator said on Tuesday. A family of five could face a bill of 360 pounds to acquire the new status Guy Verhofstadt told May Brexit Secretary David Davis.',
    ),
]

print('='*70)
for true_label, title, text in samples:
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    print(f'TRUE LABEL : {true_label}')
    print(f'TITLE      : {title}')
    for name, model in ml_models.items():
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(features)[0][1])
        else:
            margin = float(model.decision_function(features)[0])
            prob = 1 / (1 + math.exp(-margin))
        pred = 'FAKE' if prob >= 0.5 else 'REAL'
        correct = 'OK' if pred == true_label else 'WRONG'
        print(f'  {name:25s}: prob_fake={prob:.3f}  => {pred}  [{correct}]')
    print()
