"""Find articles from the dataset that the model confidently and CORRECTLY classifies."""
import sys, math
sys.path.insert(0, '.')
import pandas as pd, joblib, numpy as np
from app.services.preprocessing import preprocess_text

artifact_dir = 'app/models_artifacts'
vectorizer = joblib.load(f'{artifact_dir}/vectorizer.joblib')
ml_models = joblib.load(f'{artifact_dir}/ml_models.joblib')

df = pd.read_csv('app/data/sample_news.csv').dropna(subset=['title','text'])
df['title'] = df['title'].str.strip()
df['text'] = df['text'].str.strip()
df = df[(df['title'].str.len() > 20) & (df['text'].str.len() > 200)]
df = df.reset_index(drop=True)

def get_svm_prob(text):
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    model = ml_models['linear_svm']
    margin = float(model.decision_function(features)[0])
    return 1 / (1 + math.exp(-margin))

print('Finding REAL articles (label=0) with high confidence correct predictions...')
real_df = df[df['label'] == 0].copy()
real_df['prob_fake'] = real_df['text'].apply(get_svm_prob)
# prob_fake < 0.2 means confidently REAL
confident_real = real_df[real_df['prob_fake'] < 0.20].sort_values('prob_fake').head(10)
print('\n=== CONFIDENT REAL (correctly classified) ===')
for _, row in confident_real.iterrows():
    print(f'\nProb_Fake={row["prob_fake"]:.3f}  TITLE: {row["title"][:120]}')
    print(f'TEXT: {row["text"][:250].strip()}')

print('\n\nFinding FAKE articles (label=1) with high confidence correct predictions...')
fake_df = df[df['label'] == 1].copy()
fake_df['prob_fake'] = fake_df['text'].apply(get_svm_prob)
# prob_fake > 0.8 means confidently FAKE
confident_fake = fake_df[fake_df['prob_fake'] > 0.80].sort_values('prob_fake', ascending=False).head(10)
print('\n=== CONFIDENT FAKE (correctly classified) ===')
for _, row in confident_fake.iterrows():
    print(f'\nProb_Fake={row["prob_fake"]:.3f}  TITLE: {row["title"][:120]}')
    print(f'TEXT: {row["text"][:250].strip()}')
