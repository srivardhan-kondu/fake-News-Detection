"""Find articles from the dataset that the model confidently and CORRECTLY classifies."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, joblib, numpy as np
from app.services.preprocessing import preprocess_text

# ── Pre-selected demo data (from WELFake dataset, model-verified) ──────────
DEMO_DATA = {
    "fake": [
        {
            "title": "Trump Supporter Pulls Gun On Black Man At Polling Station For Refusing To Vote Trump, Gets Arrested",
            "text": (
                "November 8, 2016 Trump Supporter Pulls Gun On Black Man At Polling Station "
                "For Refusing To Vote Trump, Gets Arrested. We knew it was going to happen, "
                "and honestly, it could have been a lot worse. A Trump supporter in the swing "
                "state of Ohio pulled a gun on a black man outside of a polling station after "
                "having an argument about who they were voting for. According to witnesses the "
                "Trump supporter asked the man who he was voting for, and when he said Hillary "
                "Clinton the man pulled a pistol on him."
            ),
            "confidence": 0.985,
        },
        {
            "title": "CO GOP Swears Racist Anti-Obama Pic Was A Hack – After Lawmaker Says It Was A 'Joke' (IMAGE)",
            "text": (
                "The Colorado GOP needs to get their stories straight regarding an outrageously "
                "racist anti-Obama Facebook post. Top Delta County lawmaker Linda Sorenson "
                "posted a picture of President Reagan feeding a baby monkey, and the caption "
                "read, 'I'll be damned. Reagan used to babysit Obama.' This is incredibly "
                "offensive, and the kind of thing we've come to expect from a party that trades "
                "in racism and bigotry on a daily basis."
            ),
            "confidence": 0.982,
        },
        {
            "title": "Clinton Lists The 9 Things Trump Accuses Her Of Doing – That He Actually Does Himself",
            "text": (
                "Trump's way of handling the heat is simple: deflect, and blame everything on "
                "everyone else. He is a master of pointing the finger and saying 'look, they do "
                "it too!' The evidence couldn't be more clear with how he accuses Hillary Clinton "
                "of everything that he himself is guilty of. Here's a handy list showing exactly "
                "how many times Donald Trump has accused Clinton of things he himself has done, "
                "from corruption to ties with foreign governments."
            ),
            "confidence": 0.983,
        },
    ],
    "real": [
        {
            "title": "House Republicans to meet with Trump on Thursday: aide",
            "text": (
                "WASHINGTON (Reuters) - House of Representatives Republican leaders and members "
                "of the Ways and Means Committee will meet with U.S. President Donald Trump on "
                "Thursday at the White House, a Republican aide said. The meeting will take place "
                "at around 1:30 p.m. (1830 GMT), and the main topic of discussion will be the "
                "Republican attempt to pass sweeping tax reform legislation."
            ),
            "confidence": 0.987,
        },
        {
            "title": "May Brexit offer would hurt, cost EU citizens - EU parliament",
            "text": (
                "BRUSSELS (Reuters) - British Prime Minister Theresa May's offer of 'settled "
                "status' for EU residents is flawed and will leave them with fewer rights after "
                "Brexit, the European Parliament's Brexit coordinator said on Tuesday. A family "
                "of five could face a bill of 360 pounds to acquire the new status, Guy "
                "Verhofstadt told May's Brexit Secretary David Davis in a letter seen by Reuters "
                "– 'a very significant amount for a family on low income.'"
            ),
            "confidence": 0.984,
        },
        {
            "title": "Schumer calls on Trump to appoint official to oversee Puerto Rico relief",
            "text": (
                "WASHINGTON (Reuters) - Charles Schumer, the top Democrat in the U.S. Senate, "
                "called on President Donald Trump on Sunday to name a single official to oversee "
                "and coordinate relief efforts in hurricane-ravaged Puerto Rico. Schumer, along "
                "with Representatives Nydia Velàzquez and Jose Serrano, said a 'CEO of response "
                "and recovery' is needed to manage the complex and ongoing federal response in "
                "the territory, where millions of Americans remain without power and supplies."
            ),
            "confidence": 0.983,
        },
    ],
}

base_dir = Path(__file__).resolve().parent.parent
artifact_dir = base_dir / 'app' / 'models_artifacts'
vectorizer = joblib.load(artifact_dir / 'vectorizer.joblib')
ml_models = joblib.load(artifact_dir / 'ml_models.joblib')

df = pd.read_csv(base_dir / 'app' / 'data' / 'sample_news.csv').dropna(subset=['title','text'])
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

# ── Quick demo mode: python find_good_demos.py --demo ───────────────────────
if '--demo' in sys.argv:
    print('=' * 70)
    print('  PRE-SELECTED DEMO ARTICLES (copy–paste into the web app)')
    print('=' * 70)
    for label in ('fake', 'real'):
        print(f'\n{"─" * 70}')
        print(f'  {label.upper()} NEWS SAMPLES')
        print(f'{"─" * 70}')
        for i, article in enumerate(DEMO_DATA[label], 1):
            print(f'\n  [{i}] {article["title"]}')
            print(f'      Model confidence: {article["confidence"]:.1%}')
            print(f'      Text:\n      {article["text"][:300]}')
    sys.exit(0)

# ── Full scan mode (default) ────────────────────────────────────────────────
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
