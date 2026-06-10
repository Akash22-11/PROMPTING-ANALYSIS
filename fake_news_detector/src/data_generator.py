"""
Synthetic dataset generator for fake news detection.
Generates realistic fake and real news articles with distinguishing linguistic features.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

REAL_TEMPLATES = [
    "According to a report published by {org}, {topic} has {change} by {pct}% in {year}. Officials cited {reason} as the primary driver.",
    "Researchers at {university} have found that {topic} correlates with {factor}. The study, peer-reviewed and published in {journal}, examined {num} cases.",
    "The {country} government announced on {day} that new legislation regarding {topic} will take effect in {month}. Experts say the measure aims to {goal}.",
    "{official}, speaking at a press conference, confirmed that {topic} remains {status}. Transparency remains a stated priority.",
    "A new analysis by {org} shows that {topic} trends have remained {status} over the past {num} months, according to verified data.",
    "Scientists from {university} published findings suggesting that {topic} may {change} under certain conditions. The research involved {num} participants.",
    "Global markets responded {status} to news that {topic} would {change}, with analysts pointing to {reason} as a key factor.",
    "Local authorities in {country} confirmed that {topic} is under review. The {org} released a detailed statement citing {reason}.",
    "According to {official} from the {org}, the {topic} initiative has reached {pct}% of its stated objective for {year}.",
    "Data released by {university} researchers indicate that {topic} affects approximately {num} people annually, prompting calls for {goal}.",
    "An independent audit commissioned by {org} found no evidence of misconduct related to {topic} in {year}.",
    "The joint committee on {topic} released its quarterly findings, noting a {pct}% {change} since {year}.",
]

FAKE_TEMPLATES = [
    "SHOCKING: {topic} is secretly {change} and the mainstream media REFUSES to report this! Sources close to the situation reveal the TRUTH.",
    "They don't want you to know this about {topic}! A brave insider exposes how {org} has been {change} for years. SHARE before it gets deleted!",
    "BREAKING: {official} caught red-handed! New evidence proves {topic} has been {change} all along. The deep state is trying to HIDE this!",
    "The {country} cover-up nobody is talking about: {topic} has been {change} since {year}. Wake up people — this is happening RIGHT NOW!",
    "100% CONFIRMED: {topic} causes {factor}! Big {org} will NEVER let this reach the public. One doctor finally speaks out!",
    "URGENT: Forward this to everyone you know! {topic} is being {change} by elites who control {org}. The proof is UNDENIABLE.",
    "They laughed when he said {topic} was {status}. Now the whole world knows the TRUTH about {org} and their secret agenda!",
    "MUST READ: How {official} lied to ALL of us about {topic}. The evidence they buried could {change} everything you believe!",
    "Scientists SILENCED after discovering {topic} actually {change}! {org} paid millions to keep this hidden from the public.",
    "WARNING: The {topic} agenda exposed! What {country} government is REALLY doing will make your blood boil. Real patriots must act NOW!",
    "This bombshell report on {topic} was REMOVED from the internet within hours. We saved a copy — read it before it disappears AGAIN.",
    "EXCLUSIVE: Whistleblower inside {org} confirms {topic} is a FRAUD. Everything you were taught about {factor} is a LIE.",
]

ORGS = ["WHO", "Reuters", "CDC", "NASA", "Federal Reserve", "UN", "IMF", "World Bank", "OECD", "Red Cross"]
UNIVERSITIES = ["MIT", "Stanford University", "Oxford University", "Harvard Medical School", "Cambridge University", "Johns Hopkins University"]
OFFICIALS = ["A spokesperson", "The Director", "A senior analyst", "The committee chair", "A senior researcher"]
TOPICS = ["climate policy", "public health funding", "economic forecasting", "infrastructure development",
          "agricultural output", "renewable energy adoption", "digital privacy regulation",
          "vaccine distribution", "trade agreement terms", "urban housing supply"]
FACTORS = ["economic inequality", "public trust", "productivity levels", "health outcomes", "migration patterns"]
REASONS = ["increased investment", "policy reform", "shifting demographics", "technological adoption", "global trade dynamics"]
GOALS = ["improve transparency", "reduce inequality", "accelerate development", "protect civil liberties", "stabilize markets"]
CHANGES = ["increased", "decreased", "stabilized", "shifted significantly", "shown measurable improvement"]
STATUSES = ["stable", "under scrutiny", "improving", "under review", "within normal parameters"]
COUNTRIES = ["Germany", "Canada", "Japan", "Brazil", "South Korea", "Australia", "India", "France"]
JOURNALS = ["Nature", "The Lancet", "Science", "JAMA", "PLOS ONE", "Cell"]
MONTHS = ["January", "March", "June", "September", "October", "November"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def fill_template(template):
    replacements = {
        "{org}": random.choice(ORGS),
        "{university}": random.choice(UNIVERSITIES),
        "{official}": random.choice(OFFICIALS),
        "{topic}": random.choice(TOPICS),
        "{factor}": random.choice(FACTORS),
        "{reason}": random.choice(REASONS),
        "{goal}": random.choice(GOALS),
        "{change}": random.choice(CHANGES),
        "{status}": random.choice(STATUSES),
        "{country}": random.choice(COUNTRIES),
        "{journal}": random.choice(JOURNALS),
        "{month}": random.choice(MONTHS),
        "{day}": random.choice(DAYS),
        "{year}": str(random.randint(2019, 2024)),
        "{num}": str(random.randint(50, 5000)),
        "{pct}": str(round(random.uniform(1.5, 48.7), 1)),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


def add_noise(text, is_fake):
    if is_fake:
        text = text.replace("!", "!!")
        if random.random() > 0.5:
            text += " This is what THEY don't want you to see..."
        if random.random() > 0.7:
            text += " #WakeUp #TruthBomb"
    else:
        suffixes = [
            " Further investigation is ongoing.",
            " The findings are subject to peer review.",
            " Independent experts have yet to comment.",
            " Full data is available in the supplementary report.",
        ]
        text += random.choice(suffixes)
    return text


def generate_title(text, is_fake):
    words = text.split()[:6]
    title = " ".join(words).rstrip(".,!?")
    if is_fake:
        prefixes = ["BREAKING: ", "SHOCKING: ", "EXPOSED: ", "MUST READ: ", "WARNING: "]
        title = random.choice(prefixes) + title
    else:
        title = title.capitalize()
    return title


def generate_dataset(n_real=600, n_fake=600):
    records = []
    for _ in range(n_real):
        body = add_noise(fill_template(random.choice(REAL_TEMPLATES)), is_fake=False)
        records.append({"title": generate_title(body, False), "text": body, "label": 0, "label_name": "REAL"})
    for _ in range(n_fake):
        body = add_noise(fill_template(random.choice(FAKE_TEMPLATES)), is_fake=True)
        records.append({"title": generate_title(body, True), "text": body, "label": 1, "label_name": "FAKE"})
    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    import os
    df = generate_dataset()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "news_dataset.csv"), index=False)
    print(f"Dataset: {len(df)} articles | Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")
