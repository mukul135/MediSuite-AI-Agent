# ============================================================
#  MediSuite-AI-Agent -- ml/generate_fraud_dataset.py
#  Feature 8: Synthetic fraud dataset generator
#
#  Run: python ml/generate_fraud_dataset.py
#  Output: dataset/fraud_detection_dataset.csv
# ============================================================

import os, csv, random
from datetime import datetime, timedelta

random.seed(99)
NUM_RECORDS = 2000

DISEASES = ["Typhoid","Dengue","Diabetes","Hypertension","Pneumonia",
            "Cardiac Arrest","Kidney Failure","Cancer","Fracture","Malaria"]

BILL_RANGES = {"normal":(3000,80000), "high":(80001,500000)}

def random_date(start=2022, end=2025):
    s = datetime(start,1,1)
    return s + timedelta(days=random.randint(0,(datetime(end,12,31)-s).days))

def generate_record(i):
    is_fraud = random.random() < 0.18   # 18% fraud rate

    bill_range = BILL_RANGES["high"] if is_fraud and random.random()<0.6 else BILL_RANGES["normal"]
    bill = round(random.uniform(*bill_range), 2)

    ver_score  = random.randint(0,40)  if is_fraud else random.randint(50,100)
    ver_status = random.choice(["Rejected","Incomplete"]) if is_fraud else \
                 ("Eligible" if ver_score>=75 else "Incomplete")
    appr_prob  = random.uniform(5,40)  if is_fraud else random.uniform(55,99)
    prev_claims= random.randint(3,10)  if is_fraud else random.randint(0,3)
    freq_7d    = random.randint(3,8)   if is_fraud else random.randint(0,2)
    ins_valid  = 0 if is_fraud and random.random()<0.4 else 1
    pol_valid  = 0 if is_fraud and random.random()<0.3 else 1
    adm_date   = random_date()
    stay_days  = random.randint(1,3) if is_fraud else random.randint(1,14)
    dis_date   = adm_date + timedelta(days=stay_days)

    return {
        "bill_amount":          bill,
        "admission_days":       stay_days,
        "verification_score":   ver_score,
        "verification_status":  ver_status,
        "approval_probability": round(appr_prob,2),
        "previous_claims":      prev_claims,
        "claim_frequency_7d":   freq_7d,
        "insurance_id_valid":   ins_valid,
        "policy_valid":         pol_valid,
        "has_doctor":           1 if random.random()>0.1 else 0,
        "has_disease":          1 if random.random()>0.05 else 0,
        "is_fraud":             int(is_fraud),
    }

def generate(path, n=NUM_RECORDS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = [generate_record(i) for i in range(n)]
    with open(path,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
    fraud_count = sum(1 for r in records if r["is_fraud"])
    print(f"Generated {path}  |  {n} records  |  {fraud_count} fraud ({fraud_count/n*100:.1f}%)")

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate(os.path.join(base,"dataset","fraud_detection_dataset.csv"))