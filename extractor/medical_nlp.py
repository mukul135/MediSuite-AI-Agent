# ============================================================
#  MediSuite-AI-Agent -- extractor/medical_nlp.py
#  Feature 5: NLP Medical Analysis
#
#  Approach: Hybrid pipeline
#    1. spaCy NER       -- catches PERSON (doctors), DATE, ORG
#    2. PhraseMatcher   -- matches exact medical terms from dicts
#    3. Matcher         -- rule-based pattern matching (dosage)
#    4. Post-processing -- deduplicate, clean, rank entities
#
#  Input  : raw OCR text string
#  Output : structured dict with diseases, medicines, treatments etc.
#
#  DO NOT import or modify claim_extractor.py from here.
#  This module is fully independent.
# ============================================================

import re
import json

# spaCy import with graceful fallback
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Load spaCy model once at module level (not per-request)
NLP = None

def load_nlp_model():
    global NLP
    if NLP is not None:
        return NLP
    if not SPACY_AVAILABLE:
        return None
    try:
        NLP = spacy.load("en_core_web_sm")
        return NLP
    except OSError:
        return None


# ============================================================
#  SECTION 1: MEDICAL DICTIONARIES
# ============================================================

DISEASES = [
    "typhoid","malaria","dengue","tuberculosis","tb","covid-19","covid",
    "coronavirus","cholera","hepatitis","hepatitis a","hepatitis b","hepatitis c",
    "jaundice","typhoid fever","viral fever","bacterial infection",
    "urinary tract infection","uti","pneumonia","influenza","flu",
    "diabetes","diabetes mellitus","type 2 diabetes","type 1 diabetes",
    "hypertension","high blood pressure","hypotension","low blood pressure",
    "asthma","copd","chronic obstructive pulmonary disease",
    "hypothyroidism","hyperthyroidism","thyroid disorder",
    "kidney disease","chronic kidney disease","ckd","renal failure",
    "liver disease","liver failure","fatty liver","cirrhosis",
    "heart disease","coronary artery disease","cad","heart failure",
    "cardiac arrest","arrhythmia","atrial fibrillation",
    "cancer","carcinoma","tumor","tumour","malignancy","lymphoma",
    "leukemia","leukaemia","sarcoma","melanoma",
    "epilepsy","seizure","stroke","migraine","dementia","alzheimer",
    "parkinson","neuropathy",
    "gastritis","gastroenteritis","ulcer","peptic ulcer","ibs",
    "irritable bowel syndrome","appendicitis","hernia","colitis","pancreatitis",
    "arthritis","rheumatoid arthritis","osteoporosis","fracture",
    "dislocation","spondylitis",
    "bronchitis","pleurisy","pulmonary embolism","pleural effusion",
    "anaemia","anemia","sepsis","shock","gangrene","abscess",
    "cellulitis","dermatitis","eczema","psoriasis","cyst",
    "fibroids","endometriosis","pcos","dengue fever","thrombocytopenia"
]

MEDICINES = [
    "paracetamol","pcm","crocin","dolo","dolo 650","dolo-650",
    "ibuprofen","brufen","meftal","aspirin","disprin","diclofenac",
    "voveran","nimesulide",
    "azithromycin","azee","zithromax","amoxicillin","augmentin",
    "amoxyclav","ciprofloxacin","cipro","levofloxacin","levaquin",
    "metronidazole","flagyl","cefixime","cefix","ceftriaxone",
    "doxycycline","tetracycline","clindamycin","clarithromycin",
    "co-amoxiclav","piperacillin","tazobactam",
    "metformin","glycomet","glipizide","glibenclamide","insulin",
    "lantus","novomix","sitagliptin","januvia","vildagliptin",
    "pioglitazone","empagliflozin","jardiance",
    "atorvastatin","lipitor","rosuvastatin","crestor","amlodipine",
    "norvasc","ramipril","cardace","losartan","cozaar",
    "telmisartan","atenolol","metoprolol","carvedilol",
    "digoxin","warfarin","heparin","clopidogrel","plavix",
    "pantoprazole","pan","omeprazole","omez","rabeprazole",
    "ranitidine","rantac","domperidone","domstal","ondansetron",
    "emeset","loperamide",
    "salbutamol","asthalin","budesonide","formoterol","foracort",
    "montelukast","singulair","cetirizine","levocetrizine",
    "fexofenadine","allegra",
    "vitamin c","vitamin d","vitamin b12","zinc","calcium",
    "folic acid","iron","ferrous sulphate","multivitamin",
    "prednisolone","dexamethasone","methylprednisolone",
    "hydrocortisone","tacrolimus",
    "oseltamivir","tamiflu","chloroquine","hydroxychloroquine",
    "remdesivir","favipiravir","acyclovir",
    "normal saline","ringer lactate","dns","dextrose","5% dextrose","iv fluids"
]

TREATMENTS = [
    "surgery","operation","laparoscopy","appendectomy","cholecystectomy",
    "bypass surgery","angioplasty","angiography","stenting",
    "knee replacement","hip replacement","amputation","biopsy",
    "excision","incision and drainage","laparotomy","hysterectomy",
    "caesarean","c-section","colostomy",
    "chemotherapy","radiotherapy","radiation therapy","immunotherapy",
    "targeted therapy","bone marrow transplant",
    "dialysis","hemodialysis","peritoneal dialysis",
    "physiotherapy","physical therapy","occupational therapy",
    "speech therapy","respiratory therapy",
    "oxygen therapy","nebulization","nebulizer",
    "iv fluids","intravenous","blood transfusion",
    "ventilator","mechanical ventilation","intubation",
    "icu","intensive care","icu admission","icu stay",
    "admission","discharge","hospitalization",
    "wound dressing","suturing","catheterization",
    "nasogastric tube","ng tube","foley catheter",
    "defibrillation","cardioversion","pacemaker",
    "mri","mri scan","ct scan","ct","x-ray","xray",
    "ultrasound","usg","ecg","ekg","electrocardiogram",
    "echocardiogram","echo","pet scan","mammography",
    "colonoscopy","endoscopy","bronchoscopy",
    "blood test","blood work","cbc","complete blood count",
    "blood culture","urine culture","urine test","urine routine",
    "liver function test","lft","kidney function test","kft",
    "lipid profile","thyroid profile","tft",
    "hba1c","fasting blood sugar","fbs","random blood sugar","rbs",
    "widal test","dengue ns1","malaria antigen",
    "covid test","rt-pcr","rapid antigen test"
]

SYMPTOMS = [
    "fever","high fever","low grade fever",
    "cough","dry cough","productive cough",
    "cold","runny nose","nasal congestion",
    "headache","migraine","dizziness","vertigo",
    "nausea","vomiting","vomiting sensation",
    "diarrhoea","diarrhea","loose stools","watery stools",
    "constipation","bloating","abdominal pain","stomach pain",
    "chest pain","chest tightness","palpitations","breathlessness",
    "shortness of breath","sob","difficulty breathing",
    "fatigue","weakness","lethargy","tiredness",
    "loss of appetite","anorexia","weight loss",
    "pain","body ache","joint pain","muscle pain",
    "rash","skin rash","itching","pruritus",
    "swelling","edema","oedema",
    "jaundice","yellowish discolouration",
    "burning sensation","dysuria",
    "blood in urine","haematuria","hematuria",
    "bleeding","haemorrhage","hemorrhage"
]

TESTS = [
    "cbc","complete blood count","hb","haemoglobin","hemoglobin",
    "wbc","white blood cell","platelet","platelet count",
    "esr","erythrocyte sedimentation rate",
    "crp","c-reactive protein",
    "lft","liver function test","sgpt","sgot","bilirubin",
    "kft","kidney function test","creatinine","urea","bun",
    "uric acid","electrolytes","sodium","potassium",
    "lipid profile","cholesterol","triglycerides","ldl","hdl",
    "thyroid","tsh","t3","t4",
    "hba1c","fbs","rbs","ppbs",
    "urine routine","urine microscopy","urine culture",
    "blood culture","sputum culture",
    "widal","dengue ns1","dengue igg","dengue igm",
    "malaria antigen","mp smear",
    "covid rt-pcr","rapid antigen",
    "x-ray chest","x-ray","xray",
    "ecg","echo","2d echo",
    "ultrasound","usg abdomen","usg pelvis",
    "ct scan","ct chest","ct abdomen","ct brain",
    "mri brain","mri spine","mri",
    "pet scan","bone scan",
    "biopsy","fnac","pap smear",
    "endoscopy","colonoscopy","bronchoscopy",
    "spirometry","pulmonary function test","pft"
]

DEPARTMENTS = [
    "cardiology","cardiothoracic","neurology","nephrology",
    "gastroenterology","pulmonology","oncology","haematology",
    "hematology","endocrinology","rheumatology","orthopedics",
    "orthopaedics","urology","gynaecology","gynecology",
    "obstetrics","paediatrics","pediatrics","dermatology",
    "ophthalmology","ent","psychiatry","general surgery",
    "general medicine","emergency","icu","critical care",
    "radiology","pathology","anaesthesia","anesthesia",
    "physiotherapy","dietetics","dental","neonatology"
]


# ============================================================
#  SECTION 2: TEXT CLEANING
# ============================================================

def clean_text_for_nlp(raw_text):
    if not raw_text:
        return ""
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\x09\x0A\x20-\x7E\u00C0-\u024F\u0900-\u097F]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
#  SECTION 3: DICTIONARY-BASED EXTRACTION
# ============================================================

def extract_by_dictionary(text, dictionary):
    """Scan text for known terms. Returns deduplicated title-cased list."""
    text_lower = text.lower()
    found = set()
    for term in dictionary:
        term_lower = term.lower()
        pattern = r'(?<!\w)' + re.escape(term_lower) + r'(?!\w)'
        if re.search(pattern, text_lower):
            if len(term) <= 5 and term.upper() == term.upper():
                found.add(term.upper())
            else:
                found.add(term.title())
    return sorted(list(found))


# ============================================================
#  SECTION 4: SPACY NER
# ============================================================

def extract_by_spacy_ner(text, nlp):
    """Run spaCy NER. Returns doctors, dates, org_names."""
    result = {"doctors": [], "dates": [], "org_names": []}
    if nlp is None:
        return result
    doc = nlp(text)
    for ent in doc.ents:
        clean_val = ent.text.strip()
        if not clean_val or len(clean_val) < 2:
            continue
        if ent.label_ == "PERSON":
            if len(clean_val) > 3 and not clean_val.isdigit():
                result["doctors"].append(clean_val)
        elif ent.label_ == "DATE":
            result["dates"].append(clean_val)
        elif ent.label_ == "ORG":
            result["org_names"].append(clean_val)
    for key in result:
        result[key] = list(dict.fromkeys(result[key]))
    return result


# ============================================================
#  SECTION 5: PHRASE MATCHER
# ============================================================

def extract_with_phrase_matcher(text, nlp, terms, label):
    """Use spaCy PhraseMatcher for fast multi-word term matching."""
    if nlp is None:
        return []
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(t.lower()) for t in terms]
    matcher.add(label, patterns)
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        span_text = doc[start:end].text.strip()
        if span_text:
            found.add(span_text.title() if len(span_text) > 4 else span_text.upper())
    return sorted(list(found))


# ============================================================
#  SECTION 6: DOSAGE EXTRACTOR
# ============================================================

def extract_dosages(text):
    """Extract dosage info via regex: 500mg, OD, BD, for 5 days etc."""
    patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?)\b',
        r'\b(?:od|bd|tds|qid|sos|prn|stat|tid|bid)\b',
        r'\b(?:once|twice|thrice)\s+(?:daily|a\s+day)\b',
        r'\bfor\s+\d+\s+(?:days?|weeks?|months?)\b',
        r'\b\d+\s+(?:tablet|tab|capsule|cap|drop|sachet|puff)s?\b',
    ]
    found = set()
    for pattern in patterns:
        for m in re.findall(pattern, text.lower(), re.IGNORECASE):
            found.add(m.strip())
    return sorted(list(found))


# ============================================================
#  SECTION 7: DOCTOR EXTRACTOR
# ============================================================

def extract_doctors(text, spacy_doctors=None):
    """Extract doctor names via spaCy NER + Dr./Prof. regex fallback."""
    found = set()
    if spacy_doctors:
        for name in spacy_doctors:
            found.add(name.strip())
    pattern = re.compile(
        r'\b(?:Dr\.?|Doctor|Prof\.?|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        full = match.group(0).strip()
        if len(full) > 4:
            found.add(full)
    return sorted(list(found))


# ============================================================
#  SECTION 8: MAIN PIPELINE
# ============================================================

def analyze_medical_text(raw_ocr_text):
    """
    Main entry point. Takes raw OCR text, returns structured dict.
    Works with or without spaCy (degrades gracefully).
    """
    text = clean_text_for_nlp(raw_ocr_text)
    if not text:
        return _empty_result("Empty OCR text provided")

    nlp = load_nlp_model()
    spacy_used = nlp is not None

    # Dictionary extraction (always runs)
    dict_diseases    = extract_by_dictionary(text, DISEASES)
    dict_medicines   = extract_by_dictionary(text, MEDICINES)
    dict_treatments  = extract_by_dictionary(text, TREATMENTS)
    dict_symptoms    = extract_by_dictionary(text, SYMPTOMS)
    dict_tests       = extract_by_dictionary(text, TESTS)
    dict_departments = extract_by_dictionary(text, DEPARTMENTS)

    # spaCy NER
    ner_results = extract_by_spacy_ner(text, nlp) if spacy_used else {}

    # PhraseMatcher (higher precision)
    if spacy_used:
        pm_diseases   = extract_with_phrase_matcher(text, nlp, DISEASES,   "DISEASE")
        pm_medicines  = extract_with_phrase_matcher(text, nlp, MEDICINES,  "MEDICINE")
        pm_treatments = extract_with_phrase_matcher(text, nlp, TREATMENTS, "TREATMENT")
        pm_symptoms   = extract_with_phrase_matcher(text, nlp, SYMPTOMS,   "SYMPTOM")
        pm_tests      = extract_with_phrase_matcher(text, nlp, TESTS,      "TEST")
    else:
        pm_diseases = pm_medicines = pm_treatments = pm_symptoms = pm_tests = []

    # Merge all sources
    diseases    = _merge(dict_diseases,   pm_diseases)
    medicines   = _merge(dict_medicines,  pm_medicines)
    treatments  = _merge(dict_treatments, pm_treatments)
    symptoms    = _merge(dict_symptoms,   pm_symptoms)
    tests       = _merge(dict_tests,      pm_tests)
    departments = dict_departments
    doctors     = extract_doctors(text, ner_results.get("doctors", []))
    dosages     = extract_dosages(text)
    dates       = ner_results.get("dates", [])

    return {
        "diseases":    diseases,
        "medicines":   medicines,
        "treatments":  treatments,
        "symptoms":    symptoms,
        "tests":       tests,
        "doctors":     doctors,
        "dosages":     dosages,
        "departments": departments,
        "dates":       dates,
        "spacy_used":  spacy_used,
        "entity_count": sum([len(diseases), len(medicines), len(treatments),
                             len(symptoms), len(tests), len(doctors)]),
        "error": None
    }


# ============================================================
#  SECTION 9: HELPERS
# ============================================================

def _merge(*lists):
    """Merge lists, deduplicate case-insensitively, sort."""
    seen = {}
    for lst in lists:
        for item in lst:
            key = item.lower().strip()
            if key not in seen:
                seen[key] = item
    return sorted(list(seen.values()))


def _empty_result(reason):
    return {
        "diseases":[],"medicines":[],"treatments":[],
        "symptoms":[],"tests":[],"doctors":[],
        "dosages":[],"departments":[],"dates":[],
        "spacy_used":False,"entity_count":0,"error":reason
    }


def result_to_json(result):
    """Serialize result dict to JSON string for DB storage."""
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
#  STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    SAMPLE = """
    APOLLO HOSPITALS LIMITED, CHENNAI

    Patient Name  : Mr. Rajesh Kumar
    Consulting    : Dr. Priya Nair (Cardiologist)
    Admitted      : 10/03/2024   Discharged: 17/03/2024

    Diagnosis: Dengue Fever with Thrombocytopenia, Type 2 Diabetes Mellitus

    Complaints: High Fever, Severe headache, Vomiting, Body ache, Rash

    Investigations: CBC, Platelet Count, Dengue NS1, LFT, KFT, HbA1c, FBS

    Treatment:
      IV Fluids - Normal Saline, DNS
      Paracetamol 500mg TDS
      Azithromycin 500mg OD x 5 days
      Pantoprazole 40mg BD
      Insulin Lantus 10 units bedtime
      Blood Transfusion x 1 unit
    """
    result = analyze_medical_text(SAMPLE)
    print("\n" + "="*55)
    print("MEDICAL NLP RESULT")
    print("="*55)
    for key in ["diseases","medicines","treatments","symptoms","tests","doctors","dosages"]:
        vals = result[key]
        print(f"\n{key.upper()} ({len(vals)}):")
        for v in vals:
            print(f"  - {v}")
    print(f"\nspaCy used: {result['spacy_used']}")
    print(f"Total entities: {result['entity_count']}")