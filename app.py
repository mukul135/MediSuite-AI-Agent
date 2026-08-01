# ============================================================
#  MediSuite-AI-Agent — app.py  (COMPLETE — with Upload System)
#  Includes: Authentication + File Upload Feature
# ============================================================

# --- IMPORTS ------------------------------------------------
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import hashlib
import os                                   # File/folder operations
from werkzeug.utils import secure_filename  # Sanitize uploaded filenames
from extractor.claim_extractor import extract_claim_fields
import json
import pytesseract          # Python wrapper for Tesseract OCR engine
from PIL import Image       # Pillow: open and process image files
from pdf2image import convert_from_path  # Convert PDF pages → images for OCR
from extractor.medical_nlp import analyze_medical_text, result_to_json
from extractor.claim_verifier import verify_claim, build_fields_from_records
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.predict_claim import predict, is_model_ready, get_model_metrics
from ml.detect_fraud import detect_fraud, is_fraud_model_ready, get_fraud_model_stats

# ============================================================
#  1. CREATE THE FLASK APP
# ============================================================
app = Flask(__name__)

# Secret key for sessions and flash messages
app.secret_key = 'medisuite_secret_key_2024_change_this_in_production'

# ============================================================
#  2. MYSQL DATABASE CONFIGURATION
# ============================================================
app.config['MYSQL_HOST']     = 'localhost'
app.config['MYSQL_USER']     = 'root'
app.config['MYSQL_PASSWORD'] = 'mukul@0906'       # ← Change to your MySQL password
app.config['MYSQL_DB']       = 'medisuite'

mysql = MySQL(app)

# ============================================================
#  3. FILE UPLOAD CONFIGURATION
# ============================================================
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}

app.config['UPLOAD_FOLDER']       = 'uploads'
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024   # 16 MB max

# Auto-create uploads/ folder if missing
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
#  4. HELPER FUNCTIONS
# ============================================================
def hash_password(password):
    """SHA-256 one-way hash for passwords."""
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    """Returns True if the file extension is in ALLOWED_EXTENSIONS."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================
#  5. ROOT ROUTE
# ============================================================
@app.route('/')
def home():
    return redirect(url_for('login'))

# ============================================================
#  6. REGISTER ROUTE  (your existing code — unchanged)
# ============================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'loggedin' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name     = request.form['name']
        email    = request.form['email']
        password = request.form['password']

        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address.', 'danger')
            return redirect(url_for('register'))

        hashed = hash_password(password)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing = cursor.fetchone()

        if existing:
            flash('An account with this email already exists.', 'warning')
            return redirect(url_for('register'))

        cursor.execute(
            'INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
            (name, email, hashed)
        )
        mysql.connection.commit()
        cursor.close()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# ============================================================
#  7. LOGIN ROUTE  (your existing code — unchanged)
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'loggedin' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        hashed   = hash_password(password)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE email = %s AND password = %s',
            (email, hashed)
        )
        user = cursor.fetchone()
        cursor.close()

        if user:
            session['loggedin'] = True
            session['id']       = user['id']
            session['name']     = user['name']
            session['email']    = user['email']
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


# ============================================================
#  MediSuite-AI-Agent -- app.py  (Version 3.0 -- with OCR)
#  ADD these imports at the very top of your existing app.py
#  below the current imports
# ============================================================


# ============================================================
#  WINDOWS ONLY: Tell pytesseract where Tesseract is installed.
#  If you installed it in a different folder, change this path.
#  On Linux/Mac, Tesseract is on PATH — delete this line.
# ============================================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ============================================================
#  ADD THIS ROUTE to your existing app.py
#  Place it after your /upload route
# ============================================================

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    """
    OCR Route — GET and POST
    GET  : Display the OCR page with the user's uploaded files and past results.
    POST : Process the selected file, extract text, save to DB, display result.
    """

    # ── SECURITY: Redirect to login if not authenticated ─────────────────────
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    extracted_text = None   # Will hold the OCR result text
    selected_file  = None   # Which file the user chose to process

    # ── FETCH: Get all files uploaded by the current user ────────────────────
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT id, filename FROM reports WHERE uploaded_by = %s',
        (session['email'],)
    )
    reports = cursor.fetchall()  # List of dicts: [{id, filename}, ...]

    # ── POST: User submitted the form — process the selected file ────────────
    if request.method == 'POST':
        selected_file = request.form.get('filename')  # Filename from dropdown

        # Validate: make sure a file was selected
        if not selected_file:
            flash('Please select a file to process.', 'warning')
            return redirect(url_for('ocr'))

        # Build the full path to the file on disk
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)

        # Validate: make sure the file actually exists on disk
        if not os.path.exists(file_path):
            flash('File not found on disk. Please re-upload it.', 'danger')
            return redirect(url_for('ocr'))

        # Get file extension to decide how to process it
        ext = selected_file.rsplit('.', 1)[-1].lower()

        try:
            # ── CASE 1: Image files (JPG, JPEG, PNG) ─────────────────────────
            if ext in ['jpg', 'jpeg', 'png']:
                img  = Image.open(file_path)          # Open image with Pillow
                text = pytesseract.image_to_string(img)  # Run OCR

            # ── CASE 2: PDF files ─────────────────────────────────────────────
            elif ext == 'pdf':
                # pdf2image converts each PDF page into a PIL Image object
                # dpi=300 gives high enough resolution for accurate OCR
                pages = convert_from_path(file_path, dpi=300)
                text  = ''
                for page_num, page in enumerate(pages):
                    page_text = pytesseract.image_to_string(page)
                    text += f'\n--- Page {page_num + 1} ---\n{page_text}'

            else:
                # File type not supported by OCR
                flash('OCR supports JPG, PNG, and PDF files only.', 'warning')
                return redirect(url_for('ocr'))

            # ── CLEAN the extracted text ──────────────────────────────────────
            # strip() removes leading/trailing whitespace
            extracted_text = text.strip()

            if not extracted_text:
                flash('No text could be extracted. The image may be low quality.', 'warning')
            else:
                # ── SAVE result to ocr_results table ─────────────────────────
                cursor.execute(
                    '''INSERT INTO ocr_results (filename, extracted_text, processed_by)
                       VALUES (%s, %s, %s)''',
                    (selected_file, extracted_text, session['email'])
                )
                mysql.connection.commit()
                flash('Text extracted successfully!', 'success')

        except Exception as e:
            # Catch ALL errors (Tesseract not found, Poppler missing, etc.)
            # and show a human-readable message instead of crashing
            flash(f'OCR Error: {str(e)}', 'danger')

    # ── FETCH: Past OCR results for this user (newest first) ─────────────────
    cursor.execute(
        '''SELECT id, filename, extracted_text, created_at
           FROM ocr_results
           WHERE processed_by = %s
           ORDER BY id DESC
           LIMIT 10''',
        (session['email'],)
    )
    past_results = cursor.fetchall()
    cursor.close()

    # ── RENDER: Pass everything to the template ───────────────────────────────
    return render_template(
        'ocr.html',
        reports       = reports,        # dropdown options
        extracted     = extracted_text, # freshly extracted text (or None)
        selected_file = selected_file,  # which file was processed
        past_results  = past_results    # history table
    )


# ============================================================
#  MediSuite-AI-Agent -- claim_route.py
#  Phase 4: Claim Autofill Routes
#
#  ADD these imports and routes to your existing app.py
#
#  Routes added:
#    GET  /claim        -- show claim form + autofill from OCR results
#    POST /claim        -- submit/save the completed claim
#    GET  /claim/api/<id> -- return JSON extraction result for an OCR record
# ============================================================

# ── NEW IMPORTS (add at top of app.py) ───────────────────────────────────────
# from extractor.claim_extractor import extract_claim_fields
# import json


# ============================================================
#  ROUTE 1: /claim  (GET + POST)
#  Main claim form page with autofill
# ============================================================

@app.route('/claim', methods=['GET', 'POST'])
def claim():
    """
    GET:
      - Show the insurance claim form
      - If ?ocr_id=X is passed in the URL, auto-extract fields
        from that OCR result and pre-fill the form

    POST:
      - Save the completed/edited claim to the claims table
    """

    # ── Security guard ────────────────────────────────────────────────────────
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    # ── Defaults (empty form) ─────────────────────────────────────────────────
    extracted    = {}       # auto-filled values from OCR
    confidences  = {}       # confidence scores per field
    selected_ocr = None     # which OCR record was used
    ocr_id       = request.args.get('ocr_id')  # from URL: /claim?ocr_id=5

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # ── GET: autofill if ocr_id was provided ──────────────────────────────────
    if ocr_id and request.method == 'GET':
        cursor.execute(
            'SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s',
            (ocr_id, session['email'])
        )
        ocr_record = cursor.fetchone()

        if ocr_record:
            selected_ocr = ocr_record
            # Run the extraction pipeline on the stored OCR text
            result = extract_claim_fields(ocr_record['extracted_text'])
            extracted   = result
            confidences = result.get('confidence_scores', {})
        else:
            flash('OCR record not found or access denied.', 'danger')

    # ── POST: save submitted claim ────────────────────────────────────────────
    if request.method == 'POST':
        # Collect form values (user may have edited autofilled values)
        patient_name  = request.form.get('patient_name', '').strip()
        hospital_name = request.form.get('hospital_name', '').strip()
        disease       = request.form.get('disease', '').strip()
        bill_amount   = request.form.get('bill_amount', '').strip()
        claim_date    = request.form.get('claim_date', '').strip()
        policy_number = request.form.get('policy_number', '').strip()
        ocr_source_id = request.form.get('ocr_source_id', '').strip()

        # Basic validation
        if not patient_name or not hospital_name or not disease or not bill_amount:
            flash('Patient Name, Hospital Name, Disease, and Bill Amount are required.', 'warning')
            return redirect(url_for('claim'))

        try:
            # Validate bill_amount is numeric
            clean_amount = bill_amount.replace(',', '')
            float(clean_amount)
        except ValueError:
            flash('Bill Amount must be a valid number.', 'warning')
            return redirect(url_for('claim'))

        try:
            cursor.execute(
                '''INSERT INTO claims
                   (patient_name, hospital_name, disease, bill_amount,
                    claim_date, policy_number, submitted_by, ocr_source_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                (patient_name, hospital_name, disease, bill_amount,
                 claim_date, policy_number, session['email'],
                 ocr_source_id if ocr_source_id else None)
            )
            mysql.connection.commit()
            flash('Claim submitted successfully!', 'success')
            return redirect(url_for('claim_history'))
        except Exception as e:
            flash(f'Error saving claim: {str(e)}', 'danger')

    # ── Fetch user's OCR results (for "autofill from OCR" dropdown) ───────────
    cursor.execute(
        'SELECT id, filename, created_at FROM ocr_results WHERE processed_by = %s ORDER BY id DESC',
        (session['email'],)
    )
    ocr_list = cursor.fetchall()
    cursor.close()

    return render_template(
        'claim.html',
        extracted    = extracted,
        confidences  = confidences,
        selected_ocr = selected_ocr,
        ocr_list     = ocr_list,
        ocr_id       = ocr_id
    )


# ============================================================
#  ROUTE 2: /claim/history  (GET)
#  View all submitted claims for current user
# ============================================================

@app.route('/claim/history')
def claim_history():
    """Show all claims submitted by the logged-in user."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        '''SELECT * FROM claims
           WHERE submitted_by = %s
           ORDER BY id DESC''',
        (session['email'],)
    )
    claims = cursor.fetchall()
    cursor.close()

    return render_template('claim_history.html', claims=claims)


# ============================================================
#  ROUTE 3: /claim/api/<int:ocr_id>  (GET — JSON API)
#  Called by JavaScript to get autofill data without page reload
# ============================================================

@app.route('/claim/api/<int:ocr_id>')
def claim_api(ocr_id):
    """
    Returns JSON with extracted claim fields for a given OCR result ID.
    Used by the frontend for live autofill when user picks an OCR record.
    """
    if 'loggedin' not in session:
        return {'error': 'Not authenticated'}, 401

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s',
        (ocr_id, session['email'])
    )
    ocr_record = cursor.fetchone()
    cursor.close()

    if not ocr_record:
        return {'error': 'Not found'}, 404

    # Run extraction
    result = extract_claim_fields(ocr_record['extracted_text'])

    return {
        'success': True,
        'patient_name':  result['patient_name'],
        'hospital_name': result['hospital_name'],
        'disease':       result['disease'],
        'bill_amount':   result['bill_amount'],
        'confidence_scores': result['confidence_scores'],
        'extraction_details': result['extraction_details']
    }

# ============================================================
#  8.NLP Medical Analysis Routes
# ============================================================

@app.route('/medical-analysis', methods=['GET'])
def medical_analysis_home():
    """
    Show list of user's OCR results so they can pick one to analyse.
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT id, filename, created_at FROM ocr_results WHERE processed_by = %s ORDER BY id DESC',
        (session['email'],)
    )
    ocr_list = cursor.fetchall()
    cursor.close()
 
    return render_template('medical_analysis_home.html', ocr_list=ocr_list)
 
 
@app.route('/medical-analysis/<int:ocr_id>', methods=['GET'])
def medical_analysis(ocr_id):
    """
    Run NLP analysis on a specific OCR result.
 
    Workflow:
    1. Load OCR text from ocr_results table
    2. Run analyze_medical_text()
    3. Save result to medical_analysis table (or update if exists)
    4. Render medical_analysis.html with structured result
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
 
    # ── Load OCR record ───────────────────────────────────────────────────────
    cursor.execute(
        'SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s',
        (ocr_id, session['email'])
    )
    ocr_record = cursor.fetchone()
 
    if not ocr_record:
        flash('OCR record not found or access denied.', 'danger')
        return redirect(url_for('medical_analysis_home'))
 
    if not ocr_record.get('extracted_text', '').strip():
        flash('This OCR record has no extracted text to analyse.', 'warning')
        return redirect(url_for('medical_analysis_home'))
 
    # ── Run NLP analysis ──────────────────────────────────────────────────────
    try:
        analysis = analyze_medical_text(ocr_record['extracted_text'])
        analysis_json_str = result_to_json(analysis)
    except Exception as e:
        flash(f'NLP analysis failed: {str(e)}', 'danger')
        return redirect(url_for('medical_analysis_home'))
 
    # ── Save to medical_analysis table ────────────────────────────────────────
    # Check if analysis already exists for this OCR record
    cursor.execute(
        'SELECT id FROM medical_analysis WHERE ocr_result_id = %s',
        (ocr_id,)
    )
    existing = cursor.fetchone()
 
    try:
        if existing:
            # Update existing record
            cursor.execute(
                '''UPDATE medical_analysis
                   SET diseases=%s, medicines=%s, treatments=%s,
                       symptoms=%s, tests=%s, doctors=%s, analysis_json=%s
                   WHERE ocr_result_id=%s''',
                (
                    json.dumps(analysis['diseases']),
                    json.dumps(analysis['medicines']),
                    json.dumps(analysis['treatments']),
                    json.dumps(analysis['symptoms']),
                    json.dumps(analysis['tests']),
                    json.dumps(analysis['doctors']),
                    analysis_json_str,
                    ocr_id
                )
            )
        else:
            # Insert new record
            cursor.execute(
                '''INSERT INTO medical_analysis
                   (ocr_result_id, diseases, medicines, treatments,
                    symptoms, tests, doctors, analysis_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                (
                    ocr_id,
                    json.dumps(analysis['diseases']),
                    json.dumps(analysis['medicines']),
                    json.dumps(analysis['treatments']),
                    json.dumps(analysis['symptoms']),
                    json.dumps(analysis['tests']),
                    json.dumps(analysis['doctors']),
                    analysis_json_str
                )
            )
        mysql.connection.commit()
    except Exception as e:
        flash(f'Database error while saving analysis: {str(e)}', 'danger')
 
    cursor.close()
 
    return render_template(
        'medical_analysis.html',
        ocr_record = ocr_record,
        analysis   = analysis,
        ocr_id     = ocr_id
    )
 
 
@app.route('/medical-analysis/history')
def medical_analysis_history():
    """
    Show all past NLP analyses for the logged-in user.
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        '''SELECT ma.id, ma.ocr_result_id, ma.created_at,
                  ma.diseases, ma.medicines, ma.treatments,
                  ocr.filename
           FROM medical_analysis ma
           JOIN ocr_results ocr ON ma.ocr_result_id = ocr.id
           WHERE ocr.processed_by = %s
           ORDER BY ma.id DESC''',
        (session['email'],)
    )
    analyses = cursor.fetchall()
    cursor.close()
 
    # Parse JSON strings back to lists for display
    for row in analyses:
        for field in ['diseases', 'medicines', 'treatments']:
            try:
                row[field] = json.loads(row[field]) if row[field] else []
            except:
                row[field] = []
 
    return render_template('medical_analysis_history.html', analyses=analyses)


# ============================================================
#  8. CLAIM VERIFICATION ROUTES 
# ============================================================

@app.route('/claim-verification', methods=['GET', 'POST'])
def claim_verification():
    """
    GET:  Show the verification form, pre-populated from a claim
          if ?claim_id=X is passed, or blank if manual entry.
    POST: User submits the form → run verify_claim() → save → show result.
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    result        = None   # verification result dict
    prefill       = {}     # values to pre-fill the form
    selected_claim = None  # the claim record used (if any)
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
 
    # ── GET: pre-fill from a claim record if claim_id passed in URL ───────────
    claim_id = request.args.get('claim_id')
    if claim_id and request.method == 'GET':
        cursor.execute(
            'SELECT * FROM claims WHERE id = %s AND submitted_by = %s',
            (claim_id, session['email'])
        )
        selected_claim = cursor.fetchone()
        if selected_claim:
            prefill = build_fields_from_records(claim_record=selected_claim)
 
    # ── POST: run verification ────────────────────────────────────────────────
    if request.method == 'POST':
        # Build fields from submitted form data
        form_data = {
            "patient_name":   request.form.get('patient_name', '').strip(),
            "hospital_name":  request.form.get('hospital_name', '').strip(),
            "disease":        request.form.get('disease', '').strip(),
            "bill_amount":    request.form.get('bill_amount', '').strip(),
            "insurance_id":   request.form.get('insurance_id', '').strip(),
            "policy_number":  request.form.get('policy_number', '').strip(),
            "admission_date": request.form.get('admission_date', '').strip(),
            "discharge_date": request.form.get('discharge_date', '').strip(),
            "doctor_name":    request.form.get('doctor_name', '').strip(),
        }
        claim_id_form = request.form.get('claim_id', '').strip()
 
        # If a saved claim was linked, load it too
        claim_record = None
        if claim_id_form:
            cursor.execute(
                'SELECT * FROM claims WHERE id = %s AND submitted_by = %s',
                (claim_id_form, session['email'])
            )
            claim_record = cursor.fetchone()
 
        fields = build_fields_from_records(
            claim_record=claim_record,
            form_data=form_data
        )
 
        # Run the verification engine
        try:
            result = verify_claim(fields)
        except Exception as e:
            flash(f'Verification error: {str(e)}', 'danger')
            return redirect(url_for('claim_verification'))
 
        # Save to claim_verification table
        try:
            # Check if a verification already exists for this claim
            existing_id = None
            if claim_id_form:
                cursor.execute(
                    'SELECT verification_id FROM claim_verification WHERE claim_id = %s AND verified_by = %s',
                    (claim_id_form, session['email'])
                )
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing['verification_id']
 
            if existing_id:
                cursor.execute(
                    '''UPDATE claim_verification
                       SET verification_status=%s, verification_score=%s,
                           missing_fields=%s, failed_rules=%s,
                           passed_rules=%s, remarks=%s,
                           input_fields=%s, created_at=NOW()
                       WHERE verification_id=%s''',
                    (
                        result['status'],
                        result['score'],
                        json.dumps(result['missing_fields']),
                        json.dumps(result['failed_rules']),
                        json.dumps(result['passed_rules']),
                        result['remarks'],
                        json.dumps(fields),
                        existing_id
                    )
                )
            else:
                cursor.execute(
                    '''INSERT INTO claim_verification
                       (claim_id, verification_status, verification_score,
                        missing_fields, failed_rules, passed_rules,
                        remarks, input_fields, verified_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (
                        int(claim_id_form) if claim_id_form else None,
                        result['status'],
                        result['score'],
                        json.dumps(result['missing_fields']),
                        json.dumps(result['failed_rules']),
                        json.dumps(result['passed_rules']),
                        result['remarks'],
                        json.dumps(fields),
                        session['email']
                    )
                )
            mysql.connection.commit()
        except Exception as e:
            flash(f'Could not save verification result: {str(e)}', 'warning')
 
        # Keep form data for re-display
        prefill = fields
 
    # ── Fetch user's past claims for the dropdown ──────────────────────────────
    cursor.execute(
        '''SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by = %s ORDER BY id DESC LIMIT 20''',
        (session['email'],)
    )
    claims_list = cursor.fetchall()
    cursor.close()
 
    return render_template(
        'claim_verification.html',
        result       = result,
        prefill      = prefill,
        claims_list  = claims_list,
        claim_id     = claim_id or request.form.get('claim_id', '')
    )
 
 
@app.route('/claim-verification/history')
def claim_verification_history():
    """Show all past verifications for the logged-in user."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        '''SELECT cv.*, c.patient_name, c.hospital_name
           FROM claim_verification cv
           LEFT JOIN claims c ON cv.claim_id = c.id
           WHERE cv.verified_by = %s
           ORDER BY cv.verification_id DESC''',
        (session['email'],)
    )
    verifications = cursor.fetchall()
    cursor.close()
 
    # Parse JSON columns
    for row in verifications:
        for col in ['missing_fields', 'failed_rules', 'passed_rules']:
            try:
                row[col] = json.loads(row[col]) if row[col] else []
            except:
                row[col] = []
 
    return render_template('claim_verification_history.html', verifications=verifications)
 
# ============================================================
#  8. AI Claim Approval Prediction Routes
# ============================================================


@app.route('/claim-prediction', methods=['GET', 'POST'])
def claim_prediction():
    """
    GET:  Show prediction form. If ?claim_id=X, pre-fill from saved claim
          and also load latest verification result for that claim.
    POST: Run prediction with submitted form data.
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    result         = None
    prefill        = {}
    model_ready    = is_model_ready()
    model_metrics  = get_model_metrics()
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
 
    claim_id = request.args.get('claim_id', '')
 
    # ── GET: pre-fill from claim + verification record ─────────────────────────
    if claim_id and request.method == 'GET':
        # Load claim
        cursor.execute(
            'SELECT * FROM claims WHERE id = %s AND submitted_by = %s',
            (claim_id, session['email'])
        )
        claim_rec = cursor.fetchone()
        if claim_rec:
            prefill.update({
                "patient_name":   claim_rec.get("patient_name", ""),
                "hospital_name":  claim_rec.get("hospital_name", ""),
                "disease":        claim_rec.get("disease", ""),
                "bill_amount":    claim_rec.get("bill_amount", ""),
                "policy_number":  claim_rec.get("policy_number", ""),
            })
 
        # Load latest verification for this claim
        cursor.execute(
            '''SELECT verification_score, verification_status
               FROM claim_verification
               WHERE claim_id = %s
               ORDER BY verification_id DESC LIMIT 1''',
            (claim_id,)
        )
        ver_rec = cursor.fetchone()
        if ver_rec:
            prefill["verification_score"]  = ver_rec.get("verification_score", 50)
            prefill["verification_status"] = ver_rec.get("verification_status", "")
 
    # ── POST: run prediction ───────────────────────────────────────────────────
    if request.method == 'POST':
        if not model_ready:
            flash('AI model not trained yet. Run: python ml/train_model.py', 'danger')
            return redirect(url_for('claim_prediction'))
 
        claim_id = request.form.get('claim_id', '').strip()
 
        # Collect form data
        claim_data = {
            "patient_age":        request.form.get('patient_age', '35').strip(),
            "gender":             request.form.get('gender', 'male').strip(),
            "hospital_type":      request.form.get('hospital_type', 'private').strip(),
            "disease":            request.form.get('disease', '').strip(),
            "bill_amount":        request.form.get('bill_amount', '0').strip(),
            "admission_date":     request.form.get('admission_date', '').strip(),
            "discharge_date":     request.form.get('discharge_date', '').strip(),
            "insurance_id":       request.form.get('insurance_id', '').strip(),
            "policy_number":      request.form.get('policy_number', '').strip(),
            "verification_score": request.form.get('verification_score', '50').strip(),
            "verification_status":request.form.get('verification_status', '').strip(),
            "previous_claims":    request.form.get('previous_claims', '0').strip(),
            "fraud_flag":         request.form.get('fraud_flag', '0').strip(),
        }
        prefill = claim_data
 
        # Run prediction
        try:
            result = predict(claim_data)
        except FileNotFoundError as e:
            flash(str(e), 'danger')
            return redirect(url_for('claim_prediction'))
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
            return redirect(url_for('claim_prediction'))
 
        # Get verification_id if claim is linked
        ver_id = None
        if claim_id:
            try:
                cursor.execute(
                    'SELECT verification_id FROM claim_verification WHERE claim_id=%s ORDER BY verification_id DESC LIMIT 1',
                    (claim_id,)
                )
                vr = cursor.fetchone()
                if vr:
                    ver_id = vr['verification_id']
            except:
                pass
 
        # Save prediction to DB
        try:
            # Avoid duplicates: update if exists for same claim
            existing_pred_id = None
            if claim_id:
                cursor.execute(
                    'SELECT prediction_id FROM claim_prediction WHERE claim_id=%s AND predicted_by=%s',
                    (claim_id, session['email'])
                )
                ep = cursor.fetchone()
                if ep:
                    existing_pred_id = ep['prediction_id']
 
            if existing_pred_id:
                cursor.execute(
                    '''UPDATE claim_prediction
                       SET prediction=%s, approval_probability=%s,
                           rejection_probability=%s, confidence=%s,
                           model_name=%s, input_features=%s,
                           verification_id=%s, created_at=NOW()
                       WHERE prediction_id=%s''',
                    (
                        result['prediction'],
                        result['approval_probability'],
                        result['rejection_probability'],
                        result['confidence'],
                        result['model_name'],
                        json.dumps(claim_data),
                        ver_id,
                        existing_pred_id
                    )
                )
            else:
                cursor.execute(
                    '''INSERT INTO claim_prediction
                       (claim_id, verification_id, prediction,
                        approval_probability, rejection_probability,
                        confidence, model_name, input_features, predicted_by)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (
                        int(claim_id) if claim_id else None,
                        ver_id,
                        result['prediction'],
                        result['approval_probability'],
                        result['rejection_probability'],
                        result['confidence'],
                        result['model_name'],
                        json.dumps(claim_data),
                        session['email']
                    )
                )
            mysql.connection.commit()
        except Exception as e:
            flash(f'Could not save prediction: {str(e)}', 'warning')
 
    # ── Load claims list for dropdown ──────────────────────────────────────────
    cursor.execute(
        '''SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by = %s ORDER BY id DESC LIMIT 20''',
        (session['email'],)
    )
    claims_list = cursor.fetchall()
    cursor.close()
 
    return render_template(
        'claim_prediction.html',
        result        = result,
        prefill       = prefill,
        claims_list   = claims_list,
        claim_id      = claim_id,
        model_ready   = model_ready,
        model_metrics = model_metrics,
    )
 
 
@app.route('/claim-prediction/history')
def claim_prediction_history():
    """Show all past predictions for the logged-in user."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        '''SELECT cp.*, c.patient_name, c.hospital_name, c.disease
           FROM claim_prediction cp
           LEFT JOIN claims c ON cp.claim_id = c.id
           WHERE cp.predicted_by = %s
           ORDER BY cp.prediction_id DESC''',
        (session['email'],)
    )
    predictions = cursor.fetchall()
    cursor.close()
 
    return render_template('claim_prediction_history.html', predictions=predictions)
 

# ============================================================
#  8. FRAUD DETECTION
# ============================================================

@app.route('/claim-fraud', methods=['GET', 'POST'])
def claim_fraud():
    """
    GET:  Show form. If ?claim_id=X, pre-fill from saved claim
          + load verification and prediction data automatically.
    POST: Run fraud detection. Save result to DB.
    """
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    result       = None
    prefill      = {}
    model_ready  = is_fraud_model_ready()
    model_stats  = get_fraud_model_stats()
    claim_id     = request.args.get('claim_id', '')
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
 
    # ── GET: load from saved claim + verification + prediction ─────────────────
    if claim_id and request.method == 'GET':
        cursor.execute(
            'SELECT * FROM claims WHERE id=%s AND submitted_by=%s',
            (claim_id, session['email'])
        )
        claim_rec = cursor.fetchone()
        if claim_rec:
            prefill.update({
                "patient_name":  claim_rec.get("patient_name", ""),
                "hospital_name": claim_rec.get("hospital_name", ""),
                "disease":       claim_rec.get("disease", ""),
                "bill_amount":   claim_rec.get("bill_amount", ""),
                "policy_number": claim_rec.get("policy_number", ""),
            })
 
        # Load verification score and status
        cursor.execute(
            '''SELECT verification_score, verification_status
               FROM claim_verification WHERE claim_id=%s
               ORDER BY verification_id DESC LIMIT 1''',
            (claim_id,))
        ver = cursor.fetchone()
        if ver:
            prefill["verification_score"]  = ver.get("verification_score", 50)
            prefill["verification_status"] = ver.get("verification_status", "")
 
        # Load AI prediction probability
        cursor.execute(
            '''SELECT approval_probability FROM claim_prediction
               WHERE claim_id=%s ORDER BY prediction_id DESC LIMIT 1''',
            (claim_id,))
        pred = cursor.fetchone()
        if pred:
            prefill["approval_probability"] = pred.get("approval_probability", 50)
 
        # Load past claims for duplicate check (FIXED QUERY)
        cursor.execute(
            '''SELECT id, bill_amount
               FROM claims
               WHERE submitted_by = %s AND id != %s''',
            (session['email'], claim_id))
        prefill["past_claims_summary"] = cursor.fetchall() or []
 
        # Claim frequency in last 7 days
        cursor.execute(
            '''SELECT COUNT(*) AS cnt FROM claims
               WHERE submitted_by=%s
               AND submitted_at >= NOW() - INTERVAL 7 DAY''',
            (session['email'],))
        freq_row = cursor.fetchone()
        prefill["claim_frequency_7d"] = freq_row['cnt'] if freq_row else 0
 
    # ── POST: run fraud detection ──────────────────────────────────────────────
    if request.method == 'POST':
        claim_id = request.form.get('claim_id', '').strip()
 
        claim_data = {
            "bill_amount":          request.form.get('bill_amount', '0').strip(),
            "admission_date":       request.form.get('admission_date', '').strip(),
            "discharge_date":       request.form.get('discharge_date', '').strip(),
            "insurance_id":         request.form.get('insurance_id', '').strip(),
            "policy_number":        request.form.get('policy_number', '').strip(),
            "doctor_name":          request.form.get('doctor_name', '').strip(),
            "disease":              request.form.get('disease', '').strip(),
            "verification_score":   request.form.get('verification_score', '50').strip(),
            "verification_status":  request.form.get('verification_status', '').strip(),
            "approval_probability": request.form.get('approval_probability', '50').strip(),
            "previous_claims":      request.form.get('previous_claims', '0').strip(),
            "claim_frequency_7d":   request.form.get('claim_frequency_7d', '0').strip(),
        }
        prefill = claim_data
 
        try:
            result = detect_fraud(claim_data)
        except Exception as e:
            flash(f'Fraud detection error: {str(e)}', 'danger')
            return redirect(url_for('claim_fraud'))
 
        # Get linked IDs from previous features
        ver_id = pred_id = None
        if claim_id:
            try:
                cursor.execute(
                    '''SELECT verification_id FROM claim_verification
                       WHERE claim_id=%s ORDER BY verification_id DESC LIMIT 1''',
                    (claim_id,))
                vr = cursor.fetchone()
                if vr:
                    ver_id = vr['verification_id']
 
                cursor.execute(
                    '''SELECT prediction_id FROM claim_prediction
                       WHERE claim_id=%s ORDER BY prediction_id DESC LIMIT 1''',
                    (claim_id,))
                pr = cursor.fetchone()
                if pr:
                    pred_id = pr['prediction_id']
            except:
                pass
 
        # Save to DB (upsert — update if exists, insert if new)
        try:
            existing_id = None
            if claim_id:
                cursor.execute(
                    '''SELECT fraud_id FROM fraud_detection
                       WHERE claim_id=%s AND detected_by=%s''',
                    (claim_id, session['email']))
                ex = cursor.fetchone()
                if ex:
                    existing_id = ex['fraud_id']
 
            if existing_id:
                cursor.execute(
                    '''UPDATE fraud_detection
                       SET fraud_status=%s, fraud_score=%s,
                           fraud_probability=%s, detected_rules=%s,
                           anomaly_detected=%s, recommendation=%s,
                           model_name=%s, created_at=NOW()
                       WHERE fraud_id=%s''',
                    (result['fraud_status'], result['fraud_score'],
                     result['fraud_probability'],
                     json.dumps(result['detected_rules']),
                     int(result['anomaly_detected']),
                     result['recommendation'], result['model_name'],
                     existing_id))
            else:
                cursor.execute(
                    '''INSERT INTO fraud_detection
                       (claim_id, verification_id, prediction_id,
                        fraud_status, fraud_score, fraud_probability,
                        detected_rules, anomaly_detected,
                        recommendation, model_name, detected_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (int(claim_id) if claim_id else None,
                     ver_id, pred_id,
                     result['fraud_status'], result['fraud_score'],
                     result['fraud_probability'],
                     json.dumps(result['detected_rules']),
                     int(result['anomaly_detected']),
                     result['recommendation'], result['model_name'],
                     session['email']))
            mysql.connection.commit()
 
        except Exception as e:
            flash(f'Could not save fraud result: {str(e)}', 'warning')
 
    # Fetch claims list for dropdown
    cursor.execute(
        '''SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by=%s ORDER BY id DESC LIMIT 20''',
        (session['email'],))
    claims_list = cursor.fetchall()
    cursor.close()
 
    return render_template(
        'fraud_detection.html',
        result      = result,
        prefill     = prefill,
        claims_list = claims_list,
        claim_id    = claim_id,
        model_ready = model_ready,
        model_stats = model_stats,
    )
 
 
@app.route('/claim-fraud/history')
def claim_fraud_history():
    """Show all past fraud analyses for logged-in user."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
 
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        '''SELECT fd.*, c.patient_name, c.hospital_name, c.disease
           FROM fraud_detection fd
           LEFT JOIN claims c ON fd.claim_id = c.id
           WHERE fd.detected_by=%s
           ORDER BY fd.fraud_id DESC''',
        (session['email'],))
    records = cursor.fetchall()
    cursor.close()
 
    for r in records:
        try:
            r['detected_rules'] = json.loads(r['detected_rules']) if r.get('detected_rules') else []
        except:
            r['detected_rules'] = []
 
    return render_template('fraud_history.html', records=records)



# ============================================================
#  8. DASHBOARD ROUTE  (your existing code — unchanged)
# ============================================================
@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    return render_template(
        'dashboard.html',
        name=session['name'],
        email=session['email']
    )

# ============================================================
#  9. UPLOAD ROUTE  ← NEW
#  URL : http://127.0.0.1:5000/upload
#  GET : Shows the upload form + list of user's past uploads
#  POST: Saves file to uploads/ folder + records in DB
# ============================================================
@app.route('/upload', methods=['GET', 'POST'])
def upload():

    # Guard: redirect to login if not authenticated
    if 'loggedin' not in session:
        flash('Please log in to upload documents.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':

        # Step 1: Was a file field present in the form?
        if 'document' not in request.files:
            flash('No file part found in the form.', 'danger')
            return redirect(url_for('upload'))

        file = request.files['document']

        # Step 2: Did the user actually pick a file?
        if file.filename == '':
            flash('No file selected. Please choose a file first.', 'warning')
            return redirect(url_for('upload'))

        # Step 3: Is the extension allowed?
        if not allowed_file(file.filename):
            flash(
                'File type not allowed. '
                'Please upload PDF, images (JPG/PNG/GIF), or Word documents.',
                'danger'
            )
            return redirect(url_for('upload'))

        # Step 4: Sanitize the filename
        # secure_filename removes path separators and dangerous characters
        filename = secure_filename(file.filename)

        # Step 5: Save file to uploads/ folder on disk
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Step 6: Record upload in the database
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                'INSERT INTO reports (filename, uploaded_by) VALUES (%s, %s)',
                (filename, session['email'])
            )
            mysql.connection.commit()
            cursor.close()

            flash(f'"{filename}" uploaded successfully!', 'success')
            return redirect(url_for('upload'))

        except Exception as db_error:
            flash(f'Database error: {str(db_error)}', 'danger')
            return redirect(url_for('upload'))

    # --- GET: load this user's past uploads to display ---
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT * FROM reports WHERE uploaded_by = %s ORDER BY id DESC',
        (session['email'],)
    )
    reports = cursor.fetchall()
    cursor.close()

    return render_template('upload.html', reports=reports)

# ============================================================
#  10. LOGOUT ROUTE
# ============================================================
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# ============================================================
#  11. RUN THE APP
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)