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