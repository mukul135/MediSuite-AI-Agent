# ============================================================
#  MediSuite-AI-Agent — app.py
#  Description: Complete Flask application handling Auth, OCR,
#               Medical NLP, Claim Verification, ML Prediction,
#               Fraud Detection, and Summarization.
# ============================================================

# --- 1. STANDARD LIBRARY IMPORTS ----------------------------
import hashlib
import json
import os
import re
import sys

# Maintain path configuration for internal modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 2. THIRD-PARTY IMPORTS ---------------------------------
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from werkzeug.utils import secure_filename

# --- 3. INTERNAL MODULE IMPORTS -----------------------------
from ai.model_loader import get_model_name, is_model_loaded
from ai.medical_summarizer import summarize
from extractor.claim_extractor import extract_claim_fields
from extractor.claim_verifier import build_fields_from_records, verify_claim
from extractor.medical_nlp import analyze_medical_text, result_to_json
from ml.detect_fraud import detect_fraud, get_fraud_model_stats, is_fraud_model_ready
from ml.predict_claim import get_model_metrics, is_model_ready, predict

# ============================================================
#  APPLICATION & SYSTEM CONFIGURATION
# ============================================================
app = Flask(__name__)

# Secret Key Configuration
app.secret_key = "medisuite_secret_key_2024_change_this_in_production"

# MySQL Database Configuration
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "mukul@0906"  # Database Password
app.config["MYSQL_DB"] = "medisuite"

# File Upload Configuration
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "gif", "doc", "docx"}
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB Limit

# Initialize Database & Upload Directories
mysql = MySQL(app)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# System-specific Binary Paths (Windows configuration)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def hash_password(password: str) -> str:
    """Generate SHA-256 one-way hash for passwords."""
    return hashlib.sha256(password.encode()).hexdigest()


def allowed_file(filename: str) -> bool:
    """Check if the provided filename contains an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
#  AUTHENTICATION & NAVIGATION ROUTES
# ============================================================
@app.route("/")
def home():
    """Root route redirecting to login."""
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """User Registration Route."""
    if "loggedin" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        # Validate Email Format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address.", "danger")
            return redirect(url_for("register"))

        hashed_pwd = hash_password(password)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("An account with this email already exists.", "warning")
            return redirect(url_for("register"))

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_pwd),
        )
        mysql.connection.commit()
        cursor.close()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User Login Route."""
    if "loggedin" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        hashed_pwd = hash_password(password)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND password = %s",
            (email, hashed_pwd),
        )
        user = cursor.fetchone()
        cursor.close()

        if user:
            session["loggedin"] = True
            session["id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Incorrect email or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Main Application Dashboard."""
    if "loggedin" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    return render_template("dashboard.html", name=session["name"], email=session["email"])


@app.route("/logout")
def logout():
    """User Logout Session Clear."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# ============================================================
#  FILE MANAGEMENT ROUTES
# ============================================================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload documents (PDF, Images, Word docs) to local storage."""
    if "loggedin" not in session:
        flash("Please log in to upload documents.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "document" not in request.files:
            flash("No file part found in the form.", "danger")
            return redirect(url_for("upload"))

        file = request.files["document"]

        if file.filename == "":
            flash("No file selected. Please choose a file first.", "warning")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash(
                "File type not allowed. Please upload PDF, images (JPG/PNG/GIF), or Word documents.",
                "danger",
            )
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                "INSERT INTO reports (filename, uploaded_by) VALUES (%s, %s)",
                (filename, session["email"]),
            )
            mysql.connection.commit()
            cursor.close()

            flash(f'"{filename}" uploaded successfully!', "success")
            return redirect(url_for("upload"))

        except Exception as db_error:
            flash(f"Database error: {str(db_error)}", "danger")
            return redirect(url_for("upload"))

    # GET Request Logic
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT * FROM reports WHERE uploaded_by = %s ORDER BY id DESC",
        (session["email"],),
    )
    reports = cursor.fetchall()
    cursor.close()

    return render_template("upload.html", reports=reports)


# ============================================================
#  OCR PROCESSING ROUTE
# ============================================================
@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    """Process uploaded images/PDFs using Tesseract OCR."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    extracted_text = None
    selected_file = None

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT id, filename FROM reports WHERE uploaded_by = %s",
        (session["email"],),
    )
    reports = cursor.fetchall()

    if request.method == "POST":
        selected_file = request.form.get("filename")

        if not selected_file:
            flash("Please select a file to process.", "warning")
            return redirect(url_for("ocr"))

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], selected_file)

        if not os.path.exists(file_path):
            flash("File not found on disk. Please re-upload it.", "danger")
            return redirect(url_for("ocr"))

        ext = selected_file.rsplit(".", 1)[-1].lower()

        try:
            # Process Image Formats
            if ext in ["jpg", "jpeg", "png"]:
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)

            # Process PDF Formats
            elif ext == "pdf":
                pages = convert_from_path(file_path, dpi=300)
                text = ""
                for page_num, page in enumerate(pages):
                    page_text = pytesseract.image_to_string(page)
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            else:
                flash("OCR supports JPG, PNG, and PDF files only.", "warning")
                return redirect(url_for("ocr"))

            extracted_text = text.strip()

            if not extracted_text:
                flash("No text could be extracted. The image may be low quality.", "warning")
            else:
                cursor.execute(
                    """INSERT INTO ocr_results (filename, extracted_text, processed_by)
                       VALUES (%s, %s, %s)""",
                    (selected_file, extracted_text, session["email"]),
                )
                mysql.connection.commit()
                flash("Text extracted successfully!", "success")

        except Exception as e:
            flash(f"OCR Error: {str(e)}", "danger")

    cursor.execute(
        """SELECT id, filename, extracted_text, created_at
           FROM ocr_results
           WHERE processed_by = %s
           ORDER BY id DESC
           LIMIT 10""",
        (session["email"],),
    )
    past_results = cursor.fetchall()
    cursor.close()

    return render_template(
        "ocr.html",
        reports=reports,
        extracted=extracted_text,
        selected_file=selected_file,
        past_results=past_results,
    )


# ============================================================
#  CLAIM HANDLING & API ROUTES
# ============================================================
@app.route("/claim", methods=["GET", "POST"])
def claim():
    """Show insurance claim form and pre-fill fields using extracted OCR data."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    extracted = {}
    confidences = {}
    selected_ocr = None
    ocr_id = request.args.get("ocr_id")

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Pre-fill OCR Data via GET Request
    if ocr_id and request.method == "GET":
        cursor.execute(
            "SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s",
            (ocr_id, session["email"]),
        )
        ocr_record = cursor.fetchone()

        if ocr_record:
            selected_ocr = ocr_record
            result = extract_claim_fields(ocr_record["extracted_text"])
            extracted = result
            confidences = result.get("confidence_scores", {})
        else:
            flash("OCR record not found or access denied.", "danger")

    # Form Submission via POST
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        hospital_name = request.form.get("hospital_name", "").strip()
        disease = request.form.get("disease", "").strip()
        bill_amount = request.form.get("bill_amount", "").strip()
        claim_date = request.form.get("claim_date", "").strip()
        policy_number = request.form.get("policy_number", "").strip()
        ocr_source_id = request.form.get("ocr_source_id", "").strip()

        if not patient_name or not hospital_name or not disease or not bill_amount:
            flash(
                "Patient Name, Hospital Name, Disease, and Bill Amount are required.",
                "warning",
            )
            return redirect(url_for("claim"))

        try:
            clean_amount = bill_amount.replace(",", "")
            float(clean_amount)
        except ValueError:
            flash("Bill Amount must be a valid number.", "warning")
            return redirect(url_for("claim"))

        try:
            cursor.execute(
                """INSERT INTO claims
                   (patient_name, hospital_name, disease, bill_amount,
                    claim_date, policy_number, submitted_by, ocr_source_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    patient_name,
                    hospital_name,
                    disease,
                    bill_amount,
                    claim_date,
                    policy_number,
                    session["email"],
                    ocr_source_id if ocr_source_id else None,
                ),
            )
            mysql.connection.commit()
            flash("Claim submitted successfully!", "success")
            return redirect(url_for("claim_history"))
        except Exception as e:
            flash(f"Error saving claim: {str(e)}", "danger")

    cursor.execute(
        """SELECT id, filename, created_at 
           FROM ocr_results 
           WHERE processed_by = %s 
           ORDER BY id DESC""",
        (session["email"],),
    )
    ocr_list = cursor.fetchall()
    cursor.close()

    return render_template(
        "claim.html",
        extracted=extracted,
        confidences=confidences,
        selected_ocr=selected_ocr,
        ocr_list=ocr_list,
        ocr_id=ocr_id,
    )


@app.route("/claim/history")
def claim_history():
    """Display all claims submitted by the active session user."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT * FROM claims WHERE submitted_by = %s ORDER BY id DESC",
        (session["email"],),
    )
    claims = cursor.fetchall()
    cursor.close()

    return render_template("claim_history.html", claims=claims)


@app.route("/claim/api/<int:ocr_id>")
def claim_api(ocr_id):
    """JSON API endpoint supplying raw extracted field data for live frontend auto-fills."""
    if "loggedin" not in session:
        return {"error": "Not authenticated"}, 401

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s",
        (ocr_id, session["email"]),
    )
    ocr_record = cursor.fetchone()
    cursor.close()

    if not ocr_record:
        return {"error": "Not found"}, 404

    result = extract_claim_fields(ocr_record["extracted_text"])

    return {
        "success": True,
        "patient_name": result["patient_name"],
        "hospital_name": result["hospital_name"],
        "disease": result["disease"],
        "bill_amount": result["bill_amount"],
        "confidence_scores": result["confidence_scores"],
        "extraction_details": result["extraction_details"],
    }


# ============================================================
#  NLP MEDICAL ANALYSIS ROUTES
# ============================================================
@app.route("/medical-analysis", methods=["GET"])
def medical_analysis_home():
    """Display list of available OCR results for medical NLP analysis."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT id, filename, created_at 
           FROM ocr_results 
           WHERE processed_by = %s 
           ORDER BY id DESC""",
        (session["email"],),
    )
    ocr_list = cursor.fetchall()
    cursor.close()

    return render_template("medical_analysis_home.html", ocr_list=ocr_list)


@app.route("/medical-analysis/<int:ocr_id>", methods=["GET"])
def medical_analysis(ocr_id):
    """Execute NLP entity extraction over a target OCR text document."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT * FROM ocr_results WHERE id = %s AND processed_by = %s",
        (ocr_id, session["email"]),
    )
    ocr_record = cursor.fetchone()

    if not ocr_record:
        flash("OCR record not found or access denied.", "danger")
        return redirect(url_for("medical_analysis_home"))

    if not ocr_record.get("extracted_text", "").strip():
        flash("This OCR record has no extracted text to analyse.", "warning")
        return redirect(url_for("medical_analysis_home"))

    try:
        analysis = analyze_medical_text(ocr_record["extracted_text"])
        analysis_json_str = result_to_json(analysis)
    except Exception as e:
        flash(f"NLP analysis failed: {str(e)}", "danger")
        return redirect(url_for("medical_analysis_home"))

    # Upsert Logic: Update if duplicate analysis exists; otherwise insert
    cursor.execute("SELECT id FROM medical_analysis WHERE ocr_result_id = %s", (ocr_id,))
    existing = cursor.fetchone()

    try:
        if existing:
            cursor.execute(
                """UPDATE medical_analysis
                   SET diseases=%s, medicines=%s, treatments=%s,
                       symptoms=%s, tests=%s, doctors=%s, analysis_json=%s
                   WHERE ocr_result_id=%s""",
                (
                    json.dumps(analysis["diseases"]),
                    json.dumps(analysis["medicines"]),
                    json.dumps(analysis["treatments"]),
                    json.dumps(analysis["symptoms"]),
                    json.dumps(analysis["tests"]),
                    json.dumps(analysis["doctors"]),
                    analysis_json_str,
                    ocr_id,
                ),
            )
        else:
            cursor.execute(
                """INSERT INTO medical_analysis
                   (ocr_result_id, diseases, medicines, treatments,
                    symptoms, tests, doctors, analysis_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    ocr_id,
                    json.dumps(analysis["diseases"]),
                    json.dumps(analysis["medicines"]),
                    json.dumps(analysis["treatments"]),
                    json.dumps(analysis["symptoms"]),
                    json.dumps(analysis["tests"]),
                    json.dumps(analysis["doctors"]),
                    analysis_json_str,
                ),
            )
        mysql.connection.commit()
    except Exception as e:
        flash(f"Database error while saving analysis: {str(e)}", "danger")

    cursor.close()

    return render_template(
        "medical_analysis.html",
        ocr_record=ocr_record,
        analysis=analysis,
        ocr_id=ocr_id,
    )


@app.route("/medical-analysis/history")
def medical_analysis_history():
    """Retrieve historical medical NLP analysis records."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT ma.id, ma.ocr_result_id, ma.created_at,
                  ma.diseases, ma.medicines, ma.treatments,
                  ocr.filename
           FROM medical_analysis ma
           JOIN ocr_results ocr ON ma.ocr_result_id = ocr.id
           WHERE ocr.processed_by = %s
           ORDER BY ma.id DESC""",
        (session["email"],),
    )
    analyses = cursor.fetchall()
    cursor.close()

    # De-serialize JSON strings back to lists
    for row in analyses:
        for field in ["diseases", "medicines", "treatments"]:
            try:
                row[field] = json.loads(row[field]) if row[field] else []
            except Exception:
                row[field] = []

    return render_template("medical_analysis_history.html", analyses=analyses)


# ============================================================
#  CLAIM VERIFICATION ROUTES
# ============================================================
@app.route("/claim-verification", methods=["GET", "POST"])
def claim_verification():
    """Run verification check rules against submitted claim parameters."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    result = None
    prefill = {}
    selected_claim = None

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    claim_id = request.args.get("claim_id")
    if claim_id and request.method == "GET":
        cursor.execute(
            "SELECT * FROM claims WHERE id = %s AND submitted_by = %s",
            (claim_id, session["email"]),
        )
        selected_claim = cursor.fetchone()
        if selected_claim:
            prefill = build_fields_from_records(claim_record=selected_claim)

    if request.method == "POST":
        form_data = {
            "patient_name": request.form.get("patient_name", "").strip(),
            "hospital_name": request.form.get("hospital_name", "").strip(),
            "disease": request.form.get("disease", "").strip(),
            "bill_amount": request.form.get("bill_amount", "").strip(),
            "insurance_id": request.form.get("insurance_id", "").strip(),
            "policy_number": request.form.get("policy_number", "").strip(),
            "admission_date": request.form.get("admission_date", "").strip(),
            "discharge_date": request.form.get("discharge_date", "").strip(),
            "doctor_name": request.form.get("doctor_name", "").strip(),
        }
        claim_id_form = request.form.get("claim_id", "").strip()

        claim_record = None
        if claim_id_form:
            cursor.execute(
                "SELECT * FROM claims WHERE id = %s AND submitted_by = %s",
                (claim_id_form, session["email"]),
            )
            claim_record = cursor.fetchone()

        fields = build_fields_from_records(claim_record=claim_record, form_data=form_data)

        try:
            result = verify_claim(fields)
        except Exception as e:
            flash(f"Verification error: {str(e)}", "danger")
            return redirect(url_for("claim_verification"))

        # Database Upsert Operation
        try:
            existing_id = None
            if claim_id_form:
                cursor.execute(
                    """SELECT verification_id FROM claim_verification 
                       WHERE claim_id = %s AND verified_by = %s""",
                    (claim_id_form, session["email"]),
                )
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing["verification_id"]

            if existing_id:
                cursor.execute(
                    """UPDATE claim_verification
                       SET verification_status=%s, verification_score=%s,
                           missing_fields=%s, failed_rules=%s,
                           passed_rules=%s, remarks=%s,
                           input_fields=%s, created_at=NOW()
                       WHERE verification_id=%s""",
                    (
                        result["status"],
                        result["score"],
                        json.dumps(result["missing_fields"]),
                        json.dumps(result["failed_rules"]),
                        json.dumps(result["passed_rules"]),
                        result["remarks"],
                        json.dumps(fields),
                        existing_id,
                    ),
                )
            else:
                cursor.execute(
                    """INSERT INTO claim_verification
                       (claim_id, verification_status, verification_score,
                        missing_fields, failed_rules, passed_rules,
                        remarks, input_fields, verified_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        int(claim_id_form) if claim_id_form else None,
                        result["status"],
                        result["score"],
                        json.dumps(result["missing_fields"]),
                        json.dumps(result["failed_rules"]),
                        json.dumps(result["passed_rules"]),
                        result["remarks"],
                        json.dumps(fields),
                        session["email"],
                    ),
                )
            mysql.connection.commit()
        except Exception as e:
            flash(f"Could not save verification result: {str(e)}", "warning")

        prefill = fields

    cursor.execute(
        """SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by = %s 
           ORDER BY id DESC LIMIT 20""",
        (session["email"],),
    )
    claims_list = cursor.fetchall()
    cursor.close()

    return render_template(
        "claim_verification.html",
        result=result,
        prefill=prefill,
        claims_list=claims_list,
        claim_id=claim_id or request.form.get("claim_id", ""),
    )


@app.route("/claim-verification/history")
def claim_verification_history():
    """Retrieve full history of performed claim verification reports."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT cv.*, c.patient_name, c.hospital_name
           FROM claim_verification cv
           LEFT JOIN claims c ON cv.claim_id = c.id
           WHERE cv.verified_by = %s
           ORDER BY cv.verification_id DESC""",
        (session["email"],),
    )
    verifications = cursor.fetchall()
    cursor.close()

    for row in verifications:
        for col in ["missing_fields", "failed_rules", "passed_rules"]:
            try:
                row[col] = json.loads(row[col]) if row[col] else []
            except Exception:
                row[col] = []

    return render_template("claim_verification_history.html", verifications=verifications)


# ============================================================
#  AI CLAIM APPROVAL PREDICTION ROUTES
# ============================================================
@app.route("/claim-prediction", methods=["GET", "POST"])
def claim_prediction():
    """Execute AI ML model inferences to calculate approval probability."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    result = None
    prefill = {}
    model_ready = is_model_ready()
    model_metrics = get_model_metrics()

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    claim_id = request.args.get("claim_id", "")

    if claim_id and request.method == "GET":
        cursor.execute(
            "SELECT * FROM claims WHERE id = %s AND submitted_by = %s",
            (claim_id, session["email"]),
        )
        claim_rec = cursor.fetchone()
        if claim_rec:
            prefill.update(
                {
                    "patient_name": claim_rec.get("patient_name", ""),
                    "hospital_name": claim_rec.get("hospital_name", ""),
                    "disease": claim_rec.get("disease", ""),
                    "bill_amount": claim_rec.get("bill_amount", ""),
                    "policy_number": claim_rec.get("policy_number", ""),
                }
            )

        cursor.execute(
            """SELECT verification_score, verification_status
               FROM claim_verification
               WHERE claim_id = %s
               ORDER BY verification_id DESC LIMIT 1""",
            (claim_id,),
        )
        ver_rec = cursor.fetchone()
        if ver_rec:
            prefill["verification_score"] = ver_rec.get("verification_score", 50)
            prefill["verification_status"] = ver_rec.get("verification_status", "")

    if request.method == "POST":
        if not model_ready:
            flash("AI model not trained yet. Run: python ml/train_model.py", "danger")
            return redirect(url_for("claim_prediction"))

        claim_id = request.form.get("claim_id", "").strip()

        claim_data = {
            "patient_age": request.form.get("patient_age", "35").strip(),
            "gender": request.form.get("gender", "male").strip(),
            "hospital_type": request.form.get("hospital_type", "private").strip(),
            "disease": request.form.get("disease", "").strip(),
            "bill_amount": request.form.get("bill_amount", "0").strip(),
            "admission_date": request.form.get("admission_date", "").strip(),
            "discharge_date": request.form.get("discharge_date", "").strip(),
            "insurance_id": request.form.get("insurance_id", "").strip(),
            "policy_number": request.form.get("policy_number", "").strip(),
            "verification_score": request.form.get("verification_score", "50").strip(),
            "verification_status": request.form.get("verification_status", "").strip(),
            "previous_claims": request.form.get("previous_claims", "0").strip(),
            "fraud_flag": request.form.get("fraud_flag", "0").strip(),
        }
        prefill = claim_data

        try:
            result = predict(claim_data)
        except FileNotFoundError as e:
            flash(str(e), "danger")
            return redirect(url_for("claim_prediction"))
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")
            return redirect(url_for("claim_prediction"))

        ver_id = None
        if claim_id:
            try:
                cursor.execute(
                    """SELECT verification_id FROM claim_verification 
                       WHERE claim_id=%s ORDER BY verification_id DESC LIMIT 1""",
                    (claim_id,),
                )
                vr = cursor.fetchone()
                if vr:
                    ver_id = vr["verification_id"]
            except Exception:
                pass

        try:
            existing_pred_id = None
            if claim_id:
                cursor.execute(
                    """SELECT prediction_id FROM claim_prediction 
                       WHERE claim_id=%s AND predicted_by=%s""",
                    (claim_id, session["email"]),
                )
                ep = cursor.fetchone()
                if ep:
                    existing_pred_id = ep["prediction_id"]

            if existing_pred_id:
                cursor.execute(
                    """UPDATE claim_prediction
                       SET prediction=%s, approval_probability=%s,
                           rejection_probability=%s, confidence=%s,
                           model_name=%s, input_features=%s,
                           verification_id=%s, created_at=NOW()
                       WHERE prediction_id=%s""",
                    (
                        result["prediction"],
                        result["approval_probability"],
                        result["rejection_probability"],
                        result["confidence"],
                        result["model_name"],
                        json.dumps(claim_data),
                        ver_id,
                        existing_pred_id,
                    ),
                )
            else:
                cursor.execute(
                    """INSERT INTO claim_prediction
                       (claim_id, verification_id, prediction,
                        approval_probability, rejection_probability,
                        confidence, model_name, input_features, predicted_by)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        int(claim_id) if claim_id else None,
                        ver_id,
                        result["prediction"],
                        result["approval_probability"],
                        result["rejection_probability"],
                        result["confidence"],
                        result["model_name"],
                        json.dumps(claim_data),
                        session["email"],
                    ),
                )
            mysql.connection.commit()
        except Exception as e:
            flash(f"Could not save prediction: {str(e)}", "warning")

    cursor.execute(
        """SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by = %s 
           ORDER BY id DESC LIMIT 20""",
        (session["email"],),
    )
    claims_list = cursor.fetchall()
    cursor.close()

    return render_template(
        "claim_prediction.html",
        result=result,
        prefill=prefill,
        claims_list=claims_list,
        claim_id=claim_id,
        model_ready=model_ready,
        model_metrics=model_metrics,
    )


@app.route("/claim-prediction/history")
def claim_prediction_history():
    """Retrieve historical claim approval prediction metrics."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT cp.*, c.patient_name, c.hospital_name, c.disease
           FROM claim_prediction cp
           LEFT JOIN claims c ON cp.claim_id = c.id
           WHERE cp.predicted_by = %s
           ORDER BY cp.prediction_id DESC""",
        (session["email"],),
    )
    predictions = cursor.fetchall()
    cursor.close()

    return render_template("claim_prediction_history.html", predictions=predictions)


# ============================================================
#  FRAUD DETECTION ROUTES
# ============================================================
@app.route("/claim-fraud", methods=["GET", "POST"])
def claim_fraud():
    """Analyze claim data for potential anomalies and fraud indicators."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    result = None
    prefill = {}
    model_ready = is_fraud_model_ready()
    model_stats = get_fraud_model_stats()
    claim_id = request.args.get("claim_id", "")

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if claim_id and request.method == "GET":
        cursor.execute(
            "SELECT * FROM claims WHERE id=%s AND submitted_by=%s",
            (claim_id, session["email"]),
        )
        claim_rec = cursor.fetchone()
        if claim_rec:
            prefill.update(
                {
                    "patient_name": claim_rec.get("patient_name", ""),
                    "hospital_name": claim_rec.get("hospital_name", ""),
                    "disease": claim_rec.get("disease", ""),
                    "bill_amount": claim_rec.get("bill_amount", ""),
                    "policy_number": claim_rec.get("policy_number", ""),
                }
            )

        cursor.execute(
            """SELECT verification_score, verification_status
               FROM claim_verification WHERE claim_id=%s
               ORDER BY verification_id DESC LIMIT 1""",
            (claim_id,),
        )
        ver = cursor.fetchone()
        if ver:
            prefill["verification_score"] = ver.get("verification_score", 50)
            prefill["verification_status"] = ver.get("verification_status", "")

        cursor.execute(
            """SELECT approval_probability FROM claim_prediction
               WHERE claim_id=%s ORDER BY prediction_id DESC LIMIT 1""",
            (claim_id,),
        )
        pred = cursor.fetchone()
        if pred:
            prefill["approval_probability"] = pred.get("approval_probability", 50)

        cursor.execute(
            """SELECT id, bill_amount
               FROM claims
               WHERE submitted_by = %s AND id != %s""",
            (session["email"], claim_id),
        )
        prefill["past_claims_summary"] = cursor.fetchall() or []

        cursor.execute(
            """SELECT COUNT(*) AS cnt FROM claims
               WHERE submitted_by=%s
               AND submitted_at >= NOW() - INTERVAL 7 DAY""",
            (session["email"],),
        )
        freq_row = cursor.fetchone()
        prefill["claim_frequency_7d"] = freq_row["cnt"] if freq_row else 0

    if request.method == "POST":
        claim_id = request.form.get("claim_id", "").strip()

        claim_data = {
            "bill_amount": request.form.get("bill_amount", "0").strip(),
            "admission_date": request.form.get("admission_date", "").strip(),
            "discharge_date": request.form.get("discharge_date", "").strip(),
            "insurance_id": request.form.get("insurance_id", "").strip(),
            "policy_number": request.form.get("policy_number", "").strip(),
            "doctor_name": request.form.get("doctor_name", "").strip(),
            "disease": request.form.get("disease", "").strip(),
            "verification_score": request.form.get("verification_score", "50").strip(),
            "verification_status": request.form.get("verification_status", "").strip(),
            "approval_probability": request.form.get("approval_probability", "50").strip(),
            "previous_claims": request.form.get("previous_claims", "0").strip(),
            "claim_frequency_7d": request.form.get("claim_frequency_7d", "0").strip(),
        }
        prefill = claim_data

        try:
            result = detect_fraud(claim_data)
        except Exception as e:
            flash(f"Fraud detection error: {str(e)}", "danger")
            return redirect(url_for("claim_fraud"))

        ver_id = pred_id = None
        if claim_id:
            try:
                cursor.execute(
                    """SELECT verification_id FROM claim_verification
                       WHERE claim_id=%s ORDER BY verification_id DESC LIMIT 1""",
                    (claim_id,),
                )
                vr = cursor.fetchone()
                if vr:
                    ver_id = vr["verification_id"]

                cursor.execute(
                    """SELECT prediction_id FROM claim_prediction
                       WHERE claim_id=%s ORDER BY prediction_id DESC LIMIT 1""",
                    (claim_id,),
                )
                pr = cursor.fetchone()
                if pr:
                    pred_id = pr["prediction_id"]
            except Exception:
                pass

        try:
            existing_id = None
            if claim_id:
                cursor.execute(
                    """SELECT fraud_id FROM fraud_detection
                       WHERE claim_id=%s AND detected_by=%s""",
                    (claim_id, session["email"]),
                )
                ex = cursor.fetchone()
                if ex:
                    existing_id = ex["fraud_id"]

            if existing_id:
                cursor.execute(
                    """UPDATE fraud_detection
                       SET fraud_status=%s, fraud_score=%s,
                           fraud_probability=%s, detected_rules=%s,
                           anomaly_detected=%s, recommendation=%s,
                           model_name=%s, created_at=NOW()
                       WHERE fraud_id=%s""",
                    (
                        result["fraud_status"],
                        result["fraud_score"],
                        result["fraud_probability"],
                        json.dumps(result["detected_rules"]),
                        int(result["anomaly_detected"]),
                        result["recommendation"],
                        result["model_name"],
                        existing_id,
                    ),
                )
            else:
                cursor.execute(
                    """INSERT INTO fraud_detection
                       (claim_id, verification_id, prediction_id,
                        fraud_status, fraud_score, fraud_probability,
                        detected_rules, anomaly_detected,
                        recommendation, model_name, detected_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        int(claim_id) if claim_id else None,
                        ver_id,
                        pred_id,
                        result["fraud_status"],
                        result["fraud_score"],
                        result["fraud_probability"],
                        json.dumps(result["detected_rules"]),
                        int(result["anomaly_detected"]),
                        result["recommendation"],
                        result["model_name"],
                        session["email"],
                    ),
                )
            mysql.connection.commit()

        except Exception as e:
            flash(f"Could not save fraud result: {str(e)}", "warning")

    cursor.execute(
        """SELECT id, patient_name, hospital_name, bill_amount
           FROM claims WHERE submitted_by=%s ORDER BY id DESC LIMIT 20""",
        (session["email"],),
    )
    claims_list = cursor.fetchall()
    cursor.close()

    return render_template(
        "fraud_detection.html",
        result=result,
        prefill=prefill,
        claims_list=claims_list,
        claim_id=claim_id,
        model_ready=model_ready,
        model_stats=model_stats,
    )


@app.route("/claim-fraud/history")
def claim_fraud_history():
    """Retrieve historical fraud investigation logs."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT fd.*, c.patient_name, c.hospital_name, c.disease
           FROM fraud_detection fd
           LEFT JOIN claims c ON fd.claim_id = c.id
           WHERE fd.detected_by=%s
           ORDER BY fd.fraud_id DESC""",
        (session["email"],),
    )
    records = cursor.fetchall()
    cursor.close()

    for r in records:
        try:
            r["detected_rules"] = (
                json.loads(r["detected_rules"]) if r.get("detected_rules") else []
            )
        except Exception:
            r["detected_rules"] = []

    return render_template("fraud_history.html", records=records)


# ============================================================
#  AI MEDICAL SUMMARIZATION ROUTES
# ============================================================
@app.route("/medical-summary", methods=["GET", "POST"])
def medical_summary():
    """Generate concise AI medical text summaries from manual text or OCR files."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    result = None
    ocr_text_used = ""
    ocr_id = request.args.get("ocr_id", "")
    selected_ocr = None

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if ocr_id and request.method == "GET":
        cursor.execute(
            "SELECT * FROM ocr_results WHERE id=%s AND processed_by=%s",
            (ocr_id, session["email"]),
        )
        selected_ocr = cursor.fetchone()
        if selected_ocr:
            ocr_text_used = selected_ocr.get("extracted_text", "")

    if request.method == "POST":
        ocr_id = request.form.get("ocr_id", "").strip()
        summary_type = request.form.get("summary_type", "medium").strip()
        manual_text = request.form.get("manual_text", "").strip()
        claim_id_form = request.form.get("claim_id", "").strip()

        if ocr_id:
            cursor.execute(
                "SELECT * FROM ocr_results WHERE id=%s AND processed_by=%s",
                (ocr_id, session["email"]),
            )
            selected_ocr = cursor.fetchone()
            if selected_ocr:
                ocr_text_used = selected_ocr.get("extracted_text", "")

        if not ocr_text_used and manual_text:
            ocr_text_used = manual_text

        if not ocr_text_used:
            flash(
                "No text to summarize. Select an OCR result or paste text manually.",
                "warning",
            )
            return redirect(url_for("medical_summary"))

        try:
            result = summarize(
                ocr_text=ocr_text_used,
                summary_type=summary_type,
                ocr_result_id=int(ocr_id) if ocr_id else None,
                claim_id=int(claim_id_form) if claim_id_form else None,
            )
        except Exception as e:
            flash(f"Summarization error: {str(e)}", "danger")
            return redirect(url_for("medical_summary"))

        if result.get("error") and not result.get("summary"):
            flash(result["error"], "warning")

        if result and result.get("summary"):
            try:
                existing_id = None
                if ocr_id:
                    cursor.execute(
                        """SELECT summary_id FROM medical_summary
                           WHERE ocr_result_id=%s AND summary_type=%s AND created_by=%s""",
                        (ocr_id, result["summary_type"], session["email"]),
                    )
                    ex = cursor.fetchone()
                    if ex:
                        existing_id = ex["summary_id"]

                if existing_id:
                    cursor.execute(
                        """UPDATE medical_summary
                           SET generated_summary=%s, original_text_length=%s,
                               summary_length=%s, compression_ratio=%s,
                               model_used=%s, created_at=NOW()
                           WHERE summary_id=%s""",
                        (
                            result["summary"],
                            result["original_word_count"],
                            result["word_count"],
                            result["compression_ratio"],
                            result["model_used"],
                            existing_id,
                        ),
                    )
                else:
                    cursor.execute(
                        """INSERT INTO medical_summary
                           (claim_id, ocr_result_id, summary_type, generated_summary,
                            original_text_length, summary_length, compression_ratio,
                            model_used, created_by)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            int(claim_id_form) if claim_id_form else None,
                            int(ocr_id) if ocr_id else None,
                            result["summary_type"],
                            result["summary"],
                            result["original_word_count"],
                            result["word_count"],
                            result["compression_ratio"],
                            result["model_used"],
                            session["email"],
                        ),
                    )
                mysql.connection.commit()
            except Exception as e:
                flash(f"Could not save summary: {str(e)}", "warning")

    cursor.execute(
        """SELECT id, filename, created_at 
           FROM ocr_results 
           WHERE processed_by=%s ORDER BY id DESC""",
        (session["email"],),
    )
    ocr_list = cursor.fetchall()

    cursor.execute(
        """SELECT id, patient_name 
           FROM claims 
           WHERE submitted_by=%s ORDER BY id DESC LIMIT 10""",
        (session["email"],),
    )
    claims_list = cursor.fetchall()
    cursor.close()

    return render_template(
        "medical_summary.html",
        result=result,
        ocr_list=ocr_list,
        ocr_id=ocr_id,
        selected_ocr=selected_ocr,
        ocr_text_used=ocr_text_used,
        claims_list=claims_list,
        model_name=get_model_name(),
        model_loaded=is_model_loaded(),
    )


@app.route("/medical-summary/history")
def medical_summary_history():
    """Retrieve historical AI medical summary records."""
    if "loggedin" not in session:
        return redirect(url_for("login"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        """SELECT ms.*, ocr.filename
           FROM medical_summary ms
           LEFT JOIN ocr_results ocr ON ms.ocr_result_id = ocr.id
           WHERE ms.created_by=%s
           ORDER BY ms.summary_id DESC""",
        (session["email"],),
    )
    summaries = cursor.fetchall()
    cursor.close()

    return render_template("medical_summary_history.html", summaries=summaries)


# ============================================================
#  MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)