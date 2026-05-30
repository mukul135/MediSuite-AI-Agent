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
app.config['MYSQL_PASSWORD'] = 'root'       # ← Change to your MySQL password
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