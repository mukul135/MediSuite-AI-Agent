# ============================================================
#  MediSuite-AI-Agent — app.py
#  Main Flask application with User Authentication System
# ============================================================
#
#  What this file does:
#  - Starts the Flask web server
#  - Connects to MySQL database
#  - Handles all URL routes (pages)
#  - Manages user sessions (login/logout state)
#  - Processes HTML form data (registration & login)
# ============================================================

# --- IMPORTS ------------------------------------------------
# Flask     : The web framework that runs our server
# render_template : Loads HTML files from templates/ folder
# request   : Reads data submitted from HTML forms
# redirect  : Sends user to a different page
# url_for   : Builds a URL for a given function name
# session   : Stores login state between page requests
# flash     : Shows one-time messages (success/error)

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL          # Connects Flask to MySQL
import MySQLdb.cursors                   # Lets us read MySQL results as dictionaries
import re                                # Regular expressions — for email validation
import hashlib                           # For hashing passwords (SHA-256)

# ============================================================
#  1. CREATE THE FLASK APP
# ============================================================
app = Flask(__name__)

# SECRET KEY — Required for sessions and flash messages to work.
# Sessions are like a "memory" Flask keeps per browser tab.
# Change this to any long random string in a real project.
app.secret_key = 'medisuite_secret_key_2024_change_this_in_production'


# ============================================================
#  2. MYSQL DATABASE CONFIGURATION
# ============================================================
# These settings tell Flask how to connect to your MySQL server.
# Make sure your MySQL server is running before starting Flask.

app.config['MYSQL_HOST']     = 'localhost'    # MySQL runs on your own computer
app.config['MYSQL_USER']     = 'root'          # Default MySQL username (change if different)
app.config['MYSQL_PASSWORD'] = 'root'              # Your MySQL root password (set yours here)
app.config['MYSQL_DB']       = 'medisuite'     # The database we will use

# Create the MySQL connection object
mysql = MySQL(app)


# ============================================================
#  3. HELPER FUNCTION — Password Hashing
# ============================================================
# NEVER store plain-text passwords in a database.
# We convert the password into a one-way hash using SHA-256.
# Even if the database is hacked, passwords remain safe.

def hash_password(password):
    """Convert a plain-text password into a SHA-256 hash string."""
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
#  4. ROOT ROUTE — Home Page
# ============================================================
# When a user visits http://127.0.0.1:5000/
# they are sent directly to the Login page.

@app.route('/')
def home():
    return redirect(url_for('login'))


# ============================================================
#  5. REGISTER ROUTE
# ============================================================
# URL  : http://127.0.0.1:5000/register
# GET  : Shows the blank registration form
# POST : Receives the filled form, validates it, saves to DB

@app.route('/register', methods=['GET', 'POST'])
def register():

    # --- POST: Form was submitted ---
    if request.method == 'POST':

        # Step 1: Read data the user typed into the form
        # request.form['field_name'] matches the name="" in your HTML input tags
        name     = request.form['name'].strip()
        email    = request.form['email'].strip()
        password = request.form['password'].strip()
        confirm  = request.form['confirm_password'].strip()

        # Step 2: Validation — check for problems before saving

        # Check if any field is empty
        if not name or not email or not password or not confirm:
            flash('All fields are required. Please fill in every field.', 'error')
            return redirect(url_for('register'))

        # Check if email format is valid (must have @ and .)
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('register'))

        # Check if password is at least 6 characters
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('register'))

        # Check if both passwords match
        if password != confirm:
            flash('Passwords do not match. Please try again.', 'error')
            return redirect(url_for('register'))

        # Step 3: Check if this email already exists in the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('An account with this email already exists. Please log in.', 'error')
            return redirect(url_for('register'))

        # Step 4: Hash the password before saving
        hashed_pw = hash_password(password)

        # Step 5: Insert the new user into the database
        cursor.execute(
            'INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
            (name, email, hashed_pw)
        )
        mysql.connection.commit()   # IMPORTANT: Saves the insert to the database
        cursor.close()

        # Step 6: Show success message and redirect to login
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    # --- GET: Show the empty registration form ---
    return render_template('register.html')


# ============================================================
#  6. LOGIN ROUTE
# ============================================================
# URL  : http://127.0.0.1:5000/login
# GET  : Shows the login form
# POST : Checks credentials against database, starts session

@app.route('/login', methods=['GET', 'POST'])
def login():

    # If the user is already logged in, send them to dashboard
    if 'loggedin' in session:
        return redirect(url_for('dashboard'))

    # --- POST: Login form was submitted ---
    if request.method == 'POST':

        # Step 1: Read form data
        email    = request.form['email'].strip()
        password = request.form['password'].strip()

        # Step 2: Check fields are not empty
        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return redirect(url_for('login'))

        # Step 3: Hash the entered password to compare with stored hash
        hashed_pw = hash_password(password)

        # Step 4: Look up the user in the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE email = %s AND password = %s',
            (email, hashed_pw)
        )
        user = cursor.fetchone()   # Returns one matching row, or None
        cursor.close()

        # Step 5: If user found — start a session
        if user:
            # session is like a dictionary Flask stores in the browser cookie
            session['loggedin'] = True          # Mark user as logged in
            session['id']       = user['id']    # Store user's database ID
            session['name']     = user['name']  # Store user's name
            session['email']    = user['email'] # Store user's email

            flash(f"Welcome back, {user['name']}!", 'success')
            return redirect(url_for('dashboard'))

        # Step 6: Wrong email or password
        else:
            flash('Incorrect email or password. Please try again.', 'error')
            return redirect(url_for('login'))

    # --- GET: Show the empty login form ---
    return render_template('login.html')


# ============================================================
#  7. DASHBOARD ROUTE — Protected Page
# ============================================================
# URL  : http://127.0.0.1:5000/dashboard
# This page is ONLY accessible when the user is logged in.
# If not logged in, the user is sent back to login.

@app.route('/dashboard')
def dashboard():

    # Check if the user is logged in
    if 'loggedin' not in session:
        # Not logged in — redirect to login with a warning
        flash('Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))

    # User is logged in — show the dashboard
    # We pass user info to the HTML template
    return render_template('dashboard.html',
                           name=session['name'],
                           email=session['email'])


# ============================================================
#  8. LOGOUT ROUTE
# ============================================================
# URL  : http://127.0.0.1:5000/logout
# Clears the session (removes all stored login data)
# Then redirects to login page

@app.route('/logout')
def logout():
    # Remove all data stored in session
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('name', None)
    session.pop('email', None)

    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))


# ============================================================
#  9. START THE FLASK SERVER
# ============================================================
# debug=True means Flask will:
# - Show detailed error messages in the browser
# - Auto-restart when you save changes to app.py

if __name__ == '__main__':
    app.run(debug=True)