import os, io, joblib
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

app = Flask(__name__)
app.secret_key = "VIP_CARDIO_SECRET_2026"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///clinic_pro.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- MODELS ---
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    prenom = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sexe = db.Column(db.String(20), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    analyses = db.relationship('Analyse', backref='patient', lazy=True)

class Analyse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    resultat = db.Column(db.Integer)
    conseil = db.Column(db.Text)
    details = db.Column(db.JSON)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

# --- IA MODELS LOAD ---
try:
    model = joblib.load('modele_heart_disease.pkl')
    scaler = joblib.load('scaler_heart_disease.pkl')
except:
    model, scaler = None, None

with app.app_context():
    db.create_all()

# --- AUTH ROUTES ---
@app.route('/')
def home(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        doc = Doctor.query.filter_by(username=request.form['username']).first()
        if doc and check_password_hash(doc.password, request.form['password']):
            session['doc_id'], session['doc_name'] = doc.id, doc.username
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form['username']
        pw = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        db.session.add(Doctor(username=user, password=pw))
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- MAIN ROUTES ---
@app.route('/dashboard')
def dashboard():
    if 'doc_id' not in session: return redirect(url_for('login'))
    patients = Patient.query.filter_by(doctor_id=session['doc_id']).all()
    total_p = len(patients)
    total_e = Analyse.query.join(Patient).filter(Patient.doctor_id == session['doc_id']).count()
    total_a = Analyse.query.join(Patient).filter(Patient.doctor_id == session['doc_id'], Analyse.resultat == 1).count()
    return render_template('dashboard.html', patients=patients, total_p=total_p, total_e=total_e, total_a=total_a)

@app.route('/about')
def about():
    if 'doc_id' not in session: return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if 'doc_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        p = Patient(nom=request.form['nom'], prenom=request.form['prenom'], 
                    age=request.form['age'], sexe=request.form['sexe'], doctor_id=session['doc_id'])
        db.session.add(p)
        db.session.commit()
        return redirect(url_for('dashboard'))
    return render_template('add_patient.html')

@app.route('/predict/<int:p_id>', methods=['GET', 'POST'])
def predict(p_id):
    if 'doc_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(p_id)
    if request.method == 'POST':
        cp_map = {"Angine typique": 0, "Angine atypique": 1, "Douleur non angineuse": 2, "Asymptomatique": 3}
        ecg_map = {"Normal": 0, "Anomalie ST-T": 1, "Hypertrophie": 2}
        slope_map = {"Ascendant": 0, "Plat": 1, "Descendant": 2}
        thal_map = {"Normal": 1, "Défaut fixe": 2, "Défaut réversible": 3}

        details = {
            "Sexe": patient.sexe,
            "Tension": f"{request.form.get('trestbps')} mmHg",
            "Cholesterol": f"{request.form.get('chol')} mg/dl",
            "Douleur": request.form.get('cp'),
            "Freq_Max": request.form.get('thalach'),
            "Glycemie": request.form.get('fbs'),
            "ECG": request.form.get('restecg'),
            "Angine_Effort": request.form.get('exang'),
            "Oldpeak": request.form.get('oldpeak'),
            "Vaisseaux": request.form.get('ca'),
            "Thal": request.form.get('thal')
        }

        features = [
            float(patient.age), 1.0 if patient.sexe == 'Homme' else 0.0,
            float(cp_map.get(request.form.get('cp'), 0)), float(request.form.get('trestbps', 120)),
            float(request.form.get('chol', 200)), 1.0 if request.form.get('fbs') == 'Oui' else 0.0,
            float(ecg_map.get(request.form.get('restecg'), 0)), float(request.form.get('thalach', 150)),
            1.0 if request.form.get('exang') == 'Oui' else 0.0, float(request.form.get('oldpeak', 0.0)),
            float(slope_map.get(request.form.get('slope'), 1)), float(request.form.get('ca', 0)),
            float(thal_map.get(request.form.get('thal'), 1))
        ]

        if model and scaler:
            res = int(model.predict(scaler.transform([features]))[0])
            cons = "Risque Faible. Santé cardiaque normale." if res == 0 else "Risque Élevé ! Consultation urgente."
            new_a = Analyse(resultat=res, conseil=cons, details=details, patient_id=patient.id)
            db.session.add(new_a)
            db.session.commit()
            return redirect(url_for('patient_history', p_id=patient.id))
    return render_template('predict.html', patient=patient)

@app.route('/history/<int:p_id>')
def patient_history(p_id):
    if 'doc_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(p_id)
    return render_template('patient_history.html', patient=patient)

# --- PDF GENERATION ROUTE (VIP VERSION UPDATED) ---
@app.route('/download_pdf/<int:a_id>')
def download_pdf(a_id):
    a = Analyse.query.get_or_404(a_id)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- Border ---
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.rect(20, 20, width-40, height-40, stroke=1)
    
    # --- Header VIP ---
    c.setFillColor(colors.HexColor("#1e293b"))
    c.rect(0, height-100, width, 100, fill=1)
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height-55, "CARDIO-PRO ELITE")
    c.setFont("Helvetica", 10)
    c.drawString(50, height-75, "CENTRE DE DIAGNOSTIC PAR INTELLIGENCE ARTIFICIELLE")
    
    # --- Patient Info ---
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, height-130, f"MÉDECIN TRAITANT : Dr. {session.get('doc_name','lamiae')}")
    c.drawString(400, height-130, f"RÉFÉRENCE : #REP-{a.id:04d}")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-175, f"DOSSIER PATIENT : {a.patient.nom.upper()} {a.patient.prenom}")
    c.setFont("Helvetica", 11)
    c.drawString(50, height-195, f"ÂGE : {a.patient.age} ans | GENRE : {a.details.get('Sexe','-')}")

    # --- Résultat Box ---
    res_bg = colors.HexColor("#fee2e2") if a.resultat == 1 else colors.HexColor("#dcfce7")
    res_text_color = colors.HexColor("#991b1b") if a.resultat == 1 else colors.HexColor("#166534")
    c.setFillColor(res_bg)
    c.rect(50, height-260, width-100, 50, fill=1, stroke=0)
    c.setFillColor(res_text_color)
    c.setFont("Helvetica-Bold", 16)
    diag = "DIAGNOSTIC : RISQUE CARDIAQUE DÉTECTÉ" if a.resultat == 1 else "DIAGNOSTIC : AUCUNE ANOMALIE"
    c.drawCentredString(width/2, height-240, diag)

    # --- Clinical Data Table Section ---
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    # Had l-jomla ghadi t-ban nqiya daba
    c.drawString(50, height-295, "ANALYSES BIOMÉTRIQUES :")
    
    data = [["PARAMÈTRE", "VALEUR", "STATUS"]]
    for k, v in a.details.items():
        data.append([k, v, "Vérifié"])
    
    table = Table(data, colWidths=[180, 150, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))

    # Hna l-far9: habbtna l-tableau l-height-580
    table.wrapOn(c, 50, height-600)
    table.drawOn(c, 70, height-580)

    # --- Footer ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 150, "NOTE DU SYSTÈME :")
    c.setFont("Helvetica-Oblique", 11)
    c.setFillColor(colors.grey)
    c.drawString(50, 130, a.conseil)
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(400, 80, "SIGNATURE & CACHET")
    c.line(400, 75, 550, 75)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"RAPPORT_VIP_{a.patient.nom}.pdf")

if __name__ == '__main__':
    app.run(debug=True)