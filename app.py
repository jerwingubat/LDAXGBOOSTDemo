from flask import Flask, request, render_template, redirect, flash
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import os
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import re
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from flask_socketio import SocketIO
import os

nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'ldaXGboost'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = joblib.load("web_ready_xgboost_model.pkl")
encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
dictionary = Dictionary.load('lda_dictionary.gensim')
lda_model = LdaModel.load('lda_model.gensim')
with open('bigram_mod.pkl', 'rb') as f:
    bigram_mod = pickle.load(f)
with open('trigram_mod.pkl', 'rb') as f:
    trigram_mod = pickle.load(f)

stop_words = set(stopwords.words('english'))

feature_names = [
    "Age", "GWA", "Attendance_Rate", "Library_Usage_Hours", "Scholarship_Status",
    "Counseling_Sessions", "Gender", "Course", "Year_Level", "School_Environment_Feedback"
]

NUM_TOPICS = 3

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\'\-\.\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

def safe_label_encode(encoder, value, default=0):
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return default
    except Exception:
        return default

def fuzzy_map_columns(df_columns, target_features):
    mapping = {}
    for feature in target_features:
        best_match, score = process.extractOne(feature, df_columns, scorer=fuzz.token_sort_ratio)
        mapping[feature] = best_match
        if score < 50:
            print(f"[Warning] Low confidence mapping for '{feature}': matched to '{best_match}' with score {score}")
    return mapping

@app.route('/')
def index():
    options = {
        "Gender": encoders["Gender"].classes_,
        "Course": encoders["Course"].classes_,
        "Year_Level": encoders["Year_Level"].classes_,
        "Scholarship_Status": encoders["Scholarship_Status"].classes_,
    }
    return render_template("index.html", options=options, active='home')

@app.route("/predict", methods=["POST"])
def predict():
    data = {key: request.form.get(key, '') for key in feature_names}
    name = request.form.get('Name', '')
    for col in ["Age", "GWA", "Attendance_Rate", "Library_Usage_Hours", "Counseling_Sessions"]:
        data[col] = float(data[col])
    for col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
        data[col] = safe_label_encode(encoders[col], data[col])
    feedback = request.form['School_Environment_Feedback']
    cleaned = preprocess_text(feedback)
    cleaned = trigram_mod[bigram_mod[cleaned]]
    bow = dictionary.doc2bow(cleaned)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_scores = [score for _, score in sorted(topic_dist)]
    for i, score in enumerate(topic_scores):
        data[f'Topic_{i+1}'] = score
    data.pop("School_Environment_Feedback", None)
    df = pd.DataFrame([data])
    df = df[model.get_booster().feature_names]
    pred = model.predict(df)[0]
    pred_label = target_encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(df)[0]
    risk_score = float(max(proba))
    if risk_score >= 0.7:
        risk_level = "Dropout Risk"
        recommendations = "Immediate intervention recommended."
    elif risk_score >= 0.4:
        risk_level = "Good"
        recommendations = "Monitor and provide support."
    else:
        risk_level = "Probation"
        recommendations = "No immediate action needed."
    options = {
        "Gender": encoders["Gender"].classes_,
        "Course": encoders["Course"].classes_,
        "Year_Level": encoders["Year_Level"].classes_,
        "Scholarship_Status": encoders["Scholarship_Status"].classes_,
    }
    return render_template(
        "index.html",
        prediction=pred_label,
        options=options,
        name=name,
        risk_score=f"{risk_score:.2f}",
        risk_level=risk_level,
        recommendations=recommendations
    )

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df = pd.read_csv(filepath, encoding="ISO-8859-1")
            column_mapping = fuzzy_map_columns(df.columns.tolist(), feature_names)
            print("Column Mapping:", column_mapping)
            original_df = df.rename(columns={v: k for k, v in column_mapping.items()})
            df = original_df.copy()
            for col in feature_names:
                if col not in df.columns:
                    if col == "School_Environment_Feedback":
                        df[col] = ""
                    elif col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
                        df[col] = "Unknown"
                    else:
                        df[col] = 0
            for col in ["Age", "GWA", "Attendance_Rate", "Library_Usage_Hours", "Counseling_Sessions"]:
                try:
                    df[col] = df[col].astype(float)
                except:
                    df[col] = 0.0
            for col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
                if col in encoders:
                    df[col] = df[col].astype(str).apply(lambda x: safe_label_encode(encoders[col], x, default=0))
                else:
                    df[col] = 0
            topic_scores = []
            for feedback in df["School_Environment_Feedback"]:
                cleaned = preprocess_text(feedback)
                cleaned = trigram_mod[bigram_mod[cleaned]]
                bow = dictionary.doc2bow(cleaned)
                topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
                scores = [score for _, score in sorted(topic_dist)]
                topic_scores.append(scores)
            topic_df = pd.DataFrame(topic_scores, columns=["Topic_1", "Topic_2", "Topic_3"])
            df = pd.concat([df.drop(columns=["School_Environment_Feedback"]), topic_df], axis=1)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df = df[model.get_booster().feature_names]
            preds = model.predict(df)
            pred_labels = target_encoder.inverse_transform(preds)
            original_df['Predicted_Academic_Standing'] = pred_labels
            for col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
                try:
                    inverse = encoders[col].inverse_transform(df[col])
                    original_df[col] = inverse
                except:
                    original_df[col] = df[col].apply(lambda x: "" if x == 0 else str(x))
            table = original_df.to_dict(orient='records')
            return render_template('batch_predict.html', table=table, columns=original_df.columns, active='report')
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    return render_template('batch_predict.html', table=None, columns=None, active='report')

@app.route('/maintenance/analysis')
def maintenance_analysis():
    return render_template('maintenance.html', active='analysis', section_name='Analysis')

@app.route('/maintenance/intervention')
def maintenance_intervention():
    return render_template('maintenance.html', active='intervention', section_name='Intervention')

@app.route('/maintenance/data')
def maintenance_data():
    return render_template('maintenance.html', active='data', section_name='Data Management')

@app.route('/maintenance/system')
def maintenance_system():
    return render_template('maintenance.html', active='system', section_name='System Maintenance')

if __name__ == "__main__":
    socketio = SocketIO(app, async_mode='threading')
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), allow_unsafe_werkzeug=True)
