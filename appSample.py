from flask import Flask, request, render_template, redirect, url_for, flash
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
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = joblib.load("web_ready_xgboost_model.pkl")
encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

dictionary = Dictionary.load('lda_dictionary.gensim')
lda_model = LdaModel.load('lda_model.gensim')
with open('bigram_mod.pkl', 'rb') as f:
    bigram_mod = pickle.load(f)
with open('trigram_mod.pkl', 'rb') as f:
    trigram_mod = pickle.load(f)

NUM_TOPICS = 3

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\'\-\.\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

feature_names = [
    "Age",
    "Gender",
    "Course",
    "Year_Level",
    "GWA",
    "Attendance_Rate",
    "Library_Usage_Hours",
    "Scholarship_Status",
    "Counseling_Sessions",
    "School_Environment_Feedback",
]


@app.route("/")
def index():
    options = {
        "Gender": encoders["Gender"].classes_,
        "Course": encoders["Course"].classes_,
        "Year_Level": encoders["Year_Level"].classes_,
        "Scholarship_Status": encoders["Scholarship_Status"].classes_,
    }
    return render_template("index.html", options=options)


@app.route("/predict", methods=["POST"])
def predict():
    data = {key: request.form.get(key, '') for key in feature_names}
    name = request.form.get('Name', '')
    for col in [
        "Age", "GWA", "Attendance_Rate", "Library_Usage_Hours", "Counseling_Sessions"
    ]:
        data[col] = float(data[col])
    for col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
        data[col] = encoders[col].transform([data[col]])[0]
    feedback = request.form['School_Environment_Feedback']
    cleaned = preprocess_text(feedback)
    cleaned = trigram_mod[bigram_mod[cleaned]]
    bow = dictionary.doc2bow(cleaned)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_scores = [score for _, score in sorted(topic_dist)]
    for i, score in enumerate(topic_scores):
        data[f'Topic_{i+1}'] = score
    data.pop("School_Environment_Feedback", None)
    feature_list = [
        "Age", "Gender", "Course", "Year_Level", "GWA", "Attendance_Rate",
        "Library_Usage_Hours", "Scholarship_Status", "Counseling_Sessions",
        "Topic_1", "Topic_2", "Topic_3"
    ]
    df = pd.DataFrame([data])
    df = df[model.get_booster().feature_names]


    pred = model.predict(df)[0]
    pred_label = target_encoder.inverse_transform([pred])[0]


    proba = model.predict_proba(df)[0]
    risk_score = float(max(proba))

    if risk_score >= 0.7:
        risk_level = "High"
        recommendations = "Immediate intervention recommended."
    elif risk_score >= 0.4:
        risk_level = "Medium"
        recommendations = "Monitor and provide support."
    else:
        risk_level = "Low"
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
            
            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                flash(f'Missing columns: {missing_cols}')
                return redirect(request.url)
        
            for col in ["Age", "GWA", "Attendance_Rate", "Library_Usage_Hours", "Counseling_Sessions"]:
                df[col] = df[col].astype(float)
            for col in ["Gender", "Course", "Year_Level", "Scholarship_Status"]:
                df[col] = encoders[col].transform(df[col])
            preds = model.predict(df[feature_names])
            pred_labels = target_encoder.inverse_transform(preds)
            df['Predicted_Academic_Standing'] = pred_labels
        
            if 'Scholarship_Status' in df.columns:
                inv_map = {v: k for k, v in enumerate(encoders['Scholarship_Status'].classes_)}
                yes_val = inv_map.get('Yes', None)
                no_val = inv_map.get('No', None)
                def scholarship_display(val):
                    if yes_val is not None and val == yes_val:
                        return 'Yes'
                    elif no_val is not None and val == no_val:
                        return 'No'
                    return val
                df['Scholarship_Status'] = df['Scholarship_Status'].apply(scholarship_display)
            table = df.to_dict(orient='records')
            return render_template('batch_predict.html', table=table, columns=df.columns)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    return render_template('batch_predict.html', table=None, columns=None)


@app.route('/feedback', methods=['POST'])
def feedback():
    name = request.form.get('Name', '')
    prediction = request.form.get('Prediction', '')
    feedback_text = request.form.get('Feedback', '')
    import csv
    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([name, prediction, feedback_text])
    flash('Thank you for your feedback!')
    options = {
        "Gender": encoders["Gender"].classes_,
        "Course": encoders["Course"].classes_,
        "Year_Level": encoders["Year_Level"].classes_,
        "Scholarship_Status": encoders["Scholarship_Status"].classes_,
    }
    return render_template("index.html", options=options)


if __name__ == "__main__":
    app.run(debug=True)
