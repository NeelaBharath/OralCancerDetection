import cv2
import pickle
from flask import Flask, request, render_template, redirect, url_for, session, flash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management

# Load your trained model
model = pickle.load(open(r'/app/SVM_Model/cnn_model.pickle', 'rb'))


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility functions for preprocessing and prediction
def preprocess_image(file):
    try:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (64, 64))
        reshaped_img = resized_img.reshape(1, 64, 64, 1).astype('float32') / 255.0
        return reshaped_img
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def predict_result(image):
    try:
        res = model.predict(image)
        probability = res[0][0]
        threshold = 0.5  # Set threshold for decision boundary
        return 'You are suffering from cancer. Please consult a doctor immediately' if probability >= threshold else 'You are not suffering from cancer. Stay healthy!'
    except Exception as e:
        print(f"Error in predicting result: {e}")
        return 'Error predicting result'

# Routes for login and upload
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html', error=None)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            image = preprocess_image(filename)
            if image is None:
                flash('Error processing image')
                return redirect(request.url)

            result = predict_result(image)
            return render_template('upload.html', result=result)

    return render_template('upload.html', result=None)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
