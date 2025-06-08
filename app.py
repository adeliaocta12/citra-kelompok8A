from flask import Flask, render_template, request, send_file, redirect, url_for
import cv2
import numpy as np
import os
from fpdf import FPDF

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ORIGINAL_FILE = os.path.join(UPLOAD_FOLDER, 'uploaded.png')
RESULT_FILE = 'static/result.png'
PDF_FILE = 'static/result.pdf'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def apply_filter(image, filter_type, brightness_val=50, contrast_val=150):
    # brightness_val: int (0-255 or 0-100 scale?), we'll treat as 0-255 increment
    # contrast_val: int (percentage, e.g., 150 means 1.5 alpha)
    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'edge':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif filter_type == 'threshold':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return result
    elif filter_type == 'invert':
        return cv2.bitwise_not(image)
    elif filter_type == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif filter_type == 'emboss':
        kernel = np.array([[0, -1, -1],
                           [1,  0, -1],
                           [1,  1,  0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'brightness':
        # Adjust brightness by brightness_val (0-255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - brightness_val
        v[v > lim] = 255
        v[v <= lim] += brightness_val
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    elif filter_type == 'contrast':
        # Adjust contrast by contrast_val (%), e.g., 150 = 1.5 alpha
        alpha = contrast_val / 100.0
        beta = 0
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    elif filter_type == 'gaussian_noise':
        row, col, ch = image.shape
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss.reshape(row, col, ch)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    elif filter_type == 'soft_sepia':
        kernel = np.array([[0.15, 0.35, 0.2],
                           [0.25, 0.50, 0.3],
                           [0.3,  0.55, 0.4]])
        sepia = cv2.transform(image, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif filter_type == 'negative_alt':
        inverted = cv2.bitwise_not(image)
        brightness_increase = 20
        hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - brightness_increase
        v[v > lim] = 255
        v[v <= lim] += brightness_increase
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_filter = None
    brightness_val = 50  # default brightness increment
    contrast_val = 150   # default contrast percentage (150% = 1.5 alpha)
    if request.method == 'POST':
        selected_filter = request.form.get('filter')

        # Ambil nilai brightness dan contrast jika tersedia dan valid
        try:
            brightness_val = int(request.form.get('brightness_val', 50))
            brightness_val = max(0, min(255, brightness_val))
        except Exception:
            brightness_val = 50

        try:
            contrast_val = int(request.form.get('contrast_val', 150))
            contrast_val = max(0, min(300, contrast_val))  # batas max 300%
        except Exception:
            contrast_val = 150

        if 'submit_upload' in request.form and 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                file.save(ORIGINAL_FILE)

        if os.path.exists(ORIGINAL_FILE):
            image = cv2.imread(ORIGINAL_FILE)
            result = apply_filter(image, selected_filter, brightness_val, contrast_val)
            cv2.imwrite(RESULT_FILE, result)
            return render_template('index.html',
                                   original='uploads/uploaded.png',
                                   result='result.png',
                                   selected_filter=selected_filter,
                                   brightness_val=brightness_val,
                                   contrast_val=contrast_val)
    return render_template('index.html',
                           brightness_val=brightness_val,
                           contrast_val=contrast_val)


@app.route('/download')
def download():
    return send_file(RESULT_FILE, as_attachment=True)


@app.route('/delete')
def delete():
    for f in [ORIGINAL_FILE, RESULT_FILE, PDF_FILE]:
        if os.path.exists(f):
            os.remove(f)
    return redirect(url_for('index'))


@app.route('/export_pdf')
def export_pdf():
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(ORIGINAL_FILE):
        pdf.image(ORIGINAL_FILE, x=10, y=30, w=90)
    if os.path.exists(RESULT_FILE):
        pdf.image(RESULT_FILE, x=110, y=30, w=90)
    pdf.output(PDF_FILE)
    return send_file(PDF_FILE, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
