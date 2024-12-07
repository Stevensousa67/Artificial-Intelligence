from flask import Flask, render_template, request, session, send_from_directory
import pandas as pd
import os, ARIMA, EDA, K_Means_Clustering, Linear_Regression, preprocessing
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session management

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Population_History_App', 'uploads')
PLOT_FOLDER = os.path.join(os.getcwd(), 'Population_History_App', 'static', 'plots')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/plots/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['PLOT_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    algorithms = ['EDA', 'Linear Regression', 'K-Means Clustering', 'ARIMA']
    selected_algo = request.form.get('algorithm', '')
    countries = session.get('countries', [])
    indicators = session.get('indicators', [])
    selected_country = request.form.get('country', '')
    selected_indicator = request.form.get('indicator', '')

    excel_previews = []

    if request.method == 'POST':
        if 'preprocess' in request.form:
            if 'file' not in request.files:
                return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error="No file part")

            file = request.files['file']
            if file.filename == '':
                return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error="No selected file")

            if file and allowed_file(file.filename):
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    cleaned_file_path = preprocessing.preprocess_data(filepath)
                    if isinstance(cleaned_file_path, str) and "Error" in cleaned_file_path:
                        return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error=cleaned_file_path)

                    if cleaned_file_path.endswith('.csv'):
                        df = pd.read_csv(cleaned_file_path)
                    elif cleaned_file_path.endswith('.xlsx'):
                        df = pd.read_excel(cleaned_file_path)
                    else:
                        return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error="Unexpected error after preprocessing.")

                    # Extract unique countries and indicators from the dataset
                    if 'Country Name' in df.columns:
                        countries = df['Country Name'].unique().tolist()
                        indicators = df.columns[2:].tolist()  # Extract indicators starting from the third column
                        session['countries'] = countries
                        session['indicators'] = indicators
                        session['cleaned_file_path'] = cleaned_file_path
                        session['preprocessed'] = True  # Set flag to indicate preprocessing is complete
                    else:
                        return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error="Country column not found in dataset.")

                except Exception as e:
                    return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error=str(e))

        elif 'analyze' in request.form:
            selected_algo = request.form.get('algorithm', '')  # Retrieve the selected algorithm from the form
            print(f"Selected algorithm: {selected_algo}")  # Print the selected algorithm to verify

            cleaned_file_path = session.get('cleaned_file_path', '')

            if cleaned_file_path.endswith('.csv'):
                df = pd.read_csv(cleaned_file_path)
            elif cleaned_file_path.endswith('.xlsx'):
                df = pd.read_excel(cleaned_file_path)
            else:
                return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error="Unexpected error after preprocessing.")

            output_dir = app.config['PLOT_FOLDER']

            try:
                if selected_algo == "EDA":
                    EDA.analyze(df, selected_country, selected_indicator, output_dir)
                    result = "EDA completed."
                elif selected_algo == "ARIMA":
                    result = ARIMA.analyze(df, selected_country, selected_indicator, output_dir)
                elif selected_algo == "Linear Regression":
                    result = Linear_Regression.analyze(df, selected_country, selected_indicator, output_dir)
                elif selected_algo == "K-Means Clustering":
                    result = K_Means_Clustering.analyze(df, selected_country, selected_indicator, output_dir)
                else:
                    result = "Algorithm not implemented or selected."

                # Collect paths of saved plots and Excel files
                plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
                plot_paths = [os.path.join('plots', f) for f in plot_files]

                excel_files = [f for f in os.listdir(output_dir) if f.endswith('.xlsx')]
                excel_paths = [os.path.join('plots', f) for f in excel_files]

                # Read Excel files for preview
                for excel_file in excel_files:
                    excel_path = os.path.join(output_dir, excel_file)
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    for sheet_name, sheet_data in excel_data.items():
                        excel_previews.append({
                            'filename': excel_file,
                            'sheet_name': sheet_name,
                            'data': sheet_data.head().to_html(classes='table table-bordered')
                        })

                return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, result=result, plot_paths=plot_paths, excel_previews=excel_previews)

            except Exception as e:
                return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, error=str(e))

    preprocessed = session.get('preprocessed', False)
    return render_template("upload.html", algorithms=algorithms, selected_algo=selected_algo, countries=countries, indicators=indicators, selected_country=selected_country, selected_indicator=selected_indicator, preprocessed=preprocessed)

if __name__ == '__main__':
    app.run(debug=True)