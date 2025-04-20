from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, send_from_directory, render_template, redirect, url_for, flash
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
from time import time
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_name = request.form.get("model_name")

        try:
            filename = f"{model_name.replace('/', '-')}___{time()}"

            create_onnx_file(model_name, filename)

            return redirect(url_for("download_file", filename=filename + ".onnx"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


from transformers.onnx import FeaturesManager
from pathlib import Path

def create_onnx_file(model_name, output_filename):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    feature = "sequence-classification"
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)

    output_path = Path(OUTPUT_DIR) / f"{output_filename}.onnx"

    export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=output_path
    )
    print("Successfully exported ONNX file")


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=5000)
