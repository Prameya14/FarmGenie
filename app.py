from flask import Flask, request, render_template, Response
import pickle
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import cv2
import io
from werkzeug.datastructures import FileStorage
import tempfile

with open("model.pkl", "rb") as file:
    model2 = pickle.load(file)

df = pandas.read_csv("[Dataset] Crop Data.csv")

app = Flask(__name__)

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = torch.device("cpu")


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return classes[preds[0].item()]


model = ResNet9(3, 38)
model.load_state_dict(
    torch.load("plant-disease-model.pth", map_location=torch.device("cpu"))
)
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crop-recommendation-system", methods=["POST", "GET"])
def desiredCrop():
    if request.method == "POST":
        params = request.form
        N = float(params["nRatio"])
        P = float(params["pRatio"])
        K = float(params["kRatio"])
        temp = float(params["temp"])
        humidity = float(params["humidity"])
        pH = float(params["pH"])
        rainfall = float(params["rainfall"])
        crop = model2.predict([[N, P, K, temp, humidity, pH, rainfall]])[0]

        cropDf = df.loc[df["label"] == crop]
        nList = list(cropDf["N"])
        pList = list(cropDf["P"])
        kList = list(cropDf["K"])
        tempList = list(cropDf["temperature"])
        humList = list(cropDf["humidity"])
        phList = list(cropDf["ph"])
        rainList = list(cropDf["rainfall"])
        details = [
            crop.title(),
            nList,
            pList,
            kList,
            tempList,
            humList,
            phList,
            rainList,
        ]
        return render_template("crop-rec.html", details=details)
    return render_template("crop-rec.html", details="")


def fetch_details(crop, disease):
    details = pandas.read_csv("details.csv")
    token = [crop, disease]
    n = len(list(details["crop"]))
    for i in range(0, n):
        if (
            token[0] == list(details["crop"])[i]
            and token[1] == list(details["disease"])[i]
        ):
            cause = list(details["causes"])[i].split(". ")
            remedial_measure = list(details["remedial_measures"])[i].split(". ")
            return [cause, remedial_measure]


def capture_image():
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    frame_resized = cv2.resize(frame, (256, 256))

    return frame_resized


def cv2_to_pillow(cv2_image):
    ret, buffer = cv2.imencode(".jpg", cv2_image)
    byte_stream = io.BytesIO(buffer)
    pillow_image = Image.open(byte_stream)
    return pillow_image


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def generate_frames():
    camera = cv2.VideoCapture(3, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_resized = cv2.resize(frame, (256, 256))
            ret, buffer = cv2.imencode(".jpg", frame_resized)

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


@app.route("/crop-disease-prediction-system", methods=["GET", "POST"])
def disease_prediction():
    if request.method == "POST":
        i = capture_image()
        i2 = cv2_to_pillow(i)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            i2.save(temp_file.name)
            temp_file.close()

            with open(temp_file.name, "rb") as file:
                file_storage = FileStorage(file)

                img = Image.open(file_storage).convert("RGB")

                to_tensor = transforms.ToTensor()
                img2 = to_tensor(img)
                prediction = predict_image(img2, model)
                crop = prediction.split("___")[0].replace("_", " ").title().strip()
                disease = prediction.split("___")[1].replace("_", " ").title().strip()

                image_bytes = BytesIO()
                img.save(image_bytes, format="JPEG")
                image_bytes.seek(0)

                base64_image = base64.b64encode(image_bytes.read())
                base64_string = base64_image.decode("utf-8")

                fd = fetch_details(crop.lower(), disease.lower())

                if fd == None:
                    fd = [[], []]

                details = [crop, disease, base64_string, fd[0], fd[1]]

                return render_template("disease-pred.html", details=details)
    return render_template("disease-pred.html", details="")


@app.route("/crop-disease-prediction-system-2", methods=["GET", "POST"])
def disease_prediction_2():
    if request.method == "POST":
        image = request.files["image"]
        img = Image.open(image).convert("RGB").resize((256, 256))
        to_tensor = transforms.ToTensor()
        img2 = to_tensor(img)
        prediction = predict_image(img2, model)
        crop = prediction.split("___")[0].replace("_", " ").title().strip()
        disease = prediction.split("___")[1].replace("_", " ").title().strip()

        image_bytes = BytesIO()
        img.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        base64_image = base64.b64encode(image_bytes.read())
        base64_string = base64_image.decode("utf-8")

        fd = fetch_details(crop.lower(), disease.lower())
        if fd == None:
            fd = [[], []]

        details = [crop, disease, base64_string, fd[0], fd[1]]

        return render_template("disease-pred-2.html", details=details)
    return render_template("disease-pred-2.html", details="")


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
