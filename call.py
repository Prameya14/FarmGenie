from flask import Flask, Response

app = Flask(__name__)

@app.route("/")
def call():
    xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Dial callerId="08047359131"> +919861722466 </Dial>
        </Response>"""
    return Response(xml_response, mimetype="application/xml")


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
