from flask import Flask, request

app = Flask(__name__)

@app.post("/audio")
def audio():
    data = request.get_data()

    print("Received:", len(data), "bytes")

    with open("test-servers/audio_chunk.raw", "ab") as f:
        f.write(data)

    return "OK", 200


app.run(host="0.0.0.0", port=8000)