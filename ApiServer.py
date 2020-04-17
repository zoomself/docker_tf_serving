import flask
import requests
import tensorflow as tf
import json


app = flask.Flask(__name__)

tf_serving_base_url = "http://192.168.99.100:8501/v1/models/"

MNIST_URL = tf_serving_base_url + "mnist:predict"
SAMPLE_URL = tf_serving_base_url + "sample:predict"


@app.route("/sample", methods=["GET"])
def sample():
    data = {"success": False, "msg": "request failed"}
    if flask.request.method == "GET":
        json_data = {
            "instances": [
                {
                    "a": 1,
                    "b": 1
                }
            ]
        }

        response = requests.post(SAMPLE_URL, json=json_data)
        response.raise_for_status()
        print(response)

        print(response.text.encode('utf8'))
        data["data"] = json.loads(response.text)["predictions"][0]
        data["msg"] = "request ok!"
        data["success"] = True

    return data


@app.route("/mnist", methods=["POST"])
def predict():
    data = {"success": False, "msg": "request failed"}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files.get("image").read()
            img = tf.image.decode_jpeg(img)
            img = tf.cast(img, tf.float32) / 255.
            img = img[tf.newaxis, ...]
            headers = {"content-type": "application/json"}
            print(img.numpy())
            print("-----------------\n")
            print(img.numpy().tolist())
            json_data = json.dumps(
                {"instances": img.numpy().tolist()})  # img.numpy()是一个eger tensor , json 里面必须要是utf-8编码的经过序列化的东西
            response = requests.post(MNIST_URL, data=json_data, headers=headers)
            response.raise_for_status()
            response_content = response.text
            result = json.loads(response_content)
            result = result["predictions"]
            data["data"] = []
            max_p = 0
            max_i = 0
            for i, r in enumerate(result[0]):
                p = float(r)
                if p >= max_p:
                    max_p = p
                    max_i = i
                r = {"label": i, "probability": p}
                data["data"].append(r)
            data["predict"] = {"label": max_i, "probability": max_p}
            data["msg"] = "request ok!"
            data["success"] = True
    return data


if __name__ == '__main__':
    app.run(debug=True)
    # test_img()
