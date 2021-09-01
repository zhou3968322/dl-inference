import os,sys

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, "../../"))
print(PROJECT_ROOT_PATH)
import requests, json, base64


url = "http://127.0.0.1:51000/predict"

image_path = os.path.join(PROJECT_ROOT_PATH, "data/testdata/document.jpeg")

with open(image_path, "rb") as fr:
    result = requests.post(url, json={"image": base64.b64encode(fr.read()).decode()}).text
res_dict = json.loads(result)

if res_dict["code"] == 200:
    for res_box in res_dict["box_list"]:
        text = "".join([chr(char_int) for char_int in res_box["text"]])
        print(text)

