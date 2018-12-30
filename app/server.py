from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np
import os


path = Path(__file__).parent
model_file_url = ''#'YOUR MODEL.h5 DIRECT / RAW DOWNLOAD URL HERE!'
dl_type = 'gdrive' # | 'raw','gdrive'
model_file_name = 'model'
with open('config.json') as f:
    config = json.load(f)
    
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = path/'static'/'saved_image.png'
PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED 
    if dl_type == 'raw':
        await download_file(model_file_url, MODEL_PATH)
    elif dl_type == 'gdrive':
        from google.cloud import storage
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config['gdrive_key']
        storage_client = storage.Client()
        bucket_name = 'my-new-bucket-ssss'
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob('test')
        blob.download_to_filename(MODEL_PATH)

    model = load_model(str(MODEL_PATH)) # Load your Custom trained model
    model._make_predict_function()
    # model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["img"].read())
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return model_predict(IMG_FILE_SRC, model)

def model_predict(img_path, model):
    result = []; img = image.load_img(img_path, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    # predictions = decode_predictions(model.predict(x), top=3)[0] # Get Top-3 Accuracy
    predictions = model.predict(x)
    class_dict = {0:"class1", 1:"class2"} # currenlty only binary
    result = class_dict[np.argmax(predictions[0], axis =1)]
    # for p in predictions: _,label,accuracy = p; result.append((label,accuracy))
    with open(PREDICTION_FILE_SRC, 'w') as f: f.write("the iamge is more "+str(result)+" confidence"+np.max(preidctions[0]))
    result_html = path/'static'/'result.html'
    return HTMLResponse(result_html.open().read())

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8081)