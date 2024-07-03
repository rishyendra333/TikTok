import io
import random
import tempfile
import uuid
from fastapi import FastAPI, File, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import ffmpeg
import numpy as np
from PIL import Image as Img
from sentence_transformers import models, SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='44a257b1-5d26-4544-9825-d51ddaaa13e3')

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

# Allow all origins during development; restrict it as needed for production
origins = ["*"]

img_model = SentenceTransformer(modules=[models.CLIPModel()])

def get_scaled_size(width, height):
    target_width = 224
    w_percent = (target_width / float(width))
    h_size = int((float(height) * float(w_percent)))
    return target_width, h_size

def to_byte_array(image):
    """Converts a numpy array image to a byte array"""
    img = Img.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()

def get_frames(video: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video.file.read())
        temp_file.flush()
        
        probe = ffmpeg.probe(temp_file.name, threads=1)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width, height = get_scaled_size(int(video_info['width']), int(video_info['height']))

        out, _ = (
            ffmpeg
            .input(temp_file.name, threads=1)
            .filter('scale', width, height)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        frames = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        
        indexes = np.random.randint(frames.shape[0], size=10)
        return [to_byte_array(frame) for frame in frames[indexes, :]]

def get_embeddings(frames):
    images = [Img.open(io.BytesIO(frame)) for frame in frames]
    vectors = img_model.encode(images)
    return vectors

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)



@app.get('/')
def index(req: Request):
    return templates.TemplateResponse(
        name="index.html",
        context={"request": req}
    )


@app.get("/form")
def form_post(request: Request):
    result = "Enter your name"
    return templates.TemplateResponse(
        "index.html", context={"request": request, "result": result}
    )


@app.post("/form")
def form_post(request: Request, name: str = Form(...)):
    result = name.capitalize()
    return templates.TemplateResponse(
        "index.html", context={"request": request, "result": result, "name": name}
    )

@app.post("/videoupload")
async def vidupload(request: Request, video: UploadFile = File(...)):
    frames = get_frames(video)
    # Process frames here (for example, save frames as images)
    embeddings = get_embeddings(frames)
    random_num = str(uuid.uuid4())
    vectors_with_metadata = [
    {
        "id": random_num+"-"+str(i),
        "values": embedding.tolist(),  # Convert embedding to list if it's a numpy array
    }
    for i, embedding in enumerate(embeddings)
    ]
    index_name = "tiktok-hackathon"
    index = pc.Index(index_name)
    index.upsert(vectors=vectors_with_metadata, namespace="video_embeddings")
    return "done"

@app.post("/search")
async def search(request: Request, query: str = Form(...)):
    # Encode the query using CLIP model
    query_embedding = img_model.encode([query])[0]

    # Search Pinecone index for the best match
    index_name = "tiktok-hackathon"
    index = pc.Index(index_name)
    query_results = index.query(
    namespace="video_embeddings",
    vector=query_embedding.tolist(),
    top_k=200,
    include_values=True,
    include_metadata=True
    )

    # Print the query results
    # print(query_results['matches'][0]['id'][:-2])    
    # print(query_results['matches'])
    matches = [i['id'] for i in query_results['matches']]
    print(matches)

    # if search_results and len(search_results) > 0:
    #     best_match = search_results[0]["id"]
    #     return templates.TemplateResponse(
    #         "search_results.html",
    #         {"request": request, "query": query, "best_match": best_match}
    #     )
    # else:
    #     return "No match found."
    return "done"
