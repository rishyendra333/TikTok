import ffmpeg
import io
import numpy as np
import tempfile
from PIL import Image
from sentence_transformers import models, SentenceTransformer

# Initialize the CLIP model
img_model = SentenceTransformer(modules=[models.CLIPModel()])

def get_scaled_size(width, height):
    target_width = 224
    w_percent = (target_width / float(width))
    h_size = int((float(height) * float(w_percent)))
    return target_width, h_size

def to_byte_array(image):
    """Converts a numpy array image to a byte array"""
    img = Image.fromarray(image)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()

def get_frames(video_path):
    with tempfile.NamedTemporaryFile() as f:
        # Read the video file
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        
        f.write(video_bytes)
        f.flush()
        
        probe = ffmpeg.probe(f.name, threads=1)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width, height = get_scaled_size(int(video_info['width']), int(video_info['height']))

        out, _ = (
            ffmpeg
            .input(f.name, threads=1)
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
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    vectors = img_model.encode(images)
    return [vector.tolist() for vector in vectors]

# Path to your video file
video_path = "video1881698513.mp4"

# Get frames from the video
frames = get_frames(video_path)

# Get embeddings for the frames
embeddings = get_embeddings(frames)

# # Print the embeddings
# for idx, emb in enumerate(embeddings):
#     print(f"Embedding {idx}: {emb}")

print(len(embeddings))