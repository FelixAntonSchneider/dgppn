import requests
from PIL import Image
from io import BytesIO
import time
import base64

def image_to_base64(pil_image, format="PNG"):
    """
    Converts a PIL.Image object to a base64 encoded string.

    Args:
        pil_image (PIL.Image.Image): The PIL image to encode.
        format (str): The format to save the image in (e.g., "PNG", "JPEG").

    Returns:
        str: The base64 encoded representation of the image.
    """
    # Save the image to an in-memory byte stream
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)  # Reset the buffer pointer to the start

    # Get the byte data and encode it to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64

url = "https://api.bfl.ml/v1/flux-pro-1.1-ultra"

payload = {
    "prompt": "A beautiful landscape with mountains and a lake",
    "seed": 42,
    "aspect_ratio": "16:9",
    "safety_tolerance": 6,
    "output_format": "jpeg",
    "raw": False,
    "image_prompt_strength": 0.1
}
headers = {
    "Content-Type": "application/json",
    "X-Key": "bfl_key"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
time.sleep(10)

img_id = response.json()['id']
retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)

img_url = retrieval.json()['result']['sample']
imgres = requests.get(img_url)
image = Image.open(BytesIO(imgres.content))
image.show()

b64_img = image_to_base64(pil_image=image)

url = "https://api.bfl.ml/v1/flux-pro-1.1-ultra"

payload = {
    "prompt": "extend the uploaded image",
    "seed": 42,
    "aspect_ratio": "16:9",
    "safety_tolerance": 6,
    "output_format": "jpeg",
    "image_prompt": b64_img,
    "raw": False,
    "image_prompt_strength": 0.1
}
headers = {
    "Content-Type": "application/json",
    "X-Key": "bfl_key"
}

response = requests.post(url, json=payload, headers=headers)
time.sleep(10)

img_id = response.json()['id']
retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)

img_url = retrieval.json()['result']['sample']
imgres = requests.get(img_url)
diffimage = Image.open(BytesIO(imgres.content))
diffimage.show()

#print(response.json())