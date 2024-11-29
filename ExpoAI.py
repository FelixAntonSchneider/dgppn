from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import argparse
import time
import re
from api_keys import open_ai_key, bfl_key
import cv2
import numpy as np
import base64

parser = argparse.ArgumentParser(description="Prompt AI model")
parser.add_argument("--patient_text", help="balbal")
args = parser.parse_args()

client = OpenAI(api_key=open_ai_key) #insert key

def lazy_prop(prop):
    """Decorator which implements lazy loading/computing functionality for instance
       properties. Requires consistent naming of property methods and hidden variables"""

    def inner(self, *args, **kwargs):
        if hasattr(self, '_' + prop.__name__):
            return self.__getattribute__('_' + prop.__name__)
        else:
            return prop(self)

    return inner


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


# def numpy_to_base64(image_array):
#     """
#     Converts a NumPy array representing an RGB image into a Base64 string.
#
#     Parameters:
#         image_array (numpy.ndarray): Input RGB image as a NumPy array.
#
#     Returns:
#         str: Base64 string representation of the image.
#     """
#     # Convert the NumPy array to a PIL Image
#     image = Image.fromarray(image_array)
#
#     # Save the image to a BytesIO object in memory
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")  # Use PNG or any format you prefer
#
#     # Encode the binary data to Base64
#     base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#
#     return base64_str


def numpy_to_base64(image_array):
    """
    Converts a NumPy array representing an RGB image into a Base64 string.
    """
    _, encoded_image = cv2.imencode('.png', image_array)  # Convert to PNG format
    return base64.b64encode(encoded_image).decode('utf-8')


def shift_image_with_mask(image, dx, dy):
    """
    Shifts an image by (dx, dy) and generates a binary mask highlighting the blank areas.

    Parameters:
        image (numpy.ndarray): Input image.
        dx (int): Horizontal shift (positive = right, negative = left).
        dy (int): Vertical shift (positive = down, negative = up).

    Returns:
        shifted_image (numpy.ndarray): The shifted image with blank areas filled with black.
        mask (numpy.ndarray): Binary mask of the blank areas (1 = blank, 0 = original image).
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Create translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Shift the image using warpAffine
    shifted_image = cv2.warpAffine(image, translation_matrix, (width, height), borderValue=(0, 0, 0))

    # Create a blank mask of the same size as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the area covered by the original image after shifting
    x_start = max(dx, 0)
    x_end = min(width + dx, width)
    y_start = max(dy, 0)
    y_end = min(height + dy, height)

    # Set the covered area to  in the mask
    mask[y_start:y_end, x_start:x_end] = 1

    # Invert the mask to highlight blank areas
    #mask = 1 - mask

    return shifted_image, mask


class ExpoAI:

    def __init__(self, patient_text=None, item_context=None, image_context=None, audiofilepath=None):
        """Initialize ExpoAI object only with patient description"""
        self._item_context = item_context
        self._image_context = image_context
        self.audiofilepath = audiofilepath

        if patient_text is not None:
            self.patient_text = patient_text
        elif audiofilepath is not None:
            self.patient_text = self.read_and_transform_audio()
        else:
            raise ValueError("patient_text and audiofilepath cannot both be None")

    def read_and_transform_audio(self):

        audio_file = open(self.audiofilepath, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text

    @property
    def image_context(self):
        return self._image_context

    @image_context.setter
    def image_context(self, value):
        self._image_context = value
        del self._image_gen_prompt
        self.update_image(prompt=value)

    @property
    def item_context(self):
        return self._item_context

    @item_context.setter
    def item_context(self, value):
        self._item_context = value
        del self._psy_text
        del self._vis_items
        del self._image_gen_prompt
        del self._image

    def psy_analyze(self):

        prime_string = "You are a renowned professional psychiatrist with high competence. You will here a patients description of his or her anxiety problems or traumatic experiences in light prosa. \
                        Your task is to think carefully what exactly the patient's history is that led to his or her issue. For a potential exposure therapy try to think carefully what the actual \
                        triggers are of the patient and how the exposure should look like. Make a list of static items which should be part of the visual exposure with a single static image. Ask the patient for more detail if you think \
                        important information is missing. Also be concise! Remember all items should be consistent with being a on a single image."

        list_instruct = " List the visual items as bullet points directly under a header called 'vis_items:'!"
        if self.item_context is not None:
            prime_string = prime_string + self.item_context + list_instruct
        else:
            prime_string = prime_string + list_instruct

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":"system",
                    "content": prime_string
                },
                {
                    "role": "user",
                    "content": self.patient_text
                }
            ]
        )

        return response.choices[0].message.content

    @property
    @lazy_prop
    def psy_text(self):
        self._psy_text = self.psy_analyze()
        return self._psy_text

    @property
    @lazy_prop
    def vis_items(self):

        # Regex pattern to match bullet points starting after 'vis_items:'
        pattern = r"vis_items:\n((?:-.*\n)+)"

        # Search for the pattern in the text
        match = re.search(pattern, self.psy_text)

        if match:
            # Extract and split the bullet points into a list
            bullet_points = [line.strip('- ').strip() for line in match.group(1).splitlines()]
            self._vis_items = bullet_points
        else:
            self._vis_items = None

        return self._vis_items

    @property
    @lazy_prop
    def image_gen_prompt(self):

        # generate image generation prompt
        prompt_string = ("You are an expert for the generation of visual images for exposure therapy for anxiety and PTSD patients. \
                         You will receive a list of items and scene aspects which should be part of the image you are about to generate. \
                         The image should be from the perspective of the patient as if he or she was in the situation described by the \
                         list items. Try to make it as immersive as possible. It should be an image of a REAL world scene. NO overlay text on the image!")

        if self.image_context is not None:
            prompt_string += self.image_context

        prompt_string += ' List of items:'
        for item in self.vis_items:
            prompt_string += f"""\n - {item}"""

        self._image_gen_prompt = prompt_string
        return self._image_gen_prompt

    def image_gen(self):

        # Black Forest Labs gen
        url = "https://api.bfl.ml/v1/flux-pro-1.1-ultra"

        payload = {
            "prompt": self.image_gen_prompt,
            "seed": 42,
            "aspect_ratio": "4:3",
            "safety_tolerance": 6,
            "output_format": "jpeg",
            "raw": False,
            "image_prompt_strength": 0.1
        }
        headers = {
            "Content-Type": "application/json",
            "X-Key": bfl_key
        }

        response = requests.post(url, json=payload, headers=headers)

        time.sleep(10)

        img_id = response.json()['id']
        retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
        ret = False
        while not ret:
            retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
            if retrieval.json()['status'] == 'Ready':
                ret = True

        img_url = retrieval.json()['result']['sample']
        imgres = requests.get(img_url)
        image = Image.open(BytesIO(imgres.content))
        return image


        # DALL-E gen
        # response = client.images.generate(
        #     model="dall-e-3",
        #     prompt=self.image_gen_prompt,
        #     size="1024x1024",
        #     quality="standard",
        #     n=1
        # )
        #
        # image_url = response.data[0].url
        #
        # # Fetch the image data
        # response = requests.get(image_url)
        # response.raise_for_status()  # Will raise an exception for bad status codes
        #
        # # View the image
        # image = Image.open(BytesIO(response.content))
        # return image

    @property
    @lazy_prop
    def image(self):
        if self.vis_items is None:
            self._image = None
        else:
            self._image = self.image_gen()
        return self._image

    def show_gen_im(self):
        self.image.show()

    def image_extension(self, dx, dy):

        img_arr = np.array(self.image)
        shifted, mask = shift_image_with_mask(img_arr, dx=dx, dy=dy)

        b64_shifted = numpy_to_base64(shifted)
        b64_mask = numpy_to_base64(mask)
        url = "https://api.bfl.ml/v1/flux-pro-1.0-fill"

        payload = {
            "image": b64_shifted,
            "mask": b64_mask,
            "prompt": "Fill the masked area with a natural extension of the image, maintaining: \
                    - **Context**: Reflect the scene, setting, or environment shown in the rest of the image. \
                    - **Continuity**: Ensure elements like patterns, textures, and objects flow seamlessly from the existing parts of the image into the masked area. \
                    - **Lighting**: Match the lighting, shadows, and color tones seen in the unmasked parts.\
                    - **Details**: Keep the level of detail consistent, including any background elements or minor features.\
                    - **Mood**: Preserve the overall mood or atmosphere of the image.\
                    \
                    **Ensure no new, unrelated elements are added; the extension should look as if it was always part of the original image.**",
            "steps": 50,
            "prompt_upsampling": False,
            "guidance": 60,
            "output_format": "jpeg",
            "safety_tolerance": 6
        }
        headers = {
            "Content-Type": "application/json",
            "X-Key": bfl_key
        }

        response = requests.post(url, json=payload, headers=headers)

        img_id = response.json()['id']
        ret = False
        retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
        while not ret:
            retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
            if retrieval.json()['status'] == 'Ready':
                ret = True

        img_url = retrieval.json()['result']['sample']

        imgres = requests.get(img_url)
        image = Image.open(BytesIO(imgres.content))
        return image, shifted, mask
    
    def update_image(self, prompt):

        b64_img = image_to_base64(self.image)

        url = "https://api.bfl.ml/v1/flux-pro-1.1-ultra"

        payload = {
            "prompt": prompt,
            "image_prompt": b64_img,
            "seed": 42,
            "aspect_ratio": "4:3",
            "safety_tolerance": 6,
            "output_format": "jpeg",
            "raw": False,
            "image_prompt_strength": 0.1
        }
        headers = {
            "Content-Type": "application/json",
            "X-Key": bfl_key
        }

        response = requests.post(url, json=payload, headers=headers)

        img_id = response.json()['id']
        ret = False
        retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
        while not ret:
            retrieval = requests.get(f"""https://api.bfl.ml/v1/get_result?id={img_id}""", headers=headers)
            if retrieval.json()['status'] == 'Ready':
                ret = True

        img_url = retrieval.json()['result']['sample']

        imgres = requests.get(img_url)
        image = Image.open(BytesIO(imgres.content))
        self._image = image

