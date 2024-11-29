from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import argparse
import re
from api_keys import open_ai_key

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
        del self._image

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

        response = client.images.generate(
            model="dall-e-3",
            prompt=self.image_gen_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url

        # Fetch the image data
        response = requests.get(image_url)
        response.raise_for_status()  # Will raise an exception for bad status codes

        # View the image
        image = Image.open(BytesIO(response.content))
        return image

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


