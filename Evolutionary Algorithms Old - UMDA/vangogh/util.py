from PIL import Image
import os

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

WD = os.getcwd()
REFERENCE_IMAGE = Image.open(f"{WD}/img/reference_image_resized.jpg").convert('RGB')
