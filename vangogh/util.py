from PIL import Image

# Get image path
path = "img"

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

REFERENCE_IMAGE = Image.open(f"{path}/reference_image_resized.jpg").convert('RGB')
