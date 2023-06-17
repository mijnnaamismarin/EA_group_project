from PIL import Image

# Get image path
path = "C:/Users/David/Desktop/Master/Q4/Evolutionary Algorithms Heuristics/img"

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

REFERENCE_IMAGE = Image.open(f"{path}/reference_image_resized.jpg").convert('RGB')
