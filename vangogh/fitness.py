import numpy as np
from PIL import Image
from imgcompare import image_diff
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree

from util import NUM_VARIABLES_PER_POINT, REFERENCE_IMAGE

QUERY_POINTS = []


def draw_voronoi_matrix(genotype, img_width, img_height, scale = 1):
	scaled_img_width = int(img_width * scale)
	scaled_img_height = int(img_height * scale)
	num_points = int(len(genotype) / NUM_VARIABLES_PER_POINT)
	coords = []
	colors = []

	for r in range(num_points):
		p = r * NUM_VARIABLES_PER_POINT
		x, y, r, g, b = genotype[p:p + NUM_VARIABLES_PER_POINT]
		coords.append((x * scale, y * scale))
		colors.append((r, g, b))

	voronoi_kdtree = KDTree(coords)
	if scale == 1:
		query_points = [(x, y) for x in range(scaled_img_width) for y in range(scaled_img_height)]
	else:
		query_points = [(x, y) for x in range(scaled_img_width) for y in range(scaled_img_height)]

	_, query_point_regions = voronoi_kdtree.query(query_points)

	data = np.zeros((scaled_img_height, scaled_img_width, 3), dtype = 'uint8')
	i = 0
	for x in range(scaled_img_width):
		for y in range(scaled_img_height):
			for j in range(3):
				data[y, x, j] = colors[query_point_regions[i]][j]
			i += 1

	return data


def draw_voronoi_image(genotype, img_width, img_height, scale = 1) -> Image:
	data = draw_voronoi_matrix(genotype, img_width, img_height, scale)
	img = Image.fromarray(data, 'RGB')
	return img


def compute_difference(genotype, reference_image: Image):
	expected = reference_image
	actual = draw_voronoi_matrix(genotype, reference_image.width, reference_image.height)

	diff = image_diff(Image.fromarray(actual, 'RGB'), expected)
	return diff


def worker(args):
	return compute_difference(args[0], args[1])


def drawing_fitness_function(genes, reference_image: Image):
	if len(QUERY_POINTS) == 0:
		QUERY_POINTS.extend([(x, y) for x in range(reference_image.width) for y in range(reference_image.height)])

	with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
		fitness_values = list(p.map(worker, zip(genes, [reference_image] * genes.shape[0])))
	return np.array(fitness_values)


# Join pixel locations with pixel colours
def pixel_join_indv(pixel_loc, pixel_colour):
	a = pixel_loc.reshape((-1, 2))
	b = pixel_colour.reshape((-1, 3))
	gene = np.hstack((a, b)).reshape(-1)
	return gene


def individual_fitness(pixel_colour, pixel_loc, reference_image = REFERENCE_IMAGE):
	fitness = worker((pixel_join_indv(pixel_loc, pixel_colour), reference_image))
	return fitness
