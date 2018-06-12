import numpy as np


def create_projection(px, py, particle_masses, bottom_left, top_right,
                      projection_array):
    """ Outputs a numpy array with the particle mass density of each pixel
    """

    resolution = np.shape(projection_array)
    num_particles = min(np.size(px), np.size(py), np.size(particle_masses))
    dx = (top_right[0] - bottom_left[0])/resolution[0]
    dy = (top_right[1] - bottom_left[1])/resolution[1]
    square_area = dx * dy
    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        image_pixel_coordinates = pixel_coordinates(x_coordinate, y_coordinate,
                                                    top_right, bottom_left,
                                                    resolution)
        projection_array[int(image_pixel_coordinates[0]),
                         int(image_pixel_coordinates[1])] += particle_masses[i]
    projection_array /= square_area


def pixel_coordinates(x, y, top_right, bottom_left, resolution):
    """ Outputs image coordinates given particle coordinates
    """
    x_normalized = (x - bottom_left[0])/(top_right[0] - bottom_left[0])
    y_normalized = (y - bottom_left[1])/(top_right[1] - bottom_left[1])
    return (x_normalized * resolution[0], y_normalized * resolution[1])
