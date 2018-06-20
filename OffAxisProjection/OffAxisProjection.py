import numpy as np
import numpy.linalg as linalg


def get_rotation_matrix(normal_vector):
    """ Returns a numpy rotation matrix corresponding to the
    rotation of the z-axis ([0, 0, 1]) to a given normal vector
    """
    if np.shape(normal_vector) != (3, ):
        return -1

    z_axis = np.array([0., 0., 1.], dtype='float_')
    normal_unit_vector = normal_vector / linalg.norm(normal_vector)
    v = np.cross(z_axis, normal_unit_vector)
    s = linalg.norm(v)
    c = np.dot(z_axis, normal_unit_vector)
    # if the normal vector is identical to the z-axis, just return the
    # identity matrix
    if np.isclose(c, 1, rtol=1e-09):
        return np.identity(3, dtype='float_')
    # if the normal vector is the negative z-axis, return error
    if np.isclose(s, 0, rtol=1e-09):
        return np.zeros((3, 3), dtype='float_')

    cross_product_matrix = np.array([[0, -1 * v[2], v[1]],
                                    [v[2], 0, -1 * v[0]],
                                    [-1 * v[1], v[0], 0]], dtype='float_')
    rotation_matrix = np.identity(3) + cross_product_matrix \
        + cross_product_matrix @ cross_product_matrix \
        * 1/(1+c)
    return rotation_matrix


def create_projection(px, py, pz, particle_masses, bottom_left, top_right,
                      projection_array, normal_vector=np.array([0, 0, 1])):
    """ Creates a numpy array with the particle mass density of each pixel
        Viewed from the perspective of normal_vector
        Function can take the whole dataset or data in chunks
        Function modifies projection_array
    """
    if np.allclose(normal_vector, np.array([0., 0., 0.]), rtol=1e-09):
        return

    resolution = np.shape(projection_array)
    num_particles = min(np.size(px), np.size(py), np.size(pz),
                        np.size(particle_masses))
    density_array = np.zeros(resolution, dtype='float_')
    dx = (top_right[0] - bottom_left[0])/resolution[0]
    dy = (top_right[1] - bottom_left[1])/resolution[1]
    square_area = dx * dy
    rotation_matrix = get_rotation_matrix(normal_vector)

    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        z_coordinate = pz[i]
        if x_coordinate > top_right[0] or y_coordinate > top_right[1]:
            continue
        if x_coordinate < bottom_left[0] or y_coordinate < bottom_left[1]:
            continue
        if z_coordinate < bottom_left[2] or z_coordinate > top_right[2]:
            continue
        coordinate_matrix = np.array([x_coordinate, y_coordinate, z_coordinate], dtype='float_')
        new_coordinates = rotation_matrix @ coordinate_matrix

        image_pixel_coordinates = pixel_coordinates(new_coordinates[0],
                                                    new_coordinates[1],
                                                    top_right, bottom_left,
                                                    resolution)
        if image_pixel_coordinates[0] < 0 or image_pixel_coordinates[0] >= resolution[0]:
            continue
        if image_pixel_coordinates[1] < 0 or image_pixel_coordinates[1] >= resolution[1]:
            continue
        density_array[int(image_pixel_coordinates[0]),
                      int(image_pixel_coordinates[1])] += particle_masses[i]
    density_array /= square_area
    for index, i in np.ndenumerate(density_array):
        projection_array[index[0], index[1]] += \
            density_array[index[0], index[1]]


def pixel_coordinates(x, y, top_right, bottom_left, resolution):
    """ Outputs image coordinates given particle coordinates
    """
    x_normalized = (x - bottom_left[0])/(top_right[0] - bottom_left[0])
    y_normalized = (y - bottom_left[1])/(top_right[1] - bottom_left[1])
    return (x_normalized * resolution[0], y_normalized * resolution[1])
