import numpy as np
import numpy.linalg as linalg


def get_rotation_matrix_COB(normal_vector):
    """ Returns a numpy rotation matrix corresponding to the
    rotation of the z-axis ([0, 0, 1]) to a given normal vector
    """
    if np.shape(normal_vector) != (3, ):
        return -1
    z_axis = np.array([0., 0., 1.], dtype='float')
    normal_unit_vector = np.divide(normal_vector,
                                   linalg.norm(normal_vector))
    # we can't have identical or parallel vectors
    if np.allclose(linalg.norm(np.cross(normal_unit_vector, z_axis)), 0,
                   rtol=1e-09):
        return -1
    # Construct change of basis matrix
    basis_vector_b = normal_unit_vector \
        - np.multiply(np.dot(normal_unit_vector, z_axis), z_axis)
    basis_unit_vector_b = np.divide(basis_vector_b,
                                    linalg.norm(basis_vector_b))
    change_of_basis = np.column_stack((z_axis, basis_unit_vector_b,
                                      np.cross(normal_unit_vector, z_axis)))

    # Construct rotation matrix around the Z-Axis
    Rz0 = np.array([np.dot(z_axis, normal_unit_vector),
                    -1 * linalg.norm(np.cross(z_axis, normal_unit_vector)),
                    0.], dtype='float')
    Rz1 = np.array([linalg.norm(np.cross(z_axis, normal_unit_vector)),
                    np.dot(z_axis, normal_unit_vector),
                    0.], dtype='float')
    Rz2 = np.array([0., 0., 1.], dtype='float')
    Rz = np.matrix([Rz0, Rz1, Rz2], dtype='float')

    # Build final Rotation Matrix
    rotation_matrix = np.matmul(np.matmul(change_of_basis, Rz),
                                linalg.inv(change_of_basis))
    return rotation_matrix


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
        return -1

    cross_product_matrix = np.array([[0, -1 * v[2], v[1]],
                                    [v[2], 0, -1 * v[0]],
                                    [-1 * v[1], v[0], 0]], dtype='float_')
    rotation_matrix = np.identity(3) + cross_product_matrix \
        + cross_product_matrix @ cross_product_matrix \
        * 1/(1+c)
    return rotation_matrix


def create_projection(px, py, particle_masses, bottom_left, top_right,
                      normal_vector, projection_array):
    """ Creates a numpy array with the particle mass density of each pixel
        Viewed from the perspective of normal_vector
        Function can take the whole dataset or data in chunks
        Function modifies projection_array
    """
    # default normal vector
    if normal_vector is None:
        normal_vector = np.array([0, 0, 1])

    resolution = np.shape(projection_array)
    num_particles = min(np.size(px), np.size(py), np.size(particle_masses))
    density_array = np.zeros(resolution, dtype='float_')
    dx = (top_right[0] - bottom_left[0])/resolution[0]
    dy = (top_right[1] - bottom_left[1])/resolution[1]
    square_area = dx * dy
    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        if x_coordinate > top_right[0] or y_coordinate > top_right[1]:
            continue
        if x_coordinate < bottom_left[0] or y_coordinate < bottom_left[1]:
            continue
        image_pixel_coordinates = pixel_coordinates(x_coordinate, y_coordinate,
                                                    top_right, bottom_left,
                                                    resolution)
        density_array[int(image_pixel_coordinates[0]),
                      int(image_pixel_coordinates[1])] += particle_masses[i]
    density_array /= square_area

    # Construct projected array
    rotation_matrix = get_rotation_matrix(normal_vector)
    for index, i in np.ndenumerate(density_array):
        coordinate_matrix = np.array([index[0], index[1], 0.], dtype='float_')
        new_coordinates = rotation_matrix @ coordinate_matrix
        # outside range of image
        if new_coordinates[0] >= resolution[0] \
                or new_coordinates[1] >= resolution[1]:
            continue
        if new_coordinates[0] < 0 or new_coordinates[1] < 0:
            continue

        projection_array[int(new_coordinates[0]), int(new_coordinates[1])] \
            += density_array[index[0], index[1]]


def pixel_coordinates(x, y, top_right, bottom_left, resolution):
    """ Outputs image coordinates given particle coordinates
    """
    x_normalized = (x - bottom_left[0])/(top_right[0] - bottom_left[0])
    y_normalized = (y - bottom_left[1])/(top_right[1] - bottom_left[1])
    return (x_normalized * resolution[0], y_normalized * resolution[1])
