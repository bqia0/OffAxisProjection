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

    z_axis = np.array([0., 0., 1.], dtype='float')
    normal_unit_vector = np.divide(normal_vector,
                                   linalg.norm(normal_vector))
    v = np.cross(z_axis, normal_unit_vector)
    s = linalg.norm(v)
    # we can't have identical or parallel vectors
    if np.isclose(s, 0, rtol=1e-09):
        return -1
    c = np.dot(z_axis, normal_unit_vector)
    cross_product_matrix = np.matrix([[0, -1 * v[2], v[1]],
                                      [v[2], 0, -1 * v[0]],
                                      [-1 * v[1], v[0], 0]], dtype='float')
    rotation_matrix = np.identity(3) + cross_product_matrix \
        + np.matmul(cross_product_matrix, cross_product_matrix) \
        * 1/(1+c)
    return rotation_matrix
