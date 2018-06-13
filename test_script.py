import numpy as np
import OffAxisProjection
import OffAxisProjection.OffAxisProjection as OffAP
import OffAxisProjection.OnAxisProjection as OnAP
import pysplat as ps

def test_rotation_matrix():
    normal_vector = np.array([3, 4, 5], dtype='float')
    mat1 = OffAP.get_rotation_matrix(normal_vector)
    mat2 = OffAP.get_rotation_matrix_COB(normal_vector)
    print(OffAP.get_rotation_matrix(normal_vector))
    print(OffAP.get_rotation_matrix_COB(normal_vector))
    #assert np.allclose(mat1, mat2, rtol=1e-09)


def test_rotation():
    normal_vector = np.array([0., 1., 0.], dtype='float')
    test_point = np.array([1., 0., 0.], dtype='float')
    rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
    #print(rotation_matrix)
    print(np.matmul(rotation_matrix, test_point))
    #assert np.shape(rotation_matrix) == (3, 3)


def test_projection():
    """ Creates a PNG file with randomy generated pixels
    """
    num_particles = 3000

    # px and py contains randomly generated values between 0 and 1
    px = np.random.random(num_particles)
    py = np.random.random(num_particles)
    particle_masses = np.random.random(num_particles)
    bottom_left = [0, 0]
    top_right = [1, 1]
    normal_vector = np.array([0, 0, 1])
    resolution = (512, 512)
    projection_array = np.zeros(resolution)
    OffAP.create_projection(px, py, particle_masses, bottom_left, top_right,
                            normal_vector, projection_array)
    projection_array_1 = np.zeros(resolution)
    OnAP.create_projection(px, py, particle_masses, bottom_left, top_right,
                           projection_array_1)
    assert np.allclose(projection_array, projection_array_1, rtol=1e-09)


def main():
    test_projection()

if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
    main()
