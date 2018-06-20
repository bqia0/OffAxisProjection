import numpy as np
import OffAxisProjection.OnAxisProjection as OnAP
import OffAxisProjection.OffAxisProjection as OffAP
import OffAxisProjection.SPH as SPH

class TestClass(object):
    def test_density_array_correct_sum(self):
        """Use this test to see if the density array has the
           correct elements BEFORE dividing by area
           To use this test, comment out the division by area in
           pixel_densities
        """
        num_particles = 3000

        # px and py contains randomly generated values between 0 and 1
        px = np.random.random(num_particles)
        py = np.random.random(num_particles)
        particle_masses = np.random.random(num_particles)

        bottom_left = [0, 0]
        top_right = [1, 1]

        resolution = (512,512)
        # assert np.isclose(np.sum(pd.pixel_densities(px, py, particle_masses,
        #                                             bottom_left,
        #                                             top_right, resolution,
        #                                             num_particles)),
        #                   np.sum(particle_masses), rtol=1e-09)

    def test_density_array_correct(self):
        """Use this test to see if the density array has the
           correct elements AFTER dividing by area
        """
        num_particles = 3000

        # px and py contains randomly generated values between 0 and 1
        px = np.random.random(num_particles)
        py = np.random.random(num_particles)
        particle_masses = np.random.random(num_particles)

        bottom_left = [0, 0]
        top_right = [1, 1]

        resolution = (512, 512)
        dx = (top_right[0] - bottom_left[0])/resolution[0]
        dy = (top_right[1] - bottom_left[1])/resolution[1]
        square_area = dx * dy
        projection_array = np.zeros(resolution, dtype='float_')
        OnAP.create_projection(px,
                               py,
                               particle_masses,
                               bottom_left,
                               top_right,
                               projection_array
                               )
        projection_array = np.multiply(projection_array, square_area)
        assert np.isclose(np.sum(projection_array), np.sum(particle_masses),
                          rtol=1e-09)

    def test_locations_correct(self):
        """Very basic test to see if particles are in correct locations
           Since resolution and top_right are the same, the area of a pixel is
           always 1. Thus, the density_array is simply the sum of masses
         """
        num_particles = 3000

        # px and py contains randomly generated values between 0 and 1
        rand_x = np.random.randint(low=0, high=2000)
        rand_y = np.random.randint(low=0, high=2000)
        px = np.random.randint(size=num_particles, low=0, high=rand_x)
        py = np.random.randint(size=num_particles, low=0, high=rand_y)
        particle_masses = np.ones(num_particles, dtype='int32')

        bottom_left = [0, 0]
        top_right = [rand_x, rand_y]

        resolution = (rand_x, rand_y)
        density_array = np.zeros(resolution, dtype='float_')
        OnAP.create_projection(px,
                               py,
                               particle_masses,
                               bottom_left,
                               top_right,
                               density_array
                               )
        for x, y in zip(px, py):
            density_array[x, y] -= 1
        assert np.sum(density_array) == 0

    def test_rotation_matrix(self):
        normal_vector = np.array([0, 1, 0])
        test_point = np.array([1, 0, 0])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[1., 0., 0.]]), rtol=1e-09)
        normal_vector = np.array([0, 1, 0])
        test_point = np.array([0, 0, 1])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[0., 1., 0.]]), rtol=1e-09)
        normal_vector = np.array([1, 0, 0])
        test_point = np.array([0, 0, 1])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[1., 0., 0.]]), rtol=1e-09)

    def test_rotations(self):
        normal_vector = np.array([0, 1, 0])
        test_point = np.array([0, 0, 2])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[0., 2., 0.]]), rtol=1e-09)
        normal_vector = np.array([1, 0, 0])
        test_point = np.array([0, 0, 2])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[2., 0., 0.]]), rtol=1e-09)
        normal_vector = np.array([0, 1, 0])
        test_point = np.array([0, 0, 1])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[0., 1., 0.]]), rtol=1e-09)
        normal_vector = np.array([0, 1, 0])
        test_point = np.array([0, 1, 1])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        print(np.matmul(rotation_matrix, test_point))
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[0., 1., -1.]]), rtol=1e-09)
        normal_vector = np.array([1, 0, 0])
        test_point = np.array([1, 0, 1])
        rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
        print(np.matmul(rotation_matrix, test_point))
        assert np.allclose(np.matmul(rotation_matrix, test_point),
                           np.array([[1., 0., -1.]]), rtol=1e-09)

    def test_projection(self):
        num_particles = 3000

        # px and py contains randomly generated values between 0 and 1
        px = np.random.random(num_particles)
        py = np.random.random(num_particles)
        pz = np.random.random(num_particles)
        particle_masses = np.random.random(num_particles)
        bottom_left = [0, 0, 0]
        top_right = [1, 1, 1]
        normal_vector = np.array([0, 0, 1])
        resolution = (64, 64)
        projection_array = np.zeros(resolution)
        OffAP.create_projection(px, py, pz, particle_masses, bottom_left,
                                top_right, projection_array, normal_vector)
        projection_array_1 = np.zeros(resolution)
        OnAP.create_projection(px, py, particle_masses, bottom_left, top_right,
                               projection_array_1)
        assert np.allclose(projection_array, projection_array_1, rtol=1e-09)

    def test_chunking(self):
        num_particles = 3000
        px = np.random.random(num_particles)
        py = np.random.random(num_particles)
        pz = np.random.random(num_particles)
        particle_masses = np.random.random(num_particles)
        bottom_left = [0, 0, 0]
        top_right = [1, 1, 1]
        normal_vector = np.array([3, 5, 6])
        resolution = (512, 512)
        projection_array = np.zeros(resolution)
        projection_array_1 = np.zeros(resolution)
        for i in range(3):
            OffAP.create_projection(px[i * 1000: (i+1) * 1000],
                                    py[i * 1000: (i+1) * 1000],
                                    pz[i * 1000: (i+1) * 1000],
                                    particle_masses[i * 1000: (i+1) * 1000],
                                    bottom_left,
                                    top_right, projection_array, normal_vector)

        OffAP.create_projection(px, py, pz, particle_masses, bottom_left,
                                top_right, projection_array_1, normal_vector)
        assert np.allclose(projection_array, projection_array_1, rtol=1e-05)

    def test_integration(self):
        passed = True
        for i in range(1000):
            num = np.random.random()
            if not np.isclose(SPH.get_dimensionless_2D_kernel(num),
                              SPH.integrate_q2((num) ** 2), rtol=1e-09):
                passed = False
                break
        assert passed
