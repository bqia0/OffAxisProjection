import numpy as np
import OffAxisProjection.pixelDensities as pd


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

        resolution = (512,512)
        dx = (top_right[0] - bottom_left[0])/resolution[0]
        dy = (top_right[1] - bottom_left[1])/resolution[1]
        square_area = dx * dy
        density_array = pd.pixel_densities(px,
                                           py,
                                           particle_masses,
                                           bottom_left,
                                           top_right,
                                           resolution,
                                           num_particles)
        density_array *= square_area
        assert np.isclose(np.sum(density_array), np.sum(particle_masses),
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
        density_array = pd.pixel_densities(px,
                                           py,
                                           particle_masses,
                                           bottom_left,
                                           top_right,
                                           resolution,
                                           num_particles)
        for x, y in zip(px, py):
            density_array[x, y] -= 1
        assert np.sum(density_array) == 0
