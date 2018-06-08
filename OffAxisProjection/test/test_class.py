import numpy as np
import pytest
import OffAxisProjection.pixelDensities
class TestClass(object):
    def test_density_array_correct_sum(self):
        """Use this test to see if the density array has the
           correct elements BEFORE dividing by area
        """
        num_particles = 3000

        # px and py contains randomly generated values between 0 and 1
        px = np.random.random(num_particles)
        py = np.random.random(num_particles)
        particle_masses = np.random.randint(low=0, high=5, size=num_particles)

        bottom_left = [0, 0]
        top_right = [1, 1]

        resolution = (512,512)
        assert np.sum(OffAxisProjection.pixelDensities.pixel_densities(px, py, particle_masses, bottom_left, top_right, resolution, num_particles)) == np.sum(particle_masses)
        #assert 2+2 == 4