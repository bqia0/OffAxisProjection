import numpy as np
import OffAxisProjection
import OffAxisProjection.OffAxisProjection as OffAP
import OffAxisProjection.OnAxisProjection as OnAP
import pysplat as ps
from yt.utilities.lib import pixelization_routines as pr
import matplotlib.pyplot as plt
import OffAxisProjection.SPH as SPH

def make_SPH_projections():
    num_particles = 500
    px = np.random.random(num_particles)
    py = np.random.random(num_particles)
    particle_masses = np.random.random(num_particles)
    particle_densities = np.random.random(num_particles)
    smoothing_length = np.random.random(num_particles)
    smoothing_quantity = np.ones(num_particles)
    bounds = [0, 1, 0, 1]
    bottom_left = [0, 0]
    top_right = [1, 1]
    resolution = (512, 512)
    buf = np.zeros(resolution)
    SPH.create_pixel_array(
        px,
        py,
        particle_masses,
        particle_densities,
        smoothing_length,
        bottom_left,
        top_right,
        buf)
    print(buf)
    plt.imsave('../OffAxisProjectionImages/SPH.png', buf)


def make_projections():
    num_particles = 96000

    # px and py contains randomly generated values between 0 and 1
    px = 2*np.random.random(num_particles) - 1
    py = 2*np.random.random(num_particles) - 1
    pz = .5 *np.random.random(num_particles) - 0.25
    particle_masses = np.ones(num_particles)
    bottom_left = [-1, -1, -1]
    top_right = [1, 1, 1]
    normal_vector = np.array([-2., 2., -5])
    resolution = (512, 512)
    cmap = plt.get_cmap('Reds')
    for i in range(160):
        projection_array = np.zeros(resolution)
        normal_vector[2] += 0.0625
        OffAP.create_projection(px, py, pz, particle_masses, bottom_left, top_right,
                                projection_array, normal_vector)
        plt.imsave('../OffAxisProjectionImages/img_' + str(i) + '.png',
                   projection_array, cmap = cmap)
        print('img_' + str(i) + '.png')

def test_integration():
    num = .5
    print(SPH.get_dimensionless_2D_kernel(num))
    print(SPH.integrate_q2(num **2))


def main():
    make_SPH_projections()

if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
    main()
