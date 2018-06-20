import numpy as np
import numpy.linalg as linalg
import yt
from yt.utilities.lib import pixelization_routines as pr
import OffAxisProjection.OffAxisProjection as OffAP
import math

def off_axis_projection_SPH(px, py, pz, particle_masses, particle_densities,
                            smoothing_lengths, bounds,
                            projection_array, normal_vector):
    if np.allclose(normal_vector, np.array([0., 0., 0.]), rtol=1e-09):
        return

    resolution = np.shape(projection_array)
    num_particles = min(np.size(px), np.size(py), np.size(pz),
                        np.size(particle_masses))
    rotation_matrix = OffAP.get_rotation_matrix(normal_vector)
    px_rotated = np.zeros(num_particles)
    py_rotated = np.zeros(num_particles)
    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        z_coordinate = pz[i]
        if x_coordinate > bounds[1] or y_coordinate > bounds[3]:
            continue
        if x_coordinate < bounds[0] or y_coordinate < bounds[2]:
            continue
        if z_coordinate < bounds[4] or z_coordinate > bounds[5]:
            continue
        coordinate_matrix = np.array([x_coordinate, y_coordinate, z_coordinate], dtype='float_')
        new_coordinates = rotation_matrix @ coordinate_matrix

        # image_pixel_coordinates = pixel_coordinates(new_coordinates[0],
        #                                             new_coordinates[1],
        #                                             top_right, bottom_left,
        #                                             resolution)
        if new_coordinates[0] < bounds[0] or new_coordinates[0] >= bounds[1]:
            continue
        if new_coordinates[1] < bounds[2] or new_coordinates[1] >= bounds[3]:
            continue
        px_rotated[i] = new_coordinates[0]
        py_rotated[i] = new_coordinates[1]
    pr.pixelize_sph_kernel_projection(
        projection_array,
        px_rotated,
        py_rotated,
        smoothing_lengths,
        particle_masses,
        particle_densities,
        np.ones(num_particles),
        bounds[:4])


def create_pixel_array(px, py, particle_masses, particle_densities,
                       smoothing_lengths, bottom_left, top_right,
                       projection_array):
    num_particles = min(np.size(px), np.size(py),
                        np.size(particle_masses), 
                        np.size(particle_densities))
    resolution = np.shape(projection_array)
    dx = (top_right[0] - bottom_left[0])/resolution[0]
    dy = (top_right[1] - bottom_left[1])/resolution[1]
    half_pixel_width = dx / 2
    for i in range(num_particles):
        # Bounds of circle formed by particle smoothing length
        print('particle # ' + str(i))
        h = max(smoothing_lengths[i], half_pixel_width)
        h_2 = h ** 2
        weight = get_dimensionless_weight(particle_masses[i],
                                          particle_densities[i], h)
        # find boundries in pixels
        x0 = int(max(px[i] - h - bottom_left[0], bottom_left[0]) / dx)
        x1 = int(min(px[i] + h - bottom_left[0], top_right[0]) / dx)
        y0 = int(max(py[i] - h - bottom_left[1], bottom_left[1]) / dy)
        y1 = int(min(py[i] + h - bottom_left[0], top_right[1]) / dy)
        #print('x: '+ str(x0)+' ' +str(x1))
        #print('y: '+str(y0) + ' ' + str(y1))
        for xi in range(x0, x1):
            x = xi + 0.5  # compare from center of pixel
            for yi in range(y0, y1):
                y = yi + 0.5
                x_diff = px[i] - x * dx
                y_diff = py[i] - y * dy
                x_diff_2 = x_diff ** 2
                y_diff_2 = y_diff ** 2
                squared_distance = x_diff_2 + y_diff_2
                if squared_distance > h_2:
                    continue
                q = (x_diff_2 + y_diff_2)/h_2
                if q > 1:
                    continue
                #print(str(xi) + ' ' + str(yi))
                projection_array[xi, yi] += weight \
                    * h \
                    * get_smoothing_kernel(x_diff_2, y_diff_2, h_2)
                #print(str(xi) + ' ' + str(yi))


def get_dimensionless_weight(mass, density, smoothing_length):
    return mass/(density * smoothing_length ** 3)  # where do i get density?


def get_smoothing_kernel(x, y, smoothing_length):
    """ x = x - xj, y = y - yj"""
    r_xy_2 = x + y
    q_xy_2 = r_xy_2 / smoothing_length
    return get_dimensionless_2D_kernel(q_xy_2)


def get_dimensionless_2D_kernel(q_xy_2):
    # q_z = z / smoothing_length  # is the z the z coordinate?
    # q = math.sqrt(q_xy ** 2 + q_z ** 2)
    integration_upper_bound = math.sqrt(1 - q_xy_2)  # 1^2 for the cubic spline
    integration_lower_bound = -1 * integration_upper_bound
    intervals = 200
    dz = (integration_upper_bound - integration_lower_bound) / intervals
    x = np.zeros(intervals, dtype='float_')
    y = np.zeros(intervals, dtype='float_')
    for i in range(intervals):
        x[i] = i * dz + integration_lower_bound
        y[i] = sph_kernel_cubic(math.sqrt((i * dz + integration_lower_bound)**2 + q_xy_2))
    return np.trapz(y, x)


def sph_kernel_cubic(x):
    C = 2.546479089470325  # 8/pi
    if x <= 0.5:
        kernel = 1.-6.*x*x*(1.-x)
    elif x > 0.5 and x <= 1.0:
        kernel = 2.*(1.-x)**3
    else:
        kernel = 0.
    return kernel * C
