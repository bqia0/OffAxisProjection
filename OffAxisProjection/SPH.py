import numpy as np
import numpy.linalg as linalg
import yt
from yt.utilities.lib import pixelization_routines
import OffAxisProjection.OffAxisProjection as OffAP
import math

def off_axis_projection_SPH(px, py, particle_masses, particle_densities,
                            smoothing_lengths, bottom_left, top_right, 
                            projection_array, normal_vector):
    if np.allclose(normal_vector, np.array([0., 0., 0.]), rtol=1e-09):
        return

    resolution = np.shape(projection_array)
    num_particles = min(np.size(px), np.size(py), np.size(pz),
                        np.size(particle_masses))
    density_array = np.zeros(resolution, dtype='float_')
    dx = (top_right[0] - bottom_left[0])/resolution[0]
    dy = (top_right[1] - bottom_left[1])/resolution[1]
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


def integrate_q2(q2):
        # See equation 30 of the SPLASH paper
        # Our bounds are -sqrt(R*R - q2) and sqrt(R*R-q2)
        # And our R is always 1; note that our smoothing kernel functions
        # expect it to run from 0 .. 1, so we multiply the integrand by 2
        N = 200
        R = 1
        R0 = -math.sqrt(R*R-q2)
        R1 = math.sqrt(R*R-q2)
        dR = (R1-R0)/N
        # Set to our bounds
        integral = 0.0
        integral += sph_kernel_cubic(math.sqrt(R0*R0 + q2))
        integral += sph_kernel_cubic(math.sqrt(R1*R1 + q2))
        # We're going to manually conduct a trapezoidal integration
        for i in range(1, N):
            qz = R0 + i * dR
            integral += 2.0*sph_kernel_cubic(math.sqrt(qz*qz + q2))
        integral *= (R1-R0)/(2*N)
        return integral