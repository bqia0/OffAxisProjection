import numpy as np
import numpy.linalg as linalg
import yt
from yt.utilities.lib import pixelization_routines
import math

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
        h = max(smoothing_lengths[i], half_pixel_width)
        weight = get_dimensionless_weight(particle_masses[i],
                                          particle_densities[i], h)
        # find boundries in pixels
        x0 = int(max(px[i] - h - bottom_left[0], bottom_left[0]) / dx)
        x1 = int(min(px[i] + h - bottom_left[0], top_right[0]) / dx)
        y0 = int(max(py[i] - h - bottom_left[1], bottom_left[1]) / dy)
        y1 = int(min(py[i] + h - bottom_left[0], top_right[1]) / dy)
        for xi in range(x0, x1):
            x = xi + 0.5  # compare from center of pixel
            for yi in range(y0, y1):
                y = yi + 0.5
                x_diff = px[i] - x
                y_diff = py[i] - y
                squared_distance = x_diff**2 + y_diff**2
                if squared_distance > h**2:
                    continue
                projection_array[xi, yi] += weight \
                    * h \
                    * get_smoothing_kernel(x_diff, y_diff, h)


def get_dimensionless_weight(mass, density, smoothing_length):
    return mass/(density * smoothing_length ** 3)  # where do i get density?


def get_smoothing_kernel(x, y, smoothing_length):
    """ x = x - xj, y = y - yj"""
    r_xy = math.sqrt(x ** 2 + y ** 2)
    q_xy = r_xy / smoothing_length
    return get_dimensionless_2D_kernel(q_xy)


def get_dimensionless_2D_kernel(q_xy):
    # q_z = z / smoothing_length  # is the z the z coordinate?
    # q = math.sqrt(q_xy ** 2 + q_z ** 2)
    integration_upper_bound = math.sqrt(1 - q_xy ** 2)  # 1^2 for the cubic spline
    integration_lower_bound = -1 * integration_upper_bound
    intervals = 300
    dz = (integration_upper_bound - integration_lower_bound) / intervals
    x = np.zeros(intervals, dtype='float_')
    y = np.zeros(intervals, dtype='float_')
    for i in range(intervals):
        x[i] = i * dz + integration_lower_bound
        y[i] = sph_kernel_cubic(math.sqrt((i * dz + integration_lower_bound)**2 + q_xy**2))
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