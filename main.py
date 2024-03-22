import copy
import random
import numpy as np
from math import inf, pi, e, sin, cos

num_particles = 300


class Particle:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w


def normal_distribution(error, sigma):
    return (1.0 / (((2.0 * pi) ** (1 / 2)) * sigma)) * (e ** (-0.5 * ((error ** 2) / sigma ** 2)))


def predict_next_p(p, a, rd):
    return Particle(p.x + rd * cos(a), p.y + rd * sin(a), p.w)


def calc_dist(p, w):
    t = -1
    w_dx, w_dy = w[1][0] - w[0][0], w[1][1] - w[0][1]

    if (w_dx**2 + w_dy**2) != 0:
        t = ((p.x - w[0][0]) * w_dx + (p.y - w[0][1]) * w_dy) / (w_dx**2 + w_dy**2)
    if t < 0:
        closest = w[0]
    elif t > 1:
        closest = w[1]
    else:
        closest = (w[0][0] + t * w_dx, w[0][1] + t * w_dy)
    dx, dy = p.x - closest[0], p.y - closest[1]
    return (dx ** 2 + dy ** 2) ** (1 / 2)


def move(p, robot, s):
    # print("Move")
    cur_s = (s[0] ** 0.5, s[1] ** 0.5)
    for item in p:
        item.x = item.x + robot[0] + random.normalvariate(-cur_s[0], cur_s[0])
        item.y = item.y + robot[1] + random.normalvariate(-cur_s[1], cur_s[1])
    return p


def sense(p, r_d, w, d, s):
    new_p = copy.deepcopy(p)
    # print("Sense before:", p, "\n", r_d, "\n", w)
    rec = False
    w_sum = 0
    while not rec:
        w_sum = 0
        for item in new_p:
            delta = 0
            for a, rd in zip(d, r_d):
                next_p = predict_next_p(item, a, rd)
                error = min(calc_dist(next_p, w[i]) for i in range(len(w)))
                delta += error ** 2
            avr_res = (delta / len(d)) ** (1 / 2)
            item.w = normal_distribution(avr_res, s)
            w_sum += item.w

        if w_sum > 0:
            rec = True
        else:
            s *= 1.01

    for item in new_p:
        item.w = item.w / w_sum
    # print("Sense after:", p, "\n", r_d, "\n", w)
    return new_p


def resample(p):
    # print("Resample", len(p), num_particles)
    w = [item.w for item in p]
    cumulative_sum = np.cumsum(w)
    indexes = np.searchsorted(cumulative_sum, np.random.random(num_particles))
    new_p = [Particle(particles[idx].x, particles[idx].y, particles[idx].w) for idx in indexes]
    return new_p


def random_in_range(f_min, f_max):
    return random.uniform(f_min, f_max)


if __name__ == '__main__':
    n = int(input())
    numbers = list(map(int, input().split()))
    v = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    # print("a:", a)
    min_x, min_y, max_x, max_y = inf, inf, -inf, -inf
    for x_i, y_i in v:
        min_x = min(min_x, x_i)
        max_x = max(max_x, x_i)
        min_y = min(min_y, y_i)
        max_y = max(max_y, y_i)

    walls = [((v[i][0], v[i][1]), (v[i + 1][0], v[i + 1][1])) for i in range(n - 1)]
    walls.append(((v[n - 1][0], v[n - 1][1]), (v[0][0], v[0][1])))
    # print('Walls:', walls, walls[0][0][0])

    m, k = map(int, input().split())
    sl, sx, sy = map(float, input().split())
    bool_dist = list(map(float, input().split()))

    particles = []
    if int(bool_dist[0]) == 1:
        mu = bool_dist[1], bool_dist[2]
        for i in range(num_particles):
            particles.append(Particle(bool_dist[1], bool_dist[2], 1.0 / num_particles))
    else:
        mu = None
        for i in range(num_particles):
            particle = Particle(random_in_range(min_x, max_x), random_in_range(min_y, max_y), 1.0 / num_particles)
            particles.append(particle)
            # if not is_inside_polygon(particle, walls):
            #     i -= 1
            # else:
            #     particles.append(particle)

    d_step = (2 * pi) / k
    degrees = [i for i in np.arange(0, (2 * pi), d_step)]
    if d_step * k == 2 * pi:
        degrees.append(2 * pi)
    # print("Degrees:", degrees)

    k_gen, x_gen = [], []
    for i in range(m):
        k_m = list(map(float, input().split()))
        k_gen.append(k_m)
        xi, yi = map(float, input().split())
        x_gen.append((xi, yi))
    k_m = list(map(float, input().split()))
    k_gen.append(k_m)

    particles = sense(particles, k_gen[0], walls, degrees, sl)
    for i in range(m):
        particles = move(particles, x_gen[i], (sx, sy))
        particles = sense(particles, k_gen[i + 1], walls, degrees, sl)
        particles = resample(particles)

    w_gen = sum(i.w for i in particles)
    for i in particles:
        i.w /= w_gen
    res_x = sum(i.x * i.w for i in particles)
    res_y = sum(i.y * i.w for i in particles)
    print(res_x, res_y)
