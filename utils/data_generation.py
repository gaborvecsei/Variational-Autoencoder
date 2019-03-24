import numpy as np
import cv2


def _create_circle_image(image_width: int,
                         image_height: int,
                         circle_x: int,
                         circle_y: int,
                         circle_radius: int,
                         bg_color_rgb: tuple = None,
                         circle_color_rgb: tuple = (70, 127, 255)):
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    if bg_color_rgb is not None:
        b, g, r = bg_color_rgb[::-1]
        image[:, :, 0].fill(b)
        image[:, :, 1].fill(g)
        image[:, :, 2].fill(r)

    image = cv2.circle(image, (circle_x, circle_y), circle_radius, circle_color_rgb, -1)
    return image


def rnd_circle_data_generator(nb_images: int,
                              image_width: int,
                              image_height: int,
                              x_boundary: tuple,
                              y_boundary: tuple,
                              radius_boundary: tuple):
    for i in range(nb_images):
        min_x, max_x = x_boundary
        rnd_x = np.random.randint(min_x, max_x)
        min_y, max_y = y_boundary
        rnd_y = np.random.randint(min_y, max_y)
        min_r, max_r = radius_boundary
        rnd_r = np.random.randint(min_r, max_r)
        image = _create_circle_image(image_width, image_height, rnd_x, rnd_y, rnd_r)
        yield image


def generate_simple_circle_animation(nb_images: int, image_width: int, image_height: int, ball_step_size: int = 2,
                                     x: int = None, y: int = None):
    images = np.zeros((nb_images, image_height, image_width, 3), dtype=np.uint8)

    if x is None:
        x = image_width // 2

    if y is None:
        y = image_height // 2

    goes_left = True

    for image in images:
        if goes_left:
            if x >= image_width:
                goes_left = False
            x += ball_step_size
        else:
            if x <= 0:
                goes_left = True
            x -= ball_step_size

        cv2.circle(image, (x, y), 10, (255, 0, 0), cv2.FILLED)

    return images
