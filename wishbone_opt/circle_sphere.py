import math
import random
import numpy as np

from tqdm import trange

def circ_to_circ(rad_one: float, rad_two: float, center_to_center: float):
    """Returns a list of tuples representing the intersection points between two circles.

    Parameters:
        rad_one (float): The radius of the first circle.
        rad_two (float): The radius of the second circle.
        center_to_center (float): The distance between the centers of the two circles.

    Returns:
        list: the chord and two distances along the c2c line

    Raises:
        ValueError: If the input parameters are invalid or if no intersections exist.

    Examples:
        >>> circ_to_circ(2, 3, 4)
        [(2, 0), (1, 2), (1, -2)]
        >>> circ_to_circ(5, 6, 7)
        []
    """
    if rad_one < 0 or rad_two < 0:
        raise ValueError("The radius cannot be negative.")

    if center_to_center < 0:
        raise ValueError("The distance between the centers cannot be negative.")

    if rad_one + rad_two < center_to_center:
        raise ValueError(f"No intersection exists. Circles are too far apart: {rad_one} + {rad_two} < {center_to_center}")

    # Check for no-intersection due to concentric like behavior
    bigger_rad = max(rad_one, rad_two)
    smaller_rad = min(rad_one, rad_two)

    if smaller_rad + center_to_center < bigger_rad:
        raise ValueError(
            f"No intersection exists. Smaller circle is completely contained: {smaller_rad} + {center_to_center} < {bigger_rad}; {rad_one} {rad_two} {center_to_center}"
        )

    # Solution from: https://mathworld.wolfram.com/Circle-CircleIntersection.html
    a = (
        1
        / center_to_center
        * math.sqrt(
            4 * center_to_center**2 * rad_one**2
            - (center_to_center**2 - rad_two**2 + rad_one**2) ** 2
        )
    )

    leg_one = (center_to_center**2 - rad_two**2 + rad_one**2) / (2 * center_to_center)
    leg_two = (center_to_center**2 + rad_two**2 - rad_one**2) / (2 * center_to_center)
    assert math.isclose(
        leg_one + leg_two, center_to_center
    ), f"{leg_one + leg_two} != {center_to_center}"

    perp = a / 2
    assert math.isclose(
        math.sqrt(perp**2 + leg_one**2), rad_one, abs_tol=1e-12, rel_tol=1e-3
    ), f"{math.sqrt(perp**2 + leg_one**2)} != {rad_one}"
    assert math.isclose(
        math.sqrt(perp**2 + leg_two**2), rad_two, abs_tol=1e-12, rel_tol=1e-3
    ), f"{math.sqrt(perp**2 + leg_two**2)} != {rad_two}"

    return a, leg_one, leg_two


def either_circ_circ():
    len_a = random.random() * 10
    len_b = random.random() * 10

    upper = len_a + len_b
    lower = abs(len_a - len_b)

    c_to_c = random.random() * (upper - lower) + lower

    circ_to_circ(len_a, len_b, c_to_c)


def test_circ_to_circ():
    assert circ_to_circ(1, 1, 2) == (0, 1, 1)
    print("Trivial case passed")

    assert circ_to_circ(5, 5, 8) == (6, 4, 4)
    print("3, 4, 5 case passed")

    for _ in trange(1_000_000):
        either_circ_circ()
    print("Fully Random Intersection passed")


#+++++++++++++++++++++++++++++++
# 3D Circle-Sphere Intersection
#+++++++++++++++++++++++++++++++

def dist_assert(a, b, dist):
    real_dist = np.linalg.norm(a - b)
    assert math.isclose(
        real_dist, dist, rel_tol=1e-4
    ), f"{real_dist} != {dist}"



def circ_to_sphere(sphere_center, sphere_radius: float, circle_center, circle_radius: float, circle_normal):    
    circle_normal /= np.linalg.norm(circle_normal)

    sphere_to_circle = circle_center - sphere_center

    ## Step one: Compute the normal and in plane components
    # Project the c2c onto the circle normal vector
    normal = np.dot(sphere_to_circle, circle_normal) * circle_normal
    # Find the residual, in-plane componenet
    in_plane = sphere_to_circle - normal

    ## Step two: use the normal distance + pythag to compute the in-plane radius of the sphere
    # Basically we are taking the slice of the sphere that is in the same plane as the circle
    in_plane_rad = math.sqrt(sphere_radius**2 - np.dot(normal, normal))

    ## Step three: solve the circle to circle
    chord, dist_one, _= circ_to_circ(in_plane_rad, circle_radius, np.linalg.norm(in_plane))

    ## Step four: convert the circle to circle intersection points back into 3D

    two_d_c2c_comp = in_plane / np.linalg.norm(in_plane) * dist_one

    # Negating this vector gives you the 2nd solution
    double_perpendicular = np.cross(circle_normal, in_plane)
    double_perpendicular /= np.linalg.norm(double_perpendicular)
    double_perpendicular *= chord / 2
 
    positive_int = sphere_center + two_d_c2c_comp + normal + double_perpendicular
    negative_int = sphere_center + two_d_c2c_comp + normal - double_perpendicular

    # Check that these points are actually on the sphere
    dist_assert(sphere_center, positive_int, sphere_radius)
    dist_assert(sphere_center, negative_int, sphere_radius)

    # Check that the point is the right distance away from the circle center
    dist_assert(circle_center, negative_int, circle_radius)
    dist_assert(circle_center, positive_int, circle_radius)

    # Check that the point is in plane with the circle
    pos_normal_comp = np.dot(positive_int - circle_center, circle_normal)
    neg_normal_comp = np.dot(negative_int - circle_center, circle_normal)

    assert math.isclose(
        pos_normal_comp, 0.0, abs_tol=1e-14, rel_tol=1e-4
    ), f"{pos_normal_comp} != {0.0}"
    assert math.isclose(
        neg_normal_comp, 0.0, abs_tol=1e-14, rel_tol=1e-4
    ), f"{neg_normal_comp} != {0.0}"

    # Return both the positive and negative intersection points
    return positive_int, negative_int

# def either_sphere_circle():
    # len_a = random.random() * 10
    # len_b = random.random() * 10
# 
    # upper = len_a + len_b
    # lower = abs(len_a - len_b)
# 
    # c_to_c = random.random() * (upper - lower) + lower
# 
    # circ_to_circ(len_a, len_b, c_to_c)

def test_sphere_to_circle():
    for _ in trange(1_000_000):
        
        sphere_center = np.random.uniform(0.0, 10.0, 3)
        circle_normal = np.array([0.0, 1.0, 0.0])

        circle_offset = np.random.uniform(0.0, 10.0)
        circle_c2c = np.random.uniform(0.1, 10.0)
        circle_center = sphere_center + np.array([0.0, circle_offset, circle_c2c])
    
        circle_radius = np.random.uniform(1e-5, 2 * circle_c2c)
    
        nearest_point = np.array((0, circle_offset, circle_c2c - circle_radius))
        farthest_point = np.array((0, circle_offset, circle_c2c + circle_radius))
        min_radius, max_radius = np.linalg.norm(nearest_point), np.linalg.norm(farthest_point)
        normal_point = sphere_center + np.array((0, circle_offset, 0))
        
        if np.linalg.norm(normal_point - circle_center) < circle_radius:
            # We need to bump the normal point up enough that it sits on the circle
            square_deficit = circle_radius ** 2 - np.linalg.norm(normal_point - circle_center) ** 2
            min_radius = np.linalg.norm(np.array((square_deficit**0.5, circle_offset, 0)))
            
        sphere_radius = np.random.uniform(min_radius, max_radius)

    
        circ_to_sphere(sphere_center, sphere_radius, circle_center, circle_radius, circle_normal)

    print("Random Sphere Intersection passed")
    
    


def test():
    test_circ_to_circ()
    test_sphere_to_circle()


if __name__ == "__main__":
    test()
