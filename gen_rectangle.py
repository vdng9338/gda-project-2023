import numpy as np
import json
import sys

def print_rectangle(bounds, steps):
    bottom_y, right_x, top_y, left_x = bounds
    bottom_steps, right_steps, top_steps, left_steps = steps
    bottom_x = np.linspace(left_x, right_x, bottom_steps, endpoint=False)
    right_y = np.linspace(bottom_y, top_y, right_steps, endpoint=False)
    top_x = np.linspace(right_x, left_x, top_steps, endpoint=False)
    left_y = np.linspace(top_y, bottom_y, left_steps, endpoint=False)
    points = []
    for x in bottom_x:
        points.append([x, bottom_y])
    for y in right_y:
        points.append([right_x, y])
    for x in top_x:
        points.append([x, top_y])
    for y in left_y:
        points.append([left_x, y])
    print(json.dumps(points))

if __name__ == '__main__':
    try:
        bounds = tuple(map(float, sys.argv[1:5]))
        steps = tuple(map(int, sys.argv[5:]))
        assert len(bounds) == 4
        assert len(steps) == 4
    except (ValueError, IndexError, AssertionError):
        print(f'Usage: python3 {sys.argv[0]} <bottom> <right> <top> <left> <bottom steps> <right steps> <top steps> <left steps>')
        sys.exit(1)
    print_rectangle(bounds, steps)
