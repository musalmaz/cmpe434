import heapq
import numpy as np
import time
from math import sqrt



class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # Using Euclidean distance for heuristic
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(maze, start, end, timeout=60):
    start_time = time.time()
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        if time.time() - start_time > timeout:
            print("Timed out!")
            return None

        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # Generate neighbors for 8-directional movement
        (x, y) = current_node.position
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]

        for next in neighbors:
            if not (0 <= next[0] < len(maze) and 0 <= next[1] < len(maze[0])):
                continue
            if maze[next[0]][next[1]] != 0:
                continue

            # Check for diagonal movement blocking through corners
            if abs(next[0] - x) == 1 and abs(next[1] - y) == 1:
                if maze[x][next[1]] != 0 and maze[next[0]][y] != 0:
                    continue

            neighbor = Node(current_node, next)
            if neighbor.position in closed_set:
                continue

            neighbor.g = current_node.g + (sqrt(2) if abs(next[0] - x) + abs(next[1] - y) == 2 else 1)
            neighbor.h = heuristic(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h

            if not add_to_open(open_list, neighbor):
                continue

            heapq.heappush(open_list, neighbor)

    return None

def add_to_open(open_list, neighbor):
    for node in open_list:
        if neighbor == node and neighbor.g > node.g:
            return False
    return True

def grid_to_local(grid_pos, grid_size, grid_origin):
    """
    Converts grid position to local simulation coordinates.
    
    :param grid_pos: tuple of grid coordinates (x, y).
    :param grid_size: size of each grid cell.
    :param grid_origin: the origin (0,0) coordinate in local space.
    :return: tuple of local simulation coordinates.
    """
    local_x = grid_pos[0] * grid_size + grid_origin[0]
    local_y = grid_pos[1] * grid_size + grid_origin[1]
    return (local_x, local_y)

def local_to_grid(local_pos, grid_size, grid_origin):
    """
    Converts local simulation coordinates to grid position.
    
    :param local_pos: tuple of local coordinates (x, y).
    :param grid_size: size of each grid cell.
    :param grid_origin: the origin (0,0) coordinate in local space.
    :return: tuple of grid coordinates.
    """
    grid_x = int((local_pos[0] - grid_origin[0]) / grid_size)
    grid_y = int((local_pos[1] - grid_origin[1]) / grid_size)
    return (grid_x, grid_y)

def create_dynamic_maze(blocked_space, goal_pos, grid_size, car_position):
    """
    Creates a grid-based maze representation with dynamic dimensions to ensure that start and goal are included.
    
    :param blocked_space: numpy array of blocked coordinates.
    :param start_pos: tuple (x, y) for start position.
    :param goal_pos: tuple (x, y) for goal position.
    :param grid_size: the size of each grid cell.
    :param car_position: current position of the car as (x, y).
    :return: 2D numpy array representing the maze and its origin.
    """
    # Include all relevant points in the calculations
    all_points = np.vstack([blocked_space, np.array([goal_pos]), np.array([car_position])])

    # Find the minimum and maximum coordinates
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    # Calculate grid dimensions
    width = int((max_x - min_x) / grid_size) + 1
    height = int((max_y - min_y) / grid_size) + 1
    print("aStar minx : miny : maxx, maxy : ", min_x, min_y, max_x, max_y)
    print("Maze size : ", width, height)
    if width > 50:
        width = 50
    if height > 50:
        height = 50
    # Calculate the origin of the grid
    origin_x = min_x
    origin_y = min_y

    # Initialize the maze
    maze = np.zeros((height, width), dtype=int)

    # Mark blocked spaces on the grid
    for x, y in blocked_space:
        grid_x = int((x - origin_x) / grid_size)
        grid_y = int((y - origin_y) / grid_size)
        if 0 <= grid_x < width and 0 <= grid_y < height:
            maze[grid_y, grid_x] = 1  # Mark as obstacle

    return maze, (origin_x, origin_y)

def create_localized_maze(blocked_space, start_pos, goal_pos, grid_size):
    """
    Create a grid-based maze limited to a rectangle defined by start and goal positions, including nearby obstacles.
    
    :param blocked_space: numpy array of blocked coordinates.
    :param start_pos: tuple (x, y) for start position.
    :param goal_pos: tuple (x, y) for goal position.
    :param grid_size: the size of each grid cell.
    :return: 2D numpy array representing the maze, its origin, and the adjusted start and goal positions in the local grid.
    """
    # Define the bounds of the rectangle
    min_x = min(start_pos[0], goal_pos[0])
    max_x = max(start_pos[0], goal_pos[0])
    min_y = min(start_pos[1], goal_pos[1])
    max_y = max(start_pos[1], goal_pos[1])

    # Expand the rectangle slightly to ensure room for pathfinding
    min_x -= grid_size
    max_x += grid_size
    min_y -= grid_size
    max_y += grid_size

    # Initialize the maze within the defined bounds
    width = int((max_x - min_x) / grid_size) + 1
    height = int((max_y - min_y) / grid_size) + 1
    maze = np.zeros((height, width), dtype=int)

    # Adjust origin based on the minimum values
    origin = (min_x, min_y)

    # Place obstacles in the maze
    for x, y in blocked_space:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            grid_x = int((x - min_x) / grid_size)
            grid_y = int((y - min_y) / grid_size)
            maze[grid_y, grid_x] = 1

    # Convert start and goal positions to local grid coordinates
    start_grid_pos = ((start_pos[0] - min_x) // grid_size, (start_pos[1] - min_y) // grid_size)
    goal_grid_pos = ((goal_pos[0] - min_x) // grid_size, (goal_pos[1] - min_y) // grid_size)

    return maze, origin, start_grid_pos, goal_grid_pos



def print_maze_with_path(maze, path, start, end):
    """
    Prints the maze with the path from start to goal marked.
    
    :param maze: 2D numpy array representing the maze.
    :param path: list of tuples representing the path from start to goal.
    :param start: tuple (x, y) representing the start coordinate in the maze.
    :param end: tuple (x, y) representing the end coordinate in the maze.
    """
    # Create a copy of the maze to modify it without changing the original
    display_maze = np.array(maze, dtype=str)
    display_maze[display_maze == '0'] = ' '
    display_maze[display_maze == '1'] = '█'  # Using a block character to represent walls

    # Mark the path on the maze
    if path:
        for (x, y) in path:
            display_maze[x, y] = 'p'
    
    # Mark the start and goal points
    start_x, start_y = start
    end_x, end_y = end
    display_maze[start_x, start_y] = 's'
    display_maze[end_x, end_y] = 'g'

    # Print the maze
    for row in display_maze:
        print(' '.join(row))



# # Define the wall array to form a closed room around the origin
# wall = np.array([
#     [-5, -5],
#     [-5, -4],
#     [-5, -3],
#     [-5, -2],
#     [-5, -1],
#     [-5, 0],
#     [-5, 1],
#     [-5, 2],
#     [-5, 3],
#     [-5, 4],
#     [-5, 5],
#     [-4, 5],
#     [-3, 5],
#     [-2, 5],
#     [-1, 5],
#     [0, 5],
#     [1, 5],
#     [2, 5],
#     [3, 5],
#     [4, 5],
#     [5, 5],
#     [5, 4],
#     [5, 3],
#     [5, 2],
#     [5, 1],
#     [5, 0],
#     [5, -1],
#     [5, -2],
#     [5, -3],
#     [5, -4],
#     [5, -5],
#     [4, -5],
#     [3, -5],
#     [2, -5],
#     [1, -5],
#     [0, -5],
#     [-1, -5],
#     [-2, -5],
#     [-3, -5],
#     [-4, -5]
# ])

# # Define obstacles as cylinders with position, radius, and height
# obstacles = np.array([[1, 3],
#                       [-2, 3],
#                       [3,3],
#                       [3, -3]])
# # [
# #     {"pos": [1, 3], "radius": 0.5, "height": 1.0},
# #     {"pos": [-2, 3], "radius": 0.4, "height": 0.8},
# #     {"pos": [3, 3], "radius": 0.3, "height": 0.6}
# # ]
# obstacles_with_wall = np.concatenate((obstacles, wall))

# car_position = (0, 0)  # Current car position
# goal_position = (4, 4)  # Goal position relative to the world
# grid_size = 1  # Each grid cell represents 1 unit
# grid_dimensions = (20, 20)  # 11x11 grid

# maze, grid_origin = create_maze(obstacles_with_wall, grid_dimensions, grid_size, car_position)


# # Convert positions to grid coordinates
# start_grid = local_to_grid(car_position, grid_size, grid_origin)  # Should be roughly at the center of the grid
# end_grid = local_to_grid(goal_position, grid_size, grid_origin)

# # Perform A* search
# path_grid = astar(maze, start_grid, end_grid)
# if path_grid:
#     path_local = [grid_to_local(pos, grid_size, grid_origin) for pos in path_grid]
#     print("Path in local coordinates:", path_local)
# else:
#     print("No path found.")

# print_maze_with_path(maze, path_grid, start_grid, end_grid)
