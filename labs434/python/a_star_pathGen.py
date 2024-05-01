import heapq
import numpy as np

class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost from start to node
        self.h = 0  # Heuristic based estimated cost from node to end
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        (x, y) = current_node.position
        # Adjacent squares (consider 8-directional movement, or reduce to 4 for orthogonal movement only)
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]

        for next in neighbors:
            # Maze boundaries
            if next[0] > (len(maze) - 1) or next[0] < 0 or next[1] > (len(maze[len(maze)-1]) - 1) or next[1] < 0:
                continue
            # Obstacle check
            if maze[next[0]][next[1]] != 0:
                continue

            # Create new node
            neighbor = Node(current_node, next)

            if neighbor in closed_list:
                continue

            # Create the f, g, and h values
            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - end_node.position[0]) ** 2) + ((neighbor.position[1] - end_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            # Child is already in the open list
            if add_to_open(open_list, neighbor):
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

def create_maze(blocked_space, grid_dimensions, grid_size, car_position):
    """
    Creates a grid-based maze representation from blocked space coordinates.
    The grid origin is set to the car's current position.
    
    :param blocked_space: numpy array of blocked coordinates.
    :param grid_dimensions: tuple (height, width) of the maze.
    :param grid_size: the size of each grid cell.
    :param car_position: current position of the car as (x, y).
    :return: 2D numpy array representing the maze.
    """
    height, width = grid_dimensions
    maze = np.zeros((height, width), dtype=int)
    
    # Calculate the bottom-left corner of the grid based on car_position
    origin_x, origin_y = car_position[0] - width // 2 * grid_size, car_position[1] - height // 2 * grid_size
    
    # Convert blocked space coordinates to grid indices
    for coord in blocked_space:
        x, y = coord
        grid_x = int((x - origin_x) / grid_size)
        grid_y = int((y - origin_y) / grid_size)
        if 0 <= grid_x < width and 0 <= grid_y < height:
            maze[grid_y, grid_x] = 1  # Mark as obstacle

    return maze, (origin_x, origin_y)

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
            display_maze[y, x] = 'p'
    
    # Mark the start and goal points
    start_x, start_y = start
    end_x, end_y = end
    display_maze[start_y, start_x] = 's'
    display_maze[end_y, end_x] = 'g'

    # Print the maze
    for row in display_maze:
        print(' '.join(row))



# Define the wall array to form a closed room around the origin
wall = np.array([
    [-5, -5],
    [-5, -4],
    [-5, -3],
    [-5, -2],
    [-5, -1],
    [-5, 0],
    [-5, 1],
    [-5, 2],
    [-5, 3],
    [-5, 4],
    [-5, 5],
    [-4, 5],
    [-3, 5],
    [-2, 5],
    [-1, 5],
    [0, 5],
    [1, 5],
    [2, 5],
    [3, 5],
    [4, 5],
    [5, 5],
    [5, 4],
    [5, 3],
    [5, 2],
    [5, 1],
    [5, 0],
    [5, -1],
    [5, -2],
    [5, -3],
    [5, -4],
    [5, -5],
    [4, -5],
    [3, -5],
    [2, -5],
    [1, -5],
    [0, -5],
    [-1, -5],
    [-2, -5],
    [-3, -5],
    [-4, -5]
])

# Define obstacles as cylinders with position, radius, and height
obstacles = np.array([[1, 3],
                      [-2, 3],
                      [3,3],
                      [3, -3]])
# [
#     {"pos": [1, 3], "radius": 0.5, "height": 1.0},
#     {"pos": [-2, 3], "radius": 0.4, "height": 0.8},
#     {"pos": [3, 3], "radius": 0.3, "height": 0.6}
# ]
obstacles_with_wall = np.concatenate((obstacles, wall))

car_position = (0, 0)  # Current car position
goal_position = (4, 4)  # Goal position relative to the world
grid_size = 1  # Each grid cell represents 1 unit
grid_dimensions = (20, 20)  # 11x11 grid

maze, grid_origin = create_maze(obstacles_with_wall, grid_dimensions, grid_size, car_position)


# Convert positions to grid coordinates
start_grid = local_to_grid(car_position, grid_size, grid_origin)  # Should be roughly at the center of the grid
end_grid = local_to_grid(goal_position, grid_size, grid_origin)

# Perform A* search
path_grid = astar(maze, start_grid, end_grid)
if path_grid:
    path_local = [grid_to_local(pos, grid_size, grid_origin) for pos in path_grid]
    print("Path in local coordinates:", path_local)
else:
    print("No path found.")

print_maze_with_path(maze, path_grid, start_grid, end_grid)
