from collections import deque

def create_grid(start, obstacles, resolution):
    # Find the bounds for the grid
    all_points = obstacles + [start]
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)

    # Adjust bounds to include padding around the perimeter
    min_x -= resolution
    max_x += resolution
    min_y -= resolution
    max_y += resolution

    # Translate start to near center of the grid dynamically
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    grid_start = (start[0] - min_x, start[1] - min_y)

    # Create the grid
    grid = [[False for _ in range(grid_height)] for _ in range(grid_width)]
    for x, y in obstacles:
        if min_x <= x < max_x and min_y <= y < max_y:
            grid[x - min_x][y - min_y] = True

    return grid, grid_start, (grid_width, grid_height)

def bfs_path_finding(start, goal, obstacles, resolution):
    grid, start, grid_size = create_grid(start, obstacles, resolution)
    
    # All 8 possible movements (N, S, E, W, NE, NW, SE, SW)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    queue = deque([(start[0], start[1], [])])  # queue of (x, y, path_so_far)
    visited = set()
    visited.add((start[0], start[1]))

    while queue:
        x, y, path = queue.popleft()
        path.append((x, y))

        if (x, y) == goal:
            return path  # Return path if goal is reached

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and not grid[nx][ny]:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, path.copy()))

    return None  # If no path found

# Example usage
obstacles = [(5, 5), (5, 6), (5, 7)]
start = (0, 0)
goal = (8, 8)
resolution = 1

path = bfs_path_finding(start, goal, obstacles, resolution)
if path:
    print("Path found:", path)
else:
    print("No path found")
