import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None

def distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def step_from_to(node1, node2, step_size=1.0):
    theta = np.arctan2(node2.y - node1.y, node2.x - node1.x)
    new_x = node1.x + step_size * np.cos(theta)
    new_y = node1.y + step_size * np.sin(theta)
    new_theta = theta  # Simplification for straight lines
    return Node(new_x, new_y, new_theta)

def check_collision(node, obstacles):
    for (ox, oy, size) in obstacles:
        if distance(node, Node(ox, oy, 0)) <= size:
            return True
    return False

def RRT(start, goal, obstacles, num_iterations=500, step_size=1.0):
    nodes = [start]
    
    for _ in range(num_iterations):
        rand = Node(random.uniform(-10, 10), random.uniform(-10, 10), 0)  # Random node
        nearest_node = min(nodes, key=lambda node: distance(node, rand))
        new_node = step_from_to(nearest_node, rand, step_size)
        
        if not check_collision(new_node, obstacles):
            new_node.parent = nearest_node
            nodes.append(new_node)
            
            if distance(new_node, goal) < step_size:
                # Goal reached
                path_x = []
                path_y = []
                while new_node:
                    path_x.append(new_node.x)
                    path_y.append(new_node.y)
                    new_node = new_node.parent
                return path_x[::-1], path_y[::-1]
    return None

def plot_path(start, goal, path, nodes, obstacles):
    fig, ax = plt.subplots()
    if path:
        for node in path:
            plt.plot([node.x], [node.y], 'ro')  # Path nodes in red
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'r')
    
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'gray')  # Tree in gray

    for (ox, oy, size) in obstacles:
        circle = plt.Circle((ox, oy), size, color='b', fill=True)
        ax.add_patch(circle)

    plt.plot(start.x, start.y, 'go', markersize=10)  # Start in green
    plt.plot(goal.x, goal.y, 'yo', markersize=10)  # Goal in yellow
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('RRT Path Planning')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# # Example Usage
# start = Node(0, 0, np.pi/2)
# goal = Node(9, 9, 0)
# obstacles = [(5, 5, 2), (3, 3, 1), (7, 7, 1.5)]  # (x, y, size)

# path, nodes = RRT(start, goal, obstacles)
# plot_path(start, goal, path, nodes, obstacles)
