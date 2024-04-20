
class CONFIG:
    def __init__(self):
        self.ACCEPTANCE_RADIUS = 0.4

        self.PATH = [(0, -1), (1, -2), (2, -3), (3, -3), (4, -2), (5, -1), (6, 0), (7, 1), (8, 2), (9,3), (10,3), (11, 2), (12, 1), (12, 0), (12, -1), (11, -2), (10, -3), (9, -3), (8, -2), (7, -1), (5, 1), (4, 2), (3, 3), (2, 3), (1, 2), (0, 1), (0, 0)]

        self.TARGET_VELOCITY = 1

        self.floor_path = "scenes/empty_floor.xml"

        self.car_model_path = "models/mushr_car/model.xml"

        self.velocity_data_path = "data/velocity.txt"

        self.position_data_path = "data/position.txt"

        self.box_size = (0.1, 0.1, 0.1)
        self.box_rgba = (0.8, 0.3, 0.3, 1.0)
        self.target_color = (0.1, 0.1, 0.8, 1.0)