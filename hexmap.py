# Computational Biology Exercise 3
# SOM for elections
# The code for drawing hexagons is partially based on https://github.com/rbaltrusch/pygame_examples/tree/master/code/hexagonal_tiles
# Yair Yariv Yardeni - 315009969
# Ron Even           - 313260317

from __future__ import annotations

import math
import pathlib
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import List
from typing import Tuple

import attr
import colour
import numpy as np
import pygame
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.ticker import AutoLocator
from scipy.stats import zscore
from numba import njit


MAX_EPOCHS = 50

FIELDS_COUNT = 14
GRID_SIZE = 9
GRID_ROW_SIZES = [5, 6, 7, 8, 9, 8, 7, 6, 5]

BLACK_RGB = (0, 0, 0)
WHITE_RGB = (255, 255, 255)

GENERAL_BORDER_COLOR = BLACK_RGB
FONT = None
SMALL_FONT = None


LEARNING_RATE = 0.3
NEIGHBOURHOOD_UPDATES = {
    0: 0.2,
    1: 0.1,
    2: 0.05
}


@attr.s(hash=False)
class HexagonTile:
    """Hexagon class"""
    radius = attr.ib(type=float)
    position = attr.ib(type=Tuple[float, float])
    highlight_offset = attr.ib(type=int, default=30)
    max_highlight_ticks = attr.ib(type=int, default=5)

    def __attrs_post_init__(self):
        self.vertices = self.compute_vertices()
        self.highlight_tick = 0
        self.data = None
        self.linked_data = None

    def update_linked_data(self, linked_data):
        """
        Updates linked data to the current hexagon
        """
        self.linked_data = linked_data
        self.average_economic_cluster = np.average([d['Economic Cluster'] for d in self.linked_data]) if self.linked_data else -1

    def update(self):
        """Updates tile highlights"""
        if self.highlight_tick > 0:
            self.highlight_tick -= 1

    def compute_vertices(self) -> List[Tuple[float, float]]:
        """Returns a list of the hexagon's vertices as x, y tuples"""
        # pylint: disable=invalid-name
        x, y = self.position
        half_radius = self.radius / 2
        minimal_radius = self.minimal_radius
        return [
            (x, y),
            (x - minimal_radius, y + half_radius),
            (x - minimal_radius, y + 3 * half_radius),
            (x, y + 2 * self.radius),
            (x + minimal_radius, y + 3 * half_radius),
            (x + minimal_radius, y + half_radius),
        ]

    def _compute_immediate_neighbours(self, hexagons: List[HexagonTile]) -> List[HexagonTile]:
        """
        Returns a list of immediate neighbours of this hexagon
        """
        return [hexagon for hexagon in hexagons if self.is_neighbour(hexagon)]

    def compute_neighbours(self, hexagons: List[HexagonTile], level) -> List[HexagonTile]:
        """
        Returns all neighbors of this hexagon in given level
        """
        prev_neighbours = self._compute_immediate_neighbours(hexagons)
        for i in range(level - 1):
            new_neighbours = []
            for prev_neighbour in prev_neighbours:
                new_neighbours += prev_neighbour._compute_immediate_neighbours(hexagons)

            new_neighbours = [x for x in new_neighbours if x not in prev_neighbours]
            prev_neighbours = new_neighbours

        return list(prev_neighbours)

    def collide_with_point(self, point: Tuple[float, float]) -> bool:
        """
        Returns True if distance from center to point is less than horizontal_length
        """
        return math.dist(point, self.center) < self.minimal_radius

    @lru_cache(maxsize=8192)
    def is_neighbour(self, hexagon: HexagonTile) -> bool:
        """
        Returns True if hexagon center is approximately
        2 minimal radiuses away from own center
        """
        distance = math.dist(hexagon.center, self.center)
        return math.isclose(distance, 2 * self.minimal_radius, rel_tol=0.1) and hexagon.data is not None

    def render(self, screen) -> None:
        """
        Renders the hexagon on the screen
        """
        pygame.draw.polygon(screen, self.highlight_color, self.vertices)
        pygame.draw.aalines(screen, GENERAL_BORDER_COLOR, closed=True, points=self.vertices)

        if self.linked_data:
            text_surface = FONT.render(f"{self.average_economic_cluster:.2f}", True, BLACK_RGB)
            screen.blit(text_surface, np.subtract(self.center, (21.0675, 13.3)))

    def render_highlight(self, screen, border_color) -> None:
        """
        Draws a border around the hexagon with the specified color
        """
        self.highlight_tick = self.max_highlight_ticks
        # pygame.draw.polygon(screen, self.highlight_color, self.vertices)
        pygame.draw.aalines(screen, border_color, closed=True, points=self.vertices)

    @property
    def center(self) -> Tuple[float, float]:
        """
        Center of the hexagon
        """
        x, y = self.position
        return x, y + self.radius

    @property
    def minimal_radius(self) -> float:
        """
        Horizontal length of the hexagon
        """
        # https://en.wikipedia.org/wiki/Hexagon#Parameters
        return self.radius * math.cos(math.radians(30))

    @property
    def highlight_color(self) -> Tuple[int, ...]:
        """
        Color of the hexagon tile when rendering highlight
        """
        offset = self.highlight_offset * self.highlight_tick
        brighten = lambda x, y: x + y if x + y < 255 else 255

        return tuple(brighten(x, offset) for x in self.color)

    @property
    def color(self) -> Tuple[int, ...]:
        """
        Returns the color of the hexagon.
        If the hexagon has no data (is a padding hexagon) the color is black (the same as the background).
        If the hexagon has no linked data (i.e no municipalities linked to it), the color is white.
        Otherwise, the color is chosen from a range, according to the average linked municipalities economic cluster.
        """
        if self.data is None:
            return BLACK_RGB

        if not self.linked_data:
            return WHITE_RGB

        palette = list(colour.Color("red").range_to(colour.Color("blue"), 11))
        value = self.average_economic_cluster
        color = palette[int(np.round(value))].rgb

        return (color[0] * 255, color[1] * 255, color[2] * 255)

    def __hash__(self):
        return self.position.__hash__()


class FlatTopHexagonTile(HexagonTile):
    def compute_vertices(self) -> List[Tuple[float, float]]:
        """
        Returns a list of the hexagon's vertices as x, y tuples
        """
        # pylint: disable=invalid-name
        x, y = self.position
        half_radius = self.radius / 2
        minimal_radius = self.minimal_radius
        return [
            (x, y),
            (x - half_radius, y + minimal_radius),
            (x, y + 2 * minimal_radius),
            (x + self.radius, y + 2 * minimal_radius),
            (x + 3 * half_radius, y + minimal_radius),
            (x + self.radius, y),
        ]

    @property
    def center(self) -> Tuple[float, float]:
        """
        Centre of the hexagon
        """
        x, y = self.position  # pylint: disable=invalid-name
        return (x, y + self.minimal_radius)


def create_hexagon(position, radius=50, flat_top=False) -> HexagonTile:
    """
    Creates a hexagon tile at the specified position
    """
    class_ = FlatTopHexagonTile if flat_top else HexagonTile
    return class_(radius, position)


def init_hexagons(num_x=8, num_y=9, flat_top=False) -> List[HexagonTile]:
    """
    Creates a hexaogonal tile map of size num_x * num_y
    """
    # pylint: disable=invalid-name
    leftmost_hexagon = create_hexagon(position=(150, 200), flat_top=flat_top)
    hexagons = [leftmost_hexagon]
    for x in range(num_y):
        if x:
            # alternate between bottom left and bottom right vertices of hexagon above
            index = 2 if x % 2 == 1 or flat_top else 4
            position = leftmost_hexagon.vertices[index]
            leftmost_hexagon = create_hexagon(position, flat_top=flat_top)
            hexagons.append(leftmost_hexagon)

        # place hexagons to the left of leftmost hexagon, with equal y-values.
        hexagon = leftmost_hexagon
        for i in range(num_x):
            x, y = hexagon.position  # type: ignore
            if flat_top:
                if i % 2 == 1:
                    position = (x + hexagon.radius * 3 / 2, y - hexagon.minimal_radius)
                else:
                    position = (x + hexagon.radius * 3 / 2, y + hexagon.minimal_radius)
            else:
                position = (x + hexagon.minimal_radius * 2, y)
            hexagon = create_hexagon(position, flat_top=flat_top)
            hexagons.append(hexagon)

    return hexagons


def render_multi_line(screen, font, color, text, x, y, fsize):
    """
    Renders multi-line text on the pygame screen
    """
    lines = text.splitlines()
    for i, l in enumerate(lines):
        screen.blit(font.render(l, 0, color), (x, y + fsize * i))


def render(screen, hexagons, quantization_error, topological_error, economic_std, epoch):
    """
    Renders hexagons on the screen
    """
    screen.fill(BLACK_RGB)
    for hexagon in hexagons:
        hexagon.update()
        hexagon.render(screen)

    render_multi_line(screen, FONT, WHITE_RGB,
                               f"Epoch : {epoch}\n"
                               f"Quantization error : {quantization_error:.2f}\n"
                               f"Topological error : {topological_error:.2f}\n"
                               f"Economic Cluster STD : {economic_std:.2f}",
                      50, 50, 24)

    # draw borders around colliding hexagons and neighbours
    mouse_pos = pygame.mouse.get_pos()
    colliding_hexagons = [
        hexagon for hexagon in hexagons if hexagon.collide_with_point(mouse_pos)
    ]

    for hexagon in colliding_hexagons:
        hexagon.render_highlight(screen, border_color=BLACK_RGB)
        municipalities = [f'({d["Economic Cluster"].values[0]}) {d["Municipality"].values[0]}' for d in hexagon.linked_data]
        render_multi_line(screen, SMALL_FONT, (255, 255, 255), "\n".join(municipalities), 1200 - 300, 50, 20)

    pygame.display.flip()


def init_grid_cells_data():
    """
    Initializes the grid random data according to the needed pattern.
    Cells that do not exist in the grid, will be represented as None
    """
    result = []
    for i in range(GRID_SIZE):
        current_row_size = GRID_ROW_SIZES[i]

        none_items_count = GRID_SIZE - current_row_size
        none_items_left_count = math.ceil(none_items_count / 2.0)
        none_items_right_count = none_items_count - none_items_left_count
        full_items = [np.random.uniform(-1, 1, FIELDS_COUNT) for _ in range(current_row_size)]

        result.append([None] * none_items_left_count + full_items + [None] * none_items_right_count)

    return result


@njit
def distance(data1, data2):
    """
    RMS distance function. njit decorator used for optimization (by pre-compiling it)
    """
    return np.sqrt(np.mean(np.square(data1 - data2)))


def _find_k_cells_with_min_distance(data_item, hexagons, k=1) -> List[HexagonTile]:
    """
    Returns a list of neighbours with the k closest hexagons to the given data item
    """
    sorted_hexagons = sorted(hexagons, key=lambda h: distance(h.data, data_item))
    return sorted_hexagons[:k]


def _update_hexagons(data_item, center_hexagon: HexagonTile, hexagons):
    """
    Runs an update iteration on with the given data item on the given center hexagon and its surroundings
    """
    first_neighbours = center_hexagon.compute_neighbours(hexagons, 1)
    second_neighbours = center_hexagon.compute_neighbours(hexagons, 2)
    levels = [[center_hexagon], first_neighbours, second_neighbours]

    for level, hexagons in enumerate(levels):
        for hexagon in hexagons:
            hexagon.data += LEARNING_RATE * NEIGHBOURHOOD_UPDATES[level] * (data_item - hexagon.data)


def run_som_epoch(data_set, hexagons):
    """
    Runs a single SOM epoch
    """
    for data_item in data_set:
        nearest_hexagon = _find_k_cells_with_min_distance(data_item, hexagons)[0]
        _update_hexagons(data_item, nearest_hexagon, hexagons)


def run_som_algorithm(data, raw_data, hexagons, epoch_count):
    """
    Runs epoch_count SOM epochs, eventually printing the error values.
    """
    global QUANTIZATION_ERRORS
    global TOPOLOGICAL_ERRORS
    global ECONOMIC_STDS

    QUANTIZATION_ERRORS = []
    TOPOLOGICAL_ERRORS = []
    ECONOMIC_STDS = []

    for epoch in range(epoch_count):
        print(f"\rEpoch {epoch}/{epoch_count}", end='')
        run_som_epoch(raw_data, hexagons)

    quantization_error, topological_error, _ = do_after_epoch_update(data, raw_data, hexagons)
    print("\rDone!")
    print_errors(quantization_error, topological_error)


def read_data(path):
    df = pd.read_csv(path)

    # Normalize economic cluster and votes counts
    for row_index, row in enumerate(df.values):
        all_votes = int(row[2])
        all_valid_votes = sum(row[3:])
        invalid_votes = all_votes - all_valid_votes
        df.iloc[row_index, 2] = (invalid_votes / all_votes) * 100  # Calculate invalid votes
        for i in range(3, row.shape[0]):
            df.iloc[row_index, i] = (df.iloc[row_index, i] / all_votes) * 100

    df = df.rename(columns={"Total Votes": "Invalid votes"})
    return df

QUANTIZATION_ERRORS = []
TOPOLOGICAL_ERRORS = []
ECONOMIC_STDS = []


def show_graph():
    """
    Shows the progress graph
    """

    fig, ax_plot = plt.subplots()
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])

    ax_plot.set_xlabel("Generations")
    ax_plot.set_ylabel("Fitness score")

    quantization_line, = ax_plot.plot([], [], label="Quantization Error")
    topological_line, = ax_plot.plot([], [], label="Topological Error")
    std_line, = ax_plot.plot([], [], label="Economic Cluster STD")

    ax_plot.set_xlabel("Generations")
    ax_plot.set_ylabel("Value")

    def update(frame):
        quantization_errors = QUANTIZATION_ERRORS[:]
        topological_errors = TOPOLOGICAL_ERRORS[:]
        economic_stds = ECONOMIC_STDS[:]

        epoch = max(len(quantization_errors), len(topological_errors), len(economic_stds))

        plt.suptitle(f"Epoch {epoch}")
        plt.legend(loc="upper right")

        if epoch >= MAX_EPOCHS:
            # No more sick cells. Stop simulation
            ani.pause()
            return

        quantization_line.set_data(np.arange(epoch), np.array(quantization_errors))
        topological_line.set_data(np.arange(epoch), np.array(topological_errors))
        std_line.set_data(np.arange(epoch), np.array(economic_stds))

        ax_plot.set_xlim(0, epoch + 1)
        ax_plot.set_ylim(0, max([0] + quantization_errors + topological_errors + economic_stds) + 0.01)
        ax_plot.xaxis.set_major_locator(AutoLocator())
        ax_plot.yaxis.set_major_locator(AutoLocator())

        return quantization_line, topological_line, std_line

    ani = animation.FuncAnimation(fig, update, interval=1)
    plt.show(block=False)


def do_after_epoch_update(data, raw_data, hexagons):
    """
    Updates the linked municipalities and the errors after an epoch is complete
    """
    topological_error = 0
    data_links = defaultdict(list)
    for i, row in enumerate(raw_data):
        min_distance_hexagons = _find_k_cells_with_min_distance(row, hexagons, k=2)
        nearest_hexagon = min_distance_hexagons[0]
        data_links[nearest_hexagon].append(data.iloc[[i]])

        second_nearest_hexagon = min_distance_hexagons[1]
        topological_error += int(not second_nearest_hexagon.is_neighbour(nearest_hexagon))

    quantization_error = 0
    economic_std = 0
    economic_std_div = 0

    for hexagon in hexagons:
        links = data_links.get(hexagon, [])
        hexagon.update_linked_data(links)

        raw_links = [raw_data[l.index.values[0]] for l in links]

        if raw_links:
            economic_std += np.std(np.array(raw_links)[:, 0])
            economic_std_div += 1

        for link in raw_links:
            quantization_error += distance(link, hexagon.data)

    topological_error /= len(data)
    quantization_error /= len(data)
    economic_std /= economic_std_div

    return quantization_error, topological_error, economic_std


def print_errors(quantization_error, topological_error):
    print(f"Quantization error : {quantization_error}")
    print(f"Topological error  : {topological_error}")


def init_hexagons_for_som():
    hexagons = init_hexagons(flat_top=False)
    grid_cells_data = init_grid_cells_data()
    for i, hexagon in enumerate(hexagons):
        x, y = np.unravel_index(i, (GRID_SIZE, GRID_SIZE))
        hexagon.data = grid_cells_data[x][y]
    # Remove all hexagons that are irrelevant (that form a full grid)
    hexagons = [h for h in hexagons if h.data is not None]
    return hexagons


def main():
    data_path = sys.argv[1]

    global FONT
    global SMALL_FONT

    # Force compiling distance function before rendering
    distance(np.arange(5), np.arange(5))

    show_graph()

    pygame.init()

    FONT = pygame.font.SysFont("arial", 24, bold=True)
    SMALL_FONT = pygame.font.SysFont("arial", 20)

    screen = pygame.display.set_mode((1200, 1000))
    clock = pygame.time.Clock()
    hexagons = init_hexagons_for_som()

    data = read_data(data_path)

    raw_data = pd.DataFrame.copy(data)
    raw_data = raw_data.drop(columns=["Municipality", "Economic Cluster"], axis=1).to_numpy()

    terminated = False
    printed_stats = False
    epoch = 0
    while not terminated:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if epoch < MAX_EPOCHS:

            run_som_epoch(raw_data, hexagons)
            epoch += 1
            quantization_error, topological_error, economic_std = do_after_epoch_update(data, raw_data, hexagons)

            QUANTIZATION_ERRORS.append(quantization_error)
            TOPOLOGICAL_ERRORS.append(topological_error)
            ECONOMIC_STDS.append(economic_std)

        elif not printed_stats:
            print_errors(QUANTIZATION_ERRORS[-1], TOPOLOGICAL_ERRORS[-1])
            output_text = "Hexagons and mapped municipalities (ordered from top left to right):\n"
            for i, hexagon in enumerate(hexagons):
                output_text += f"Hexagon {i}: {', '.join([m['Municipality'].values[0] for m in hexagon.linked_data]) if len(hexagon.linked_data) else 'None'}\n"

            print(output_text)

            time.sleep(1)
            output_base = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

            pygame.image.save(screen, f"{output_base}_map.jpg")
            plt.savefig(f"{output_base}_plots.jpg")
            pathlib.Path(f"{output_base}_results.txt").write_text(output_text)

            printed_stats = True

        render(screen, hexagons, QUANTIZATION_ERRORS[-1], TOPOLOGICAL_ERRORS[-1], ECONOMIC_STDS[-1], epoch)
        clock.tick(30)

    pygame.display.quit()


if __name__ == "__main__":
    main()
