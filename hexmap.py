"""
Created on Sun Jan 23 14:07:18 2022
@author: richa
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from colorsys import hls_to_rgb, hsv_to_rgb
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List, Set
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
GENERAL_BORDER_COLOR = (0, 0, 0)

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
    highlight_offset = attr.ib(type=int, default=3)
    max_highlight_ticks = attr.ib(type=int, default=15)

    def __attrs_post_init__(self):
        self.vertices = self.compute_vertices()
        self.highlight_tick = 0
        self.data = None
        self.linked_data = None
        average_economic_cluster = -1

    def update(self, linked_data):
        """Updates tile highlights"""
        if self.highlight_tick > 0:
            self.highlight_tick -= 1

        self.linked_data = linked_data
        self.average_economic_cluster = np.average([d['Economic Cluster'] for d in self.linked_data]) if self.linked_data else -1

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

    def _compute_single_level_neighbours(self, hexagons: List[HexagonTile]) -> List[HexagonTile]:
        # could cache results for performance
        return [hexagon for hexagon in hexagons if self.is_neighbour(hexagon)]

    def compute_neighbours(self, hexagons: List[HexagonTile], distance) -> List[HexagonTile]:
        # could cache results for performance

        prev_neighbours = self._compute_single_level_neighbours(hexagons)
        for i in range(distance - 1):
            new_neighbours = []
            for prev_neighbour in prev_neighbours:
                new_neighbours += prev_neighbour._compute_single_level_neighbours(hexagons)

            new_neighbours = [x for x in new_neighbours if x not in prev_neighbours]
            prev_neighbours = new_neighbours

        return list(prev_neighbours)

    def collide_with_point(self, point: Tuple[float, float]) -> bool:
        """Returns True if distance from center to point is less than horizontal_length"""
        return math.dist(point, self.center) < self.minimal_radius

    @lru_cache(maxsize=8192)
    def is_neighbour(self, hexagon: HexagonTile) -> bool:
        """Returns True if hexagon center is approximately
        2 minimal radiuses away from own center
        """
        distance = math.dist(hexagon.center, self.center)
        return math.isclose(distance, 2 * self.minimal_radius, rel_tol=0.1) and hexagon.data is not None

    def render(self, screen) -> None:
        """Renders the hexagon on the screen"""
        pygame.draw.polygon(screen, self.highlight_color, self.vertices)
        pygame.draw.aalines(screen, GENERAL_BORDER_COLOR, closed=True, points=self.vertices)

        if self.linked_data:
            font = pygame.font.SysFont("arial", 24, bold=True)
            text_surface = font.render(f"{self.average_economic_cluster:.2f}", True, (0, 0, 0))
            screen.blit(text_surface, np.subtract(self.center, (15, 15)))
            # TODO: Centre text

    def render_highlight(self, screen, border_color) -> None:
        """Draws a border around the hexagon with the specified color"""
        self.highlight_tick = self.max_highlight_ticks
        # pygame.draw.polygon(screen, self.highlight_color, self.vertices)
        pygame.draw.aalines(screen, border_color, closed=True, points=self.vertices)

    @property
    def center(self) -> Tuple[float, float]:
        """Centre of the hexagon"""
        x, y = self.position  # pylint: disable=invalid-name
        return (x, y + self.radius)

    @property
    def minimal_radius(self) -> float:
        """Horizontal length of the hexagon"""
        # https://en.wikipedia.org/wiki/Hexagon#Parameters
        return self.radius * math.cos(math.radians(30))

    @property
    def highlight_color(self) -> Tuple[int, ...]:
        """color of the hexagon tile when rendering highlight"""
        offset = self.highlight_offset * self.highlight_tick
        brighten = lambda x, y: x + y if x + y < 255 else 255
        # TODO: Brighten
        brighten = lambda x, y: x

        return tuple(brighten(x, offset) for x in self.color)

    @property
    def color(self) -> Tuple[int, ...]:
        if self.data is None:
            return (0, 0, 0)

        if not self.linked_data:
            return (255, 255, 255)

        palette = list(colour.Color("red").range_to(colour.Color("blue"), 11))
        value = self.average_economic_cluster
        color = palette[int(np.round(value))].rgb

        return (color[0] * 255, color[1] * 255, color[2] * 255)

        value = (value, 0.8, 0.8)

        rgb = hsv_to_rgb(*value)

        return (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)

        return (0, 255 / np.average([d["Economic Cluster"] for d in self.linked_data]), 0)

        return (0, 0, 0) if self.data is None else (0, 255 * self.data[0], 0)


    def __hash__(self):
        return self.position.__hash__()


class FlatTopHexagonTile(HexagonTile):
    def compute_vertices(self) -> List[Tuple[float, float]]:
        """Returns a list of the hexagon's vertices as x, y tuples"""
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
        """Centre of the hexagon"""
        x, y = self.position  # pylint: disable=invalid-name
        return (x, y + self.minimal_radius)


def create_hexagon(position, radius=50, flat_top=False) -> HexagonTile:
    """Creates a hexagon tile at the specified position"""
    class_ = FlatTopHexagonTile if flat_top else HexagonTile
    return class_(radius, position)


def init_hexagons(num_x=8, num_y=9, flat_top=False) -> List[HexagonTile]:
    """Creates a hexaogonal tile map of size num_x * num_y"""
    # pylint: disable=invalid-name
    leftmost_hexagon = create_hexagon(position=(150, 150), flat_top=flat_top)
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
    lines = text.splitlines()
    for i, l in enumerate(lines):
        screen.blit(font.render(l, 0, color), (x, y + fsize * i))


def render(screen, hexagons, quantization_error, topological_error, economic_std, epoch):
    """Renders hexagons on the screen"""
    screen.fill((0, 0, 0))
    for hexagon in hexagons:
        hexagon.render(screen)

    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 20)

    render_multi_line(screen, font, (255, 255, 255),
                               f"Epoch : {epoch}\n"
                               f"Quantization error : {quantization_error:.2f}\n"
                               f"Topological error : {topological_error:.2f}\n"
                               f"Economic Cluster STD : {economic_std:.2f}",
                      50, 50, 24)

    # # draw borders around colliding hexagons and neighbours
    # TODO: Make it sparkle
    mouse_pos = pygame.mouse.get_pos()
    colliding_hexagons = [
        hexagon for hexagon in hexagons if hexagon.collide_with_point(mouse_pos)
    ]

    for hexagon in colliding_hexagons:
        hexagon.render_highlight(screen, border_color=(255, 0, 0))
        municipalities = [f'({d["Economic Cluster"].values[0]}) {d["Municipality"].values[0]}' for d in hexagon.linked_data]
        render_multi_line(screen, small_font, (255, 255, 255), "\n".join(municipalities), 1200 - 300, 50, 20)


    pygame.display.flip()


def init_grid_cells_data():
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
    return np.sqrt(np.mean(np.square(data1 - data2)))


def _find_k_cells_with_min_distance(data_item, hexagons, k=1) -> List[HexagonTile]:
    sorted_hexagons = sorted(hexagons, key=lambda h: distance(h.data, data_item))
    return sorted_hexagons[:k]


    min_distance, min_distance_hexagon = math.inf, None
    for hexagon in hexagons:
        # Find the hexagon with the least values
        curr_distance = distance(hexagon.data, data_item)



        min_distance = min(min_distance, curr_distance)
        if curr_distance == min_distance:
            min_distance_hexagon = hexagon

    return min_distance_hexagon


def _update_hexagons(data_item, center_hexagon: HexagonTile, hexagons):
    first_neighbours = center_hexagon.compute_neighbours(hexagons, 1)
    second_neighbours = center_hexagon.compute_neighbours(hexagons, 2)
    levels = [[center_hexagon], first_neighbours, second_neighbours]

    for level, hexagons in enumerate(levels):
        for hexagon in hexagons:
            hexagon.data += LEARNING_RATE * NEIGHBOURHOOD_UPDATES[level] * (data_item - hexagon.data)
            #hexagon.data = (1 - LEARNING_RATE) * hexagon.data +  LEARNING_RATE * NEIGHBOURHOOD_UPDATES[level] * (data_item - hexagon.data)


def run_som_epoch(data_set, hexagons):
    # np.random.shuffle(data_set)
    for data_item in data_set:
        nearest_hexagon = _find_k_cells_with_min_distance(data_item, hexagons)[0]
        _update_hexagons(data_item, nearest_hexagon, hexagons)
        # print(list(map(lambda x: x.data, hexagons)))


def run_som_algorithm(data_set, hexagons, epoch_count):
    for epoch in range(epoch_count):
        print(f"\rEpoch {epoch}/{epoch_count}", end='')
        
        run_som_epoch(data_set, hexagons)


def read_data(path):
    df = pd.read_csv(path)
    return df

    data = np.genfromtxt(path, delimiter=',')[1:]  # Skip header line

    data_items = {}


    # Normalize economic cluster and votes counts
    for item in data:
        total_votes = sum(item[3:])
        for i in range(3, item.shape[0]):
            item[i] /= total_votes

        item[1] /= 10.0

    for item in data:
        data_items[item[0]] = np.array([item[1]] + list(item[3:]))

    return data_items

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

def main():
    """Main function"""

    distance(np.arange(5), np.arange(5))

    show_graph()

    pygame.init()
    screen = pygame.display.set_mode((1200, 1200))
    clock = pygame.time.Clock()
    hexagons = init_hexagons(flat_top=False)

    grid_cells_data = init_grid_cells_data()

    for i, hexagon in enumerate(hexagons):
        x, y = np.unravel_index(i, (GRID_SIZE, GRID_SIZE))
        hexagon.data = grid_cells_data[x][y]

    # Remove all hexagons that are irrelevant (that form a full grid)
    hexagons = [h for h in hexagons if h.data is not None]

    data = read_data("/Users/rone/Downloads/Elec_24.csv")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    raw_data = pd.DataFrame.copy(data)
    raw_data[numeric_columns] = raw_data[numeric_columns].apply(zscore)

    raw_data = raw_data.drop(columns=["Municipality", "Economic Cluster"], axis=1).to_numpy()

    terminated = False
    epoch = 0
    while not terminated:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if epoch < MAX_EPOCHS:

            run_som_epoch(raw_data, hexagons)
            epoch += 1
            topological_error = 0

            data_links = defaultdict(list)
            for i, row in enumerate(raw_data):
                min_distance_hexagons = _find_k_cells_with_min_distance(row, hexagons, k=2)
                nearest_hexagon = min_distance_hexagons[0]
                data_links[nearest_hexagon].append(data.iloc[[i]])

                second_nearest_hexagon = min_distance_hexagons[1]
                topological_error += distance(nearest_hexagon.data, second_nearest_hexagon.data)

            quantization_error = 0

            economic_std = 0
            economic_std_div = 0

            for hexagon in hexagons:
                links = data_links.get(hexagon, [])
                hexagon.update(links)

                raw_links = [raw_data[l.index.values[0]] for l in links]

                if raw_links:
                    economic_std += np.std(np.array(raw_links)[:, 0])
                    economic_std_div += 1

                for link in raw_links:
                    quantization_error += distance(link, hexagon.data)

            topological_error /= len(data)
            quantization_error /= len(data)
            economic_std /= economic_std_div

        QUANTIZATION_ERRORS.append(quantization_error)
        TOPOLOGICAL_ERRORS.append(topological_error)
        ECONOMIC_STDS.append(economic_std)

        render(screen, hexagons, quantization_error, topological_error, economic_std, epoch)
        clock.tick(10)
    pygame.display.quit()


if __name__ == "__main__":
    main()
