from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import tikzplotlib
from typing import Union, List


class Section(Enum):
    PRELIMINARIES = 1
    GAUSSIAN_COX_PROCESSES = 2
    FAVARD_KERNELS = 3
    GRAPH_ANOMALIES = 4


class Size(Enum):
    SINGLE_PLOT = 1
    DOUBLE_PLOT = 2


class Plot(ABC):
    def __init__(self, name: str, use_tikz: bool = True):
        self.name = name
        self.use_tikz = use_tikz

    def run_plot(self, save: bool = False, folder_name: str = None):
        plt.clf()
        plt.cla()
        self._plot_code()
        if save:
            if self.use_tikz:
                tikzplotlib.save(
                    self._get_tikz_location(folder_name),
                    axis_height="\\{}height".format(self.name.replace("_", "")),
                    axis_width="\\{}width".format(self.name.replace("_", "")),
                )
            else:
                print(self.name)
                breakpoint()
                plt.savefig(
                    folder_name + self.name + "_plot.png",
                    bbox_inches="tight",
                    dpi=400,
                )

        else:
            plt.show()

    def _get_tikz_location(self, folder_name: str = None):
        return folder_name + self.name + "_diagram.tex"

    @abstractmethod
    def _plot_code(self):
        pass


class Figure(ABC):
    """
    The Figure class is an abstract class that represents a figure in a paper.
    It is passed a list of plots, which will be placed as subfigures in the
    figure. To use a Figure, first instantiate the plots you want in it,
    place them in a list, and then pass that list to the Figure constructor,
    along with the appropriate section and the name of the figure.
    The name of the figure corresponds to the
    label:
        \label{section(taken from the Section):fig:name}
    """

    def __init__(self, plots: List[Plot], section: Section, name: str):
        self.plots = plots
        self.name = name
        self.section = section
        self.subfigure_width = 0.9 / len(plots)

    def run_figure(self, save: bool = False):
        for plot in self.plots:
            folder_name = self._get_folder_name()
            plot.run_plot(save=save, folder_name=folder_name)

        if save:
            self._save_figure()

    def _get_figure_label(self):
        return self.section.name.lower() + ":fig:" + self.name

    def _save_figure(self):
        """
        Saves the figure code to the corresponding tex file.
        """
        figure_string = self._get_figure_string()
        with open(self._get_figure_location(), "w") as f:
            f.write(figure_string)
            f.close()

    def _get_figure_string(self):
        figure_string = "\\begin{figure}[ht]\n"
        figure_string += "\t\\centering\n"
        for plot in self.plots:
            figure_string += (
                "\t\t\\begin{subfigure}{" + str(self.subfigure_width) + "\\textwidth}\n"
            )
            figure_string += (
                "\t\t\\input{"
                + plot._get_tikz_location(self._get_folder_name())
                + "}\n"
            )
            figure_string += "\t\t\\end{subfigure}\n"
        figure_string += "\t\\caption[" + self._get_short_caption() + "]"
        figure_string += "{" + self._get_long_caption() + "}\n"
        figure_string += "\t\\label{" + self._get_figure_label() + "}\n"
        figure_string += "\\end{figure}"
        return figure_string

    def _get_figure_location(self):
        return self._get_folder_name() + self.name + "_figure.tex"

    def _get_folder_name(self):
        return (
            "/home/william/phd/tex_projects/phdtex/figures/"
            + self.section.name.lower()
            + "/"
        )

    @abstractmethod
    def _get_short_caption(self) -> str:
        """
        The caption for the figure that will appear in  the "
        List of Figures" section.
        """
        pass

    @abstractmethod
    def _get_long_caption(self) -> str:
        """
        The caption for the figure that will appear below the figure.
        """
        pass


class ClassificationPlot(Plot):
    def __init__(self, section: Section, name: str):
        self.name = name
        self.section = section

    def _plot_code(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_xlabel("x label")
        ax.set_ylabel("y label")
        ax.set_title("Simple Plot")

    def _get_caption(self):
        return "This is the caption for a test figure."


if __name__ == "__main__":
    section = Section.GAUSSIAN_COX_PROCESSES
    name = "test"
    # caption = "test caption"
    classification_plot = ClassificationPlot(
        section,
        name,
    )
    classification_plot.run_plot(save=True)
