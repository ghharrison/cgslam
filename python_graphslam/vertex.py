# Copyright (c) 2020 Jeff Irion and contributors

"""A ``Vertex`` class.

"""

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from .pose.r2 import PoseR2
from .pose.se2 import PoseSE2

# Colors for robots and for landmarks
VERTEX_COLOR = ['r', 'g', 'm', 'y', 'c', 'k']

# pylint: disable=too-few-public-methods
class Vertex:
    """A class for representing a vertex in Graph SLAM.

    Parameters
    ----------
    vertex_id : int
        The vertex's unique ID
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex
    vertex_index : int, None
        The vertex's index in the graph's ``vertices`` list
    fixed : bool
        Whether this vertex should be fixed

    Attributes
    ----------
    id : int
        The vertex's unique ID
    index : int, None
        The vertex's index in the graph's ``vertices`` list
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex
    fixed : bool
        Whether this vertex should be fixed

    """
    def __init__(self, vertex_id, pose, vertex_index=None, fixed=False):
        self.id = vertex_id
        self.pose = pose
        self.index = vertex_index
        self.fixed = fixed

    def __eq__(self, other):
        return (self.id == other.id)

    def to_g2o(self):
        """Export the vertex to the .g2o format.

        Returns
        -------
        str
            The vertex in .g2o format

        """
        if isinstance(self.pose, PoseSE2):
            return "VERTEX_SE2 {} {} {} {}\n".format(self.id, self.pose[0], self.pose[1], self.pose[2])

        raise NotImplementedError

    def plot(self, color='r', marker='o', markersize=3):
        """Plot the vertex.

        Parameters
        ----------
        color : str
            The color that will be used to plot the vertex
        marker : str
            The marker that will be used to plot the vertex
        markersize : int
            The size of the plotted vertex

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        if isinstance(self.pose, (PoseR2, PoseSE2)):
            x, y = self.pose.position
            if "_" in self.id:
                color_num = int(self.id[0])
            else:
                color_num = 6
                markersize = 9
            plt.plot(x, y, color=VERTEX_COLOR[color_num-1], marker=marker, markersize=markersize)

        else:
            raise NotImplementedError
