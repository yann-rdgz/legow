import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


class Heatmap:
    """
    A class to represent a heatmap for a slide image.
    Attributes
    ----------
    path_slide : str
        Path to the slide image file.
    coordinates_tiles : list or np.ndarray
        List or array of coordinates for the tiles.
    scores_tiles : list or np.ndarray
        List or array of scores for the tiles.
    dz_level : int, optional
        DeepZoom level to use for tiling (default is None).
    tile_size : int, optional
        Size of the tiles (default is 224).
    Methods
    -------
    plot_heatmap():
        Plots the heatmap of the slide image.
    """

    def __init__(
        self,
        path_slide: str,
        coordinates_tiles: np.ndarray,
        scores_tiles: np.ndarray,
        dz_level: int,
        tile_size=224,
    ):
        """Constructs all the necessary attributes for the Heatmap object.
        Parameters
        ----------
        path_slide : str
            Path to the slide image file.
        coordinates_tiles : np.ndarray
            Array of coordinates for the tiles.
        scores_tiles : np.ndarray
            Array of scores for the tiles.
        dz_level : int
            DeepZoom level to use for tiling.
        tile_size : int, optional
            Size of the tiles (default is 224).
        """
        self._path_slide = path_slide
        self._coordinat_tiles = coordinates_tiles
        self._scores_tiles = scores_tiles
        self._tile_size = tile_size
        self._dz_level = dz_level

    def plot_heatmap(self, img_size=(1000, 1000)):
        """Plots the heatmap of the slide image.

        Parameters
        ----------
        img_size : tuple, optional
            Size of the image (default is (1000, 1000)).
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        axs : np.ndarray of matplotlib.axes._subplots.AxesSubplot
            The axes objects containing the subplots.
        """
        slide = OpenSlide(self._path_slide)
        dz = DeepZoomGenerator(slide, self._tile_size, overlap=0)

        dim_tiling = dz.level_tiles[self._dz_level]

        heatmap = np.empty(dim_tiling) * np.nan
        heatmap[self._coordinat_tiles[:, 0], self._coordinat_tiles[:, 1]] = self._scores_tiles

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        img = np.array(slide.get_thumbnail(img_size))
        axs[0].imshow(img)
        axs[1].imshow(heatmap.T, alpha=0.9)
        return fig, axs

    def export_to_abstra(self):
        raise NotImplementedError("Method not implemented yet")
