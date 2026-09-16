"""Visualization tools for FluidGym environments."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, cKDTree
from skimage import measure

from fluidgym.simulation.helpers import get_cell_centers

DEFAULT_VIEW_KWARGS = {"elev": 15, "azim": 45}


def _crop_img(
    img: np.ndarray,
    x_margin: int = 0,
    y_margin: int = 170,
) -> np.ndarray:
    x_0 = max(x_margin, 0)
    x_1 = min(img.shape[1] - x_margin, img.shape[1])

    y_0 = max(y_margin, 0)
    y_1 = min(img.shape[0] - y_margin, img.shape[0])

    return img[y_0:y_1, x_0:x_1, :]


def _format_3d(fig: Figure, ax: Axes) -> None:
    """Format 3D plot with labels and aspect ratio."""
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Turn off ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore[attr-defined]

    # Turn off tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])  # type: ignore[attr-defined]

    # Turn off axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")  # type: ignore[attr-defined]

    # Disable panes (background faces)
    ax.xaxis.pane.set_visible(False)  # type: ignore[attr-defined]
    ax.yaxis.pane.set_visible(False)  # type: ignore[attr-defined]
    ax.zaxis.pane.set_visible(False)  # type: ignore[attr-defined]

    # Disable grid
    ax.grid(False)

    # Disable axis lines
    ax.xaxis.line.set_visible(False)  # type: ignore[attr-defined]
    ax.yaxis.line.set_visible(False)  # type: ignore[attr-defined]
    ax.zaxis.line.set_visible(False)  # type: ignore[attr-defined]
    ax.grid(False)


def _get_savefig_kwargs(filename: str) -> dict[str, str | float]:
    """Get the file format from the filename extension."""
    kwargs: dict[str, str | float] = {}
    if "." not in filename:
        raise ValueError("Filename must have an extension to determine the format.")
    kwargs["format"] = filename.split(".")[-1]
    if kwargs["format"] == "png":
        kwargs["dpi"] = 500
        kwargs["transparent"] = True
    return kwargs


def _fig_to_array(fig: Figure) -> np.ndarray:
    fig.canvas.draw()  # Render the figure to the canvas
    w, h = fig.canvas.get_width_height()
    img_np = np.frombuffer(
        fig.canvas.tostring_rgb(),  # type: ignore
        dtype=np.uint8,
    ).reshape(h, w, 3)

    return _crop_img(img_np)


def _add_cylinder(
    ax: Axes,
    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    radius: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    """Add a cylinder to a 3D plot.

    Parameters
    ----------
    ax : Axes
        The 3D axes to add the cylinder to.

    extent : tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        The extent of the plot in (x, y, z) directions.

    radius : float
        The radius of the cylinder.

    center_x : float
        The x-coordinate of the cylinder center.

    center_y : float
        The y-coordinate of the cylinder center.
    """
    color = "black"
    theta, y = np.meshgrid(
        np.linspace(0, 2 * np.pi, 100), np.linspace(0, extent[1][1], 100)
    )

    x = radius * np.cos(theta) + center_x

    # Note: In the 3D plot, z is vertical axis,
    # but in general convention, y is vertical axis
    z = radius * np.sin(theta) + center_y

    # Cylinder surface
    ax.plot_surface(  # type: ignore
        x, y, z, color=color, alpha=1.0, rstride=5, cstride=5, edgecolor="none"
    )

    r, theta_face = np.meshgrid(
        np.linspace(0, radius, 50), np.linspace(0, 2 * np.pi, 100)
    )

    x_face = r * np.cos(theta_face)
    y_face = r * np.sin(theta_face)

    # Bottom face (z=0)
    ax.plot_surface(  # type: ignore
        x_face,
        np.zeros_like(x_face),
        y_face,
        color=color,
        alpha=1.0,
        edgecolor="none",
    )

    # Top face (z=height)
    ax.plot_surface(  # type: ignore
        x_face,
        extent[2][1] * np.ones_like(x_face),
        y_face,
        color=color,
        alpha=1.0,
        edgecolor="none",
    )


def _add_airfoil(
    ax: Axes,
    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    mask: np.ndarray,
) -> None:
    """Add a cylinder to a 3D plot.

    Parameters
    ----------
    ax: Axes
        The 3D axes to add the cylinder to.

    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        The extent of the plot in (x, y, z) directions.

    mask: np.ndarray
        The airfoil mask.
    """
    color = "black"
    x_2d = mask[0, :]
    y_2d = mask[1, :]

    # Create z direction extrusion
    z_vals = np.linspace(extent[2][0], extent[2][1], 100)

    # Convert 1D shape to 2D grid for extrusion
    x, z = np.meshgrid(x_2d, z_vals)
    y, _ = np.meshgrid(y_2d, z_vals)

    # Plot the extruded surface
    ax.plot_surface(  # type: ignore[attr-defined]
        x, z, y, color=color, alpha=1.0, rstride=5, cstride=5, edgecolor="none"
    )

    ax.plot_surface(  # type: ignore[attr-defined]
        x[:1, :],
        np.full_like(x[:1, :], extent[2][0]),
        y[:1, :],
        color=color,
        alpha=1.0,
        edgecolor="none",
    )

    ax.plot_surface(  # type: ignore[attr-defined]
        x[-1:, :],
        np.full_like(x[-1:, :], extent[2][1]),
        y[-1:, :],
        color=color,
        alpha=1.0,
        edgecolor="none",
    )


def render_3d_iso(
    iso_field: np.ndarray,
    iso: float | list[float],
    color_range: tuple[float, float],
    output_path: Path | None = None,
    color_field: np.ndarray | None = None,
    colormap: str = "rainbow",
    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    figsize: tuple[int, int] = (10, 8),
    view_kwargs: dict | None = None,
    cylinder_kwargs: dict | None = None,
    airfoil_coords: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render a 3-D iso-surface plot of a given field.

    Parameters
    ----------
    field: ndarray
        velocity/vorticity/temperature field with shape (X, Y, Z).

    iso: float or list of float
        Iso-surface value(s) to plot.

    color_range: tuple[float, float]
        Min and max values for the color mapping.

    output_path: Path | None
        If provided, save the figure to this path. Defaults to None.

    color_field: ndarray | None
        Field used for coloring the iso-surface. Must have the same shape as `iso_field`
        if provided. Defaults to None.

    colormap: str
        Colormap to use for the color mapping. Defaults to "rainbow".

    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        The extent of the plot in (x, y, z) directions. Defaults to ((0.0, 1.0),
        (0.0, 1.0), (0.0, 1.0)).

    figsize: tuple[int, int]
        Size of the figure. Defaults to (10, 8).

    view_kwargs: dict | None
        Additional keyword arguments for setting the view angle. Defaults to None.

    cylinder_kwargs: dict | None
        If provided, a cylinder will be added to the plot with the given parameters.
        Defaults to None.

    airfoil_coords: ndarray | None
        If provided, an airfoil will be added to the plot with the given coordinates.
        Defaults to None.

    Returns
    -------
    ndarray
        The rendered figure as a numpy array.
    """
    if iso_field.ndim != 3:
        raise ValueError("Field must have shape (X, Y, Z).")

    if color_field is not None and iso_field.shape != color_field.shape:
        raise ValueError("`color_field` must have the same shape as `iso_field`.")

    if not isinstance(iso, list):
        iso = [iso]

    # Transpose y and z and flip them
    iso_field = np.transpose(iso_field, (0, 2, 1))
    if color_field is not None:
        color_field = np.transpose(color_field, (0, 2, 1))

    v_min, v_max = color_range
    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap(colormap)

    extent = (
        (extent[0][0], extent[0][1]),
        (extent[2][0], extent[2][1]),
        (extent[1][0], extent[1][1]),
    )

    spacing = (
        (extent[0][1] - extent[0][0]) / iso_field.shape[0],
        (extent[1][1] - extent[1][0]) / iso_field.shape[1],
        (extent[2][1] - extent[2][0]) / iso_field.shape[2],
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for iso_val in iso:
        verts, faces, _, _ = measure.marching_cubes(
            volume=iso_field,
            level=iso_val,
            spacing=spacing,
            step_size=1,
            allow_degenerate=True,
        )

        if color_field is None:
            rgba = list(cmap(iso_val))
            face_colors = tuple(rgba)
        else:
            x_coords = verts[:, 0] / spacing[0]
            y_coords = verts[:, 1] / spacing[1]
            z_coords = verts[:, 2] / spacing[2]

            vertex_colors = color_field[
                x_coords.astype(int),
                y_coords.astype(int),
                z_coords.astype(int),
            ]
            face_colors = cmap(norm(vertex_colors[faces].mean(axis=1)))

        verts[:, 0] += extent[0][0]
        verts[:, 1] += extent[1][0]
        verts[:, 2] += extent[2][0]

        mesh = Poly3DCollection(verts[faces], facecolors=face_colors)

        ax.add_collection3d(mesh)  # type: ignore

    if cylinder_kwargs is not None:
        _add_cylinder(ax, extent, **cylinder_kwargs)

    if airfoil_coords is not None:
        _add_airfoil(ax, extent, airfoil_coords)

    ax.invert_xaxis()
    ax.invert_yaxis()

    _format_3d(fig, ax)

    if view_kwargs is None:
        view_kwargs = {}

    ax.view_init(**{**DEFAULT_VIEW_KWARGS, **view_kwargs})  # type: ignore[attr-defined]

    ax.set_xlim(extent[0][1], extent[0][0])
    ax.set_ylim(extent[1][0], extent[1][1])
    ax.set_zlim(extent[2][0], extent[2][1])  # type: ignore[attr-defined]
    ax.set_box_aspect(
        (  # type: ignore[arg-type]
            (extent[0][1] - extent[0][0]),
            (extent[1][1] - extent[1][0]),
            (extent[2][1] - extent[2][0]),
        )
    )

    fig.subplots_adjust(left=-0.1, right=1.07, top=1.1, bottom=-0.1)

    if output_path is not None:
        plt.savefig(output_path, **_get_savefig_kwargs(output_path.name))

    buf = _fig_to_array(fig)

    plt.close()

    return buf


def _shade_faces(
    triangles: np.ndarray,
    face_colors: np.ndarray | tuple,
    view_kwargs: dict,
    ambient: float = 0.5,
    light_offset: tuple[float, float] = (25.0, -30.0),
) -> np.ndarray:
    """Lambert-shade triangle colors by the orientation of each triangle.

    Matplotlib draws a `Poly3DCollection` in flat colors, so a surface whose
    color field is constant on it reads as a silhouette. Weighting each face by
    how much it faces the light restores the relief.

    The light follows the camera, offset up and to the side by ``light_offset``,
    so the surface is lit from the viewer's shoulder for any view angle. Faces
    are lit on both sides (the diffuse term uses ``|n . l|``): an iso-surface
    encloses a volume, and its far side is seen from the inside.

    Parameters
    ----------
    triangles : np.ndarray
        Triangle vertices of shape [F, 3, 3], in the plot's axis order and in
        data coordinates.
    face_colors : np.ndarray | tuple
        Per-face RGBA of shape [F, 4], or a single RGBA tuple.
    view_kwargs : dict
        The ``elev``/``azim`` the plot is rendered with, in degrees.
    ambient : float
        Brightness of a face turned away from the light, relative to its
        unshaded color.
    light_offset : tuple[float, float]
        (elevation, azimuth) offset of the light from the camera, in degrees.

    Returns
    -------
    np.ndarray
        Shaded per-face RGBA of shape [F, 4].
    """
    colors = np.array(face_colors, dtype=float)
    if colors.ndim == 1:
        colors = np.tile(colors, (triangles.shape[0], 1))

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths == 0.0, 1.0, lengths)

    # The first two axes are drawn inverted (see the axis limits below), which
    # mirrors the scene: without the flip the light would come from behind
    normals = normals * np.array([-1.0, -1.0, 1.0])

    elev = np.radians(float(view_kwargs.get("elev", 0.0)) + light_offset[0])
    azim = np.radians(float(view_kwargs.get("azim", 0.0)) + light_offset[1])
    light = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )

    intensity = np.abs(normals @ light)
    colors[:, :3] *= (ambient + (1.0 - ambient) * intensity)[:, None]

    return np.clip(colors, 0.0, 1.0)


def render_3d_iso_rectilinear(
    iso_field: np.ndarray,
    iso: float | list[float],
    color_range: tuple[float, float],
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path | None = None,
    color_field: np.ndarray | None = None,
    colormap: str = "rainbow",
    figsize: tuple[int, int] = (10, 8),
    view_kwargs: dict | None = None,
    cylinder_kwargs: dict | None = None,
    airfoil_coords: np.ndarray | None = None,
    shade: bool = True,
    shade_ambient: float = 0.5,
) -> np.ndarray:
    """
    Render a 3-D iso-surface plot of a field given on its original grid.

    Same plot as :func:`render_3d_iso`, but the field is taken on the simulation
    grid instead of a uniform resampling of it: marching cubes runs in index
    space and the resulting vertices are mapped to physical positions by
    interpolating the per-axis cell-center coordinates. Graded grids therefore
    keep their near-wall resolution, which the uniform resampling of
    :func:`render_3d_iso` smears out.

    The grid has to be rectilinear (axis-aligned and separable), so that the
    cell centers are the tensor product of the three coordinate vectors in
    ``coords``. ``coords`` follows the axis order of ``iso_field``, as ``extent``
    does in :func:`render_3d_iso`.

    Parameters
    ----------
    iso_field: ndarray
        velocity/vorticity/temperature field with shape (X, Y, Z).

    iso: float or list of float
        Iso-surface value(s) to plot.

    color_range: tuple[float, float]
        Min and max values for the color mapping.

    coords: tuple[ndarray, ndarray, ndarray]
        Strictly increasing 1-D cell-center coordinates of the three axes of
        `iso_field`, with lengths matching its shape.

    output_path: Path | None
        If provided, save the figure to this path. Defaults to None.

    color_field: ndarray | None
        Field used for coloring the iso-surface. Must have the same shape as `iso_field`
        if provided. Defaults to None.

    colormap: str
        Colormap to use for the color mapping. Defaults to "rainbow".

    figsize: tuple[int, int]
        Size of the figure. Defaults to (10, 8).

    view_kwargs: dict | None
        Additional keyword arguments for setting the view angle. Defaults to None.

    cylinder_kwargs: dict | None
        If provided, a cylinder will be added to the plot with the given parameters.
        Defaults to None.

    airfoil_coords: ndarray | None
        If provided, an airfoil will be added to the plot with the given coordinates.
        Defaults to None.

    shade: bool
        Whether to shade the surface by its orientation, which is what gives it
        its 3D relief -- an iso-surface colored by a field that is constant on it
        is otherwise a flat silhouette. Defaults to True.

    shade_ambient: float
        Brightness of a face turned away from the light, as a fraction of its
        unshaded color. 1.0 is equivalent to `shade=False`, 0.0 puts such faces
        at black. Defaults to 0.5.

    Returns
    -------
    ndarray
        The rendered figure as a numpy array.
    """
    if iso_field.ndim != 3:
        raise ValueError("Field must have shape (X, Y, Z).")

    if color_field is not None and iso_field.shape != color_field.shape:
        raise ValueError("`color_field` must have the same shape as `iso_field`.")

    if len(coords) != 3:
        raise ValueError("`coords` must hold one coordinate vector per axis.")

    axis_coords = [np.asarray(c, dtype=np.float64).squeeze() for c in coords]
    for axis, c in enumerate(axis_coords):
        if c.ndim != 1 or c.shape[0] != iso_field.shape[axis]:
            raise ValueError(
                f"`coords[{axis}]` must be 1-D with {iso_field.shape[axis]} entries, "
                f"got shape {c.shape}."
            )
        if np.any(np.diff(c) <= 0):
            raise ValueError(f"`coords[{axis}]` must be strictly increasing.")

    if not isinstance(iso, list):
        iso = [iso]

    # Transpose y and z and flip them, as in render_3d_iso, so that the plot axes
    # are (x, z, y). The coordinates follow the same swap
    iso_field = np.transpose(iso_field, (0, 2, 1))
    if color_field is not None:
        color_field = np.transpose(color_field, (0, 2, 1))
    axis_coords = [axis_coords[0], axis_coords[2], axis_coords[1]]

    extent = (
        (float(axis_coords[0][0]), float(axis_coords[0][-1])),
        (float(axis_coords[1][0]), float(axis_coords[1][-1])),
        (float(axis_coords[2][0]), float(axis_coords[2][-1])),
    )

    v_min, v_max = color_range
    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap(colormap)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for iso_val in iso:
        # Unit spacing: the surface is extracted in index space and only then
        # mapped onto the (non-uniform) cell-center coordinates
        verts, faces, _, _ = measure.marching_cubes(
            volume=iso_field,
            level=iso_val,
            spacing=(1.0, 1.0, 1.0),
            step_size=1,
            allow_degenerate=True,
        )

        face_colors: Any
        if color_field is None:
            rgba = list(cmap(iso_val))
            rgba[3] = 1.0
            face_colors = tuple(rgba)
        else:
            # Trilinear, not nearest-cell: the vertices sit *between* cell
            # centers, exactly where the field crosses the iso value, so reading
            # the cell they fall into picks up values from off the surface. On a
            # surface of |u| = c that turns a field that is c everywhere on it
            # into the full range of the neighbouring cells
            vertex_colors = map_coordinates(
                color_field, verts.T, order=1, mode="nearest"
            )
            face_colors = cmap(norm(vertex_colors[faces].mean(axis=1)))
            face_colors[:, 3] = 1.0

        # Index -> physical position. The vertices sit between cell centers, so
        # the index axis is interpolated linearly, matching how marching cubes
        # placed them between the two sample values
        for axis, c in enumerate(axis_coords):
            verts[:, axis] = np.interp(verts[:, axis], np.arange(c.shape[0]), c)

        if shade:
            face_colors = _shade_faces(
                verts[faces],
                face_colors,
                view_kwargs={**DEFAULT_VIEW_KWARGS, **(view_kwargs or {})},
                ambient=shade_ambient,
            )

        mesh = Poly3DCollection(
            verts[faces],
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0.0,
            antialiaseds=False,
            alpha=1.0,
        )

        ax.add_collection3d(mesh)  # type: ignore

    if cylinder_kwargs is not None:
        _add_cylinder(ax, extent, **cylinder_kwargs)

    if airfoil_coords is not None:
        _add_airfoil(ax, extent, airfoil_coords)

    ax.invert_xaxis()
    ax.invert_yaxis()

    _format_3d(fig, ax)

    if view_kwargs is None:
        view_kwargs = {}

    ax.view_init(**{**DEFAULT_VIEW_KWARGS, **view_kwargs})  # type: ignore[attr-defined]

    ax.set_xlim(extent[0][1], extent[0][0])
    ax.set_ylim(extent[1][0], extent[1][1])
    ax.set_zlim(extent[2][0], extent[2][1])  # type: ignore[attr-defined]
    ax.set_box_aspect(
        (  # type: ignore[arg-type]
            (extent[0][1] - extent[0][0]),
            (extent[1][1] - extent[1][0]),
            (extent[2][1] - extent[2][0]),
        )
    )

    fig.subplots_adjust(left=-0.1, right=1.07, top=1.1, bottom=-0.1)

    if output_path is not None:
        plt.savefig(output_path, **_get_savefig_kwargs(output_path.name))

    buf = _fig_to_array(fig)

    plt.close()

    return buf


def render_3d_voxels(
    field: np.ndarray,
    ds: int,
    field_range: tuple[float, float],
    output_path: Path | None = None,
    colormap: str = "rainbow",
    figsize: tuple[int, int] = (10, 8),
    view_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Plot a 3D cube showing the three orthogonal sides (xy, xz, yz) of a 3D field in
    voxels, with the front faces (xz and yz) only showing the lower half in the
    z-direction.

    Parameters
    ----------
    field: ndarray
        Velocity/vorticity/temperature field with shape (X, Y, Z).

    ds: int
        Downsampling factor for faster rendering.

    field_range: tuple[float, float]
        Min and max values for the color mapping.

    output_path: Path | None
        If provided, save the figure to this path. Defaults to None.

    colormap: str
        Colormap to use for the color mapping. Defaults to "rainbow".

    figsize: tuple[int, int]
        Size of the figure. Defaults to (10, 8).

    view_kwargs: dict | None
        Additional keyword arguments for setting the view angle. Defaults to None.

    Returns
    -------
    ndarray
        The rendered figure as a numpy array.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    v_min, v_max = field_range
    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap(colormap)

    field = np.transpose(field, (0, 2, 1))

    # Downsample for faster rendering if too large
    field = field[::ds, ::ds, ::ds]

    # normalized values for color + alpha
    vals = norm(field)
    colors = cmap(vals)
    alpha = np.log1p(vals)
    alpha /= alpha.max()
    colors[..., 3] = alpha  # alpha ∝ value

    # show only nonzero voxels
    filled = vals > 0.0

    ax.voxels(  # type: ignore
        filled,
        facecolors=colors,
        edgecolor="none",
        shade=False,
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    _format_3d(fig, ax)

    # Set aspect ratio
    ax.set_box_aspect(field.shape)

    fig.subplots_adjust(left=-0.1, right=1.07, top=1.1, bottom=-0.1)

    # Set viewing angle
    ax.view_init(**{**DEFAULT_VIEW_KWARGS, **view_kwargs})  # type: ignore

    if output_path is not None:
        plt.savefig(output_path, **_get_savefig_kwargs(output_path.name))

    buf = _fig_to_array(fig)

    plt.close()

    return buf


class _SliceAxis(NamedTuple):
    """How a slice axis maps onto coordinates, data axes and the render grid."""

    component: int
    """Index of this axis in the (x, y, z) coordinates and in ``render_shape``."""

    spatial_axis: int
    """Index of this axis in the (Z, Y, X) spatial axes of the block data."""

    plane_components: tuple[int, int]
    """The (horizontal, vertical) coordinate components spanning the slice."""


_SLICE_AXES = {
    "x": _SliceAxis(component=0, spatial_axis=2, plane_components=(1, 2)),
    "y": _SliceAxis(component=1, spatial_axis=1, plane_components=(0, 2)),
    "z": _SliceAxis(component=2, spatial_axis=0, plane_components=(0, 1)),
}


def slice_plane_axes(axis: str | None) -> tuple[str, str]:
    """Return the (vertical, horizontal) axis names of a resampled plane.

    Parameters
    ----------
    axis : str | None
        The axis a 3D domain is sliced along, "x", "y" or "z", or None for a 2D
        domain, which is resampled in the x-y plane as a whole.

    Returns
    -------
    tuple[str, str]
        The names of the axes spanning the plane, in the order of the axes of
        the resampled array: ("z", "x") for a slice along "y", say.
    """
    if axis is None:
        return ("y", "x")

    if axis not in _SLICE_AXES:
        raise ValueError(f"Invalid axis {axis!r}, expected one of 'x', 'y', 'z'")

    names = ("x", "y", "z")
    horizontal, vertical = _SLICE_AXES[axis].plane_components

    return (names[vertical], names[horizontal])


def _infer_ndims(coords: torch.Tensor | np.ndarray) -> int:
    """Return the dimensionality of vertex coords shaped [D, ...] or [1, D, ...]."""
    shape = tuple(coords.shape)

    if len(shape) >= 3 and shape[0] in (2, 3) and len(shape) == shape[0] + 1:
        return int(shape[0])
    if len(shape) >= 4 and shape[1] in (2, 3) and len(shape) == shape[1] + 2:
        return int(shape[1])

    raise ValueError(f"Cannot infer dimensionality from vertex coords of shape {shape}")


def _as_coord_array(coords: torch.Tensor | np.ndarray, ndims: int) -> np.ndarray:
    """Return vertex coords as a numpy array [D, ...], dropping any batch axis."""
    if isinstance(coords, torch.Tensor):
        array = coords.detach().cpu().float().numpy()
    else:
        array = np.asarray(coords, dtype=np.float32)

    if array.ndim == ndims + 2:
        array = array[0]

    if array.ndim != ndims + 1 or array.shape[0] != ndims:
        raise ValueError(
            f"Expected vertex coords [{ndims}, ...] with {ndims} spatial axes, "
            f"got shape {tuple(array.shape)}"
        )

    return array


def _as_block_data(data: torch.Tensor | np.ndarray, ndims: int) -> np.ndarray:
    """Return block data as a numpy array [C, ...], dropping any batch axis."""
    if isinstance(data, torch.Tensor):
        array = data.detach().cpu().float().numpy()
    else:
        array = np.asarray(data, dtype=np.float32)

    if array.ndim == ndims + 2:
        array = array[0]
    elif array.ndim == ndims:
        array = array[None]

    if array.ndim != ndims + 1:
        raise ValueError(
            f"Expected block data [C, ...] with {ndims} spatial axes, "
            f"got shape {tuple(array.shape)}"
        )

    return array


def _block_boundary_polygon(vertex_coords: np.ndarray) -> np.ndarray:
    """Return the closed outer boundary of a 2D block as an (N, 2) point list.

    Parameters
    ----------
    vertex_coords : np.ndarray
        Vertex coordinates of shape [2, Y+1, X+1], dim-0 being (x, y).

    Returns
    -------
    np.ndarray
        The boundary vertices in order, shape (N, 2).
    """
    v = np.moveaxis(vertex_coords, 0, -1)  # [Y+1, X+1, 2]

    bottom = v[0, :, :]
    right = v[1:, -1, :]
    top = v[-1, -2::-1, :]
    left = v[-2:0:-1, 0, :]

    return np.concatenate([bottom, right, top, left], axis=0)


def _is_rectilinear(centers: np.ndarray, rtol: float = 1e-4) -> bool:
    """Whether cell centers form a rectilinear (axis-aligned, separable) grid.

    That is the case when every coordinate component varies only along its own
    spatial axis, which is what makes the cheap tensor-product interpolation
    valid.

    Parameters
    ----------
    centers : np.ndarray
        Cell centers of shape [D, Y, X] or [D, Z, Y, X], dim-0 being (x, y[, z]).
    rtol : float
        Tolerance relative to the domain extent. The default is loose enough to
        absorb the float32 round-off of the grid construction.

    Returns
    -------
    bool
        True if the grid is rectilinear.
    """
    ndims = centers.shape[0]
    scale = max(float(centers[c].max() - centers[c].min()) for c in range(ndims))
    if scale == 0.0:
        return False

    for component in range(ndims):
        # Component 0 (x) belongs to the last spatial axis, component 1 (y) to
        # the one before it, and so on
        own_axis = ndims - 1 - component
        other_axes = tuple(a for a in range(ndims) if a != own_axis)

        coords = centers[component]
        along_own_axis = np.expand_dims(coords.mean(axis=other_axes), other_axes)
        if np.abs(coords - along_own_axis).max() > rtol * scale:
            return False

    return True


class MultiblockResampler:
    """Resamples cell data of a multi-block domain onto a uniform render grid.

    The resampler holds the geometry of a full 2D or 3D domain, but resamples
    planes only: a 2D domain as a whole with ``__call__``, and a 3D domain one
    plane at a time with :meth:`extract_slice`. A slice of a 3D grid is itself a
    2D multi-block grid, so both run through the same code. Resampling a full 3D
    volume is deliberately not supported: the 3D triangulation does not scale to
    the size of the 3D grids here. Use the PICT sampler for full volumes.

    Which strategy fits is read off the grid and hidden from the caller:

    * A single rectilinear block is interpolated with a tensor-product linear
      interpolant, which needs no triangulation.
    * Anything else (several blocks, or a curvilinear block) is interpolated
      linearly over the Delaunay triangulation of the cell centers of all blocks
      at once. That is continuous across block interfaces and avoids the
      block-local artifacts of the PICT sampler.

    Everything that depends only on the grid is built once and reused, so build
    one resampler per domain and call it per field.

    Parameters
    ----------
    vertex_coord_list : Sequence[torch.Tensor]
        Per-block vertex coordinates, each [1, D, ...] or [D, ...] with dim-0
        being (x, y[, z]), as returned by ``domain.getVertexCoordinates()``.
    render_shape : Sequence[int]
        Output resolution, (nx, ny) in 2D or (nx, ny, nz) in 3D.
    bounds : Sequence[float], optional
        Target region as (x_min, x_max, y_min, y_max[, z_min, z_max]). Defaults
        to the bounding box of all block vertices.
    fill_value : float
        Value written where no block covers an output point.
    mask_uncovered : bool
        If True, output points outside every block are set to ``fill_value``.
        This is what keeps holes in the domain (the inside of a cylinder, say)
        from being filled in by the convex hull of the triangulation. Disable
        only if the domain is convex and gap-free.
    """

    def __init__(
        self,
        vertex_coord_list: Sequence[torch.Tensor] | Sequence[np.ndarray],
        render_shape: Sequence[int],
        bounds: Sequence[float] | None = None,
        fill_value: float = np.nan,
        mask_uncovered: bool = True,
    ):
        if len(vertex_coord_list) == 0:
            raise ValueError("Empty domain: nothing to resample")

        self._ndims = _infer_ndims(vertex_coord_list[0])
        if len(render_shape) != self._ndims:
            raise ValueError(
                f"render_shape {tuple(render_shape)} does not match the "
                f"{self._ndims}D domain"
            )

        self._vertex_coords = [
            _as_coord_array(coords, self._ndims) for coords in vertex_coord_list
        ]
        self._render_shape = tuple(int(n) for n in render_shape)
        self._fill_value = fill_value
        self._mask_uncovered = mask_uncovered
        self._bounds = None if bounds is None else tuple(float(b) for b in bounds)

        self._slice_cache: dict[tuple[str, int], MultiblockResampler] = {}
        self._slice_layers: dict[tuple[str, int], list[int]] = {}

        self._block_shapes = [
            tuple(n - 1 for n in coords.shape[1:]) for coords in self._vertex_coords
        ]
        self._axis_grids = self._make_axis_grids()

        # A 3D domain only ever resamples slices, and each slice builds its own
        # 2D resampler on demand, so there is nothing to prepare here for it
        if self._ndims == 2:
            self._build_plane(
                [
                    get_cell_centers(torch.from_numpy(coords)).numpy()
                    for coords in self._vertex_coords
                ]
            )

    @property
    def ndims(self) -> int:
        """The number of spatial dimensions of the domain."""
        return self._ndims

    @property
    def uncovered_mask(self) -> np.ndarray:
        """Output points of a 2D domain not covered by any block, shape (ny, nx).

        All False if the resampler was built with ``mask_uncovered=False``. For a
        3D domain, use :meth:`slice_uncovered_mask`.
        """
        if self._ndims != 2:
            raise ValueError(
                "uncovered_mask is only defined for a 2D domain; use "
                "slice_uncovered_mask(axis, index) for a 3D domain"
            )

        nx_out, ny_out = self._render_shape
        if self._uncovered is None:
            return np.zeros((ny_out, nx_out), dtype=bool)

        return self._uncovered.reshape(ny_out, nx_out)

    def slice_uncovered_mask(self, axis: str, index: int | None = None) -> np.ndarray:
        """Points of a slice not covered by any block. See :meth:`extract_slice`."""
        return self._slice_resampler(axis, index).uncovered_mask

    def axis_index(self, axis: str, position: float) -> int:
        """Return the render-grid index along ``axis`` closest to a position.

        Use this to slice at a physical position rather than at a grid index,
        without having to know how the render grid is laid out.

        Parameters
        ----------
        axis : str
            The axis to index along, "x", "y" or "z".
        position : float
            The position along ``axis``, in the coordinates of the domain.

        Returns
        -------
        int
            The index of the closest point of the render grid, for use as the
            ``index`` of :meth:`extract_slice`.
        """
        if axis not in _SLICE_AXES:
            raise ValueError(f"Invalid axis {axis!r}, expected one of 'x', 'y', 'z'")

        grid = self._axis_grids[_SLICE_AXES[axis].component]

        return int(np.abs(grid - position).argmin())

    def __call__(self, data_list: Sequence[torch.Tensor]) -> np.ndarray:
        """Resample per-block cell data of a 2D domain onto the uniform grid.

        Parameters
        ----------
        data_list : Sequence[torch.Tensor]
            Per-block cell data, each [1, C, Y, X], [C, Y, X] or [Y, X]. The
            number of channels must agree across blocks, and the spatial shapes
            must match the grid this resampler was built from.

        Returns
        -------
        np.ndarray
            Resampled data of shape (ny, nx) for single-channel data, else
            (C, ny, nx). Row 0 corresponds to the lower bound of y.
        """
        if self._ndims != 2:
            raise ValueError(
                "A 3D domain cannot be resampled as a whole; use "
                "extract_slice(data_list, axis, index) for a single plane, or "
                "the PICT sampler for a full volume"
            )

        values, channels = self._collect_values(data_list)

        if self._regular:
            out = np.stack(
                [
                    RegularGridInterpolator(
                        self._axes,
                        values[:, channel].reshape(self._block_shapes[0]),
                        method="linear",
                    )(self._query_regular)
                    for channel in range(channels)
                ],
                axis=-1,
            )
        else:
            out = LinearNDInterpolator(self._triangulation, values)(self._query)
            out[self._nearest_targets] = values[self._nearest_sources]

        if self._uncovered is not None:
            out[self._uncovered] = self._fill_value

        nx_out, ny_out = self._render_shape
        out = np.moveaxis(out.reshape(ny_out, nx_out, channels), -1, 0)

        return out[0] if channels == 1 else out

    def extract_slice(
        self,
        data_list: Sequence[torch.Tensor],
        axis: str,
        index: int | None = None,
    ) -> np.ndarray:
        """Resample a single plane of a 3D domain onto the uniform render grid.

        The plane is the one at ``index`` along ``axis`` of the render grid. Of
        each block, the cell layer closest to that plane is taken and resampled
        as a 2D grid, so a slice costs no more than resampling a 2D domain.

        Parameters
        ----------
        data_list : Sequence[torch.Tensor]
            Per-block cell data, each [1, C, Z, Y, X], [C, Z, Y, X] or [Z, Y, X].
        axis : str
            The axis to slice along, "x", "y" or "z".
        index : int, optional
            Index of the plane along ``axis`` of the render grid. Defaults to the
            middle of that axis.

        Returns
        -------
        np.ndarray
            The plane, of shape (n_vertical, n_horizontal) for single-channel
            data, else (C, n_vertical, n_horizontal): (ny, nx) for a slice along
            "z", (nz, nx) along "y", and (nz, ny) along "x".
        """
        if self._ndims != 3:
            raise ValueError(
                "extract_slice is only defined for a 3D domain; a 2D domain is "
                "resampled as a whole by calling the resampler"
            )

        resampler = self._slice_resampler(axis, index)
        spatial_axis = _SLICE_AXES[axis].spatial_axis
        layers = self._slice_layers[(axis, self._slice_index(axis, index))]

        sliced = [
            np.take(_as_block_data(data, self._ndims), layer, axis=spatial_axis + 1)
            for data, layer in zip(data_list, layers, strict=True)
        ]

        return resampler(sliced)

    def _make_axis_grids(self) -> list[np.ndarray]:
        """Return the uniform output coordinates along each axis, (x, y[, z])."""
        if self._bounds is None:
            lower = [
                min(float(coords[c].min()) for coords in self._vertex_coords)
                for c in range(self._ndims)
            ]
            upper = [
                max(float(coords[c].max()) for coords in self._vertex_coords)
                for c in range(self._ndims)
            ]
        else:
            lower = list(self._bounds[0::2])
            upper = list(self._bounds[1::2])

        return [
            np.linspace(lower[c], upper[c], self._render_shape[c])
            for c in range(self._ndims)
        ]

    def _build_plane(self, centers: list[np.ndarray]) -> None:
        """Build the interpolation machinery for this 2D grid."""
        nx_out, ny_out = self._render_shape
        gx, gy = np.meshgrid(self._axis_grids[0], self._axis_grids[1], indexing="xy")
        self._query = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # [ny*nx, 2]
        self._uncovered: np.ndarray | None = None

        self._regular = len(centers) == 1 and _is_rectilinear(centers[0])
        if self._regular:
            # A single rectilinear block: interpolate along each axis separately,
            # no triangulation needed. The outermost output points sit half a cell
            # outside the outermost cell centers, so clamp the query to the cell
            # centers to get the edge value there instead of an extrapolation
            self._axes = [
                centers[0][1].mean(axis=1),  # y, varies along spatial axis 0
                centers[0][0].mean(axis=0),  # x, varies along spatial axis 1
            ]
            self._query_regular = np.stack(
                [
                    np.clip(self._query[:, 1], self._axes[0][0], self._axes[0][-1]),
                    np.clip(self._query[:, 0], self._axes[1][0], self._axes[1][-1]),
                ],
                axis=-1,
            )
            return

        points = np.concatenate([c.reshape(2, -1).T for c in centers], axis=0)

        # The triangulation is the expensive part and depends only on the grid
        self._triangulation = Delaunay(points)

        # Cell centers sit half a cell inside the block border, so the outermost
        # band of output points falls outside the hull of the triangulation and
        # gets no linear value. Those points are still inside the domain, so fall
        # back to the nearest cell center there. Which points these are is fixed
        # by the geometry, so resolve them once
        outside_hull = self._triangulation.find_simplex(self._query) < 0
        self._nearest_targets = np.flatnonzero(outside_hull)
        _, nearest_sources = cKDTree(points).query(self._query[self._nearest_targets])
        self._nearest_sources = np.atleast_1d(nearest_sources)

        if not self._mask_uncovered:
            return

        # Query points exactly on a block edge are ambiguous for contains_points,
        # which would spuriously mask the domain border. Grow each polygon by a
        # fraction of a cell; the sign of `radius` depends on the winding order,
        # so take the union of both signs to grow regardless of orientation
        eps = 1e-3 * min(
            (self._axis_grids[0][-1] - self._axis_grids[0][0]) / nx_out,
            (self._axis_grids[1][-1] - self._axis_grids[1][0]) / ny_out,
        )
        covered = np.zeros(self._query.shape[0], dtype=bool)
        for coords in self._vertex_coords:
            path = MplPath(_block_boundary_polygon(coords))
            covered |= path.contains_points(self._query, radius=eps)
            covered |= path.contains_points(self._query, radius=-eps)

        self._uncovered = ~covered

    def _slice_index(self, axis: str, index: int | None) -> int:
        """Validate and default the render-grid index of a slice."""
        if axis not in _SLICE_AXES:
            raise ValueError(f"Invalid axis {axis!r}, expected one of 'x', 'y', 'z'")

        n_out = self._render_shape[_SLICE_AXES[axis].component]
        if index is None:
            return n_out // 2
        if not -n_out <= index < n_out:
            raise IndexError(
                f"Slice index {index} is out of range for axis {axis!r} of the "
                f"render grid with {n_out} points"
            )

        return index % n_out

    def _slice_resampler(self, axis: str, index: int | None) -> "MultiblockResampler":
        """Return the (cached) 2D resampler for the cross-section of a slice."""
        key = (axis, self._slice_index(axis, index))
        if key in self._slice_cache:
            return self._slice_cache[key]

        axis_spec = _SLICE_AXES[axis]
        position = self._axis_grids[axis_spec.component][key[1]]

        cross_coords = []
        layers = []
        for coords in self._vertex_coords:
            # Take the cell layer whose center is closest to the plane
            vertices = np.moveaxis(
                coords[axis_spec.component], axis_spec.spatial_axis, 0
            )
            vertices = vertices.reshape(vertices.shape[0], -1).mean(axis=-1)
            cell_centers = 0.5 * (vertices[:-1] + vertices[1:])
            layer = int(np.abs(cell_centers - position).argmin())
            layers.append(layer)

            # The cross-section of that cell layer is the mid-surface between its
            # two vertex layers, which is itself a 2D block
            lower = np.take(coords, layer, axis=axis_spec.spatial_axis + 1)
            upper = np.take(coords, layer + 1, axis=axis_spec.spatial_axis + 1)
            cross_coords.append(0.5 * (lower + upper)[list(axis_spec.plane_components)])

        plane_bounds = None
        if self._bounds is not None:
            plane_bounds = tuple(
                self._bounds[2 * c + side]
                for c in axis_spec.plane_components
                for side in (0, 1)
            )

        resampler = MultiblockResampler(
            vertex_coord_list=cross_coords,
            render_shape=[self._render_shape[c] for c in axis_spec.plane_components],
            bounds=plane_bounds,
            fill_value=self._fill_value,
            mask_uncovered=self._mask_uncovered,
        )

        self._slice_cache[key] = resampler
        self._slice_layers[key] = layers

        return resampler

    def _collect_values(
        self, data_list: Sequence[torch.Tensor]
    ) -> tuple[np.ndarray, int]:
        """Return per-block data as one [n_cells, C] array plus the channel count."""
        if len(data_list) != len(self._block_shapes):
            raise ValueError(
                f"Got {len(data_list)} data blocks, but this resampler was built "
                f"for {len(self._block_shapes)} blocks"
            )

        values = []
        channels: int | None = None
        for block_idx, (data, block_shape) in enumerate(
            zip(data_list, self._block_shapes, strict=True)
        ):
            block_data = _as_block_data(data, self._ndims)
            if tuple(block_data.shape[1:]) != block_shape:
                raise ValueError(
                    f"Block {block_idx}: data spatial shape "
                    f"{tuple(block_data.shape[1:])} does not match cell-center "
                    f"shape {block_shape}"
                )
            if channels is None:
                channels = block_data.shape[0]
            elif block_data.shape[0] != channels:
                raise ValueError(
                    f"Block {block_idx}: has {block_data.shape[0]} channels, "
                    f"expected {channels}"
                )
            values.append(block_data.reshape(channels, -1).T)

        assert channels is not None

        return np.concatenate(values, axis=0), channels


def resample(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    data: np.ndarray,
    render_shape: tuple[int, int],
) -> np.ndarray:
    """Resample 2D cell-centered data onto a uniform grid of ``render_shape``."""
    if data.shape != (len(x_centers), len(y_centers)):
        raise ValueError(
            f"Data shape {data.shape} does not match x_centers {len(x_centers)} "
            f"and y_centers {len(y_centers)}"
        )

    fn = RegularGridInterpolator((x_centers, y_centers), data, method="linear")

    # Create a uniform meshgrid for the target resolution
    ux = np.linspace(x_centers.min(), x_centers.max(), render_shape[0])
    uy = np.linspace(y_centers.min(), y_centers.max(), render_shape[1])

    # Generate the 3D grid points for evaluation
    pts = np.meshgrid(ux, uy, indexing="ij")
    pts_flat = np.array([p.flatten() for p in pts]).T

    # Resample
    uniform_data = fn(pts_flat).reshape(render_shape)

    return uniform_data


def resample_3d(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    data: np.ndarray,
    render_shape: tuple[int, ...],
) -> np.ndarray:
    """Resample 3D cell-centered data onto a uniform grid of ``render_shape``."""
    if data.shape != (len(x_centers), len(y_centers), len(z_centers)):
        raise ValueError(
            f"Data shape {data.shape} does not match x_centers {len(x_centers)}, "
            f"y_centers {len(y_centers)}, z_centers {len(z_centers)}"
        )

    fn = RegularGridInterpolator(
        (x_centers, y_centers, z_centers), data, method="linear"
    )

    ux = np.linspace(x_centers.min(), x_centers.max(), render_shape[0])
    uy = np.linspace(y_centers.min(), y_centers.max(), render_shape[1])
    uz = np.linspace(z_centers.min(), z_centers.max(), render_shape[2])

    pts = np.meshgrid(ux, uy, uz, indexing="ij")
    pts_flat = np.array([p.flatten() for p in pts]).T

    return fn(pts_flat).reshape(render_shape)
