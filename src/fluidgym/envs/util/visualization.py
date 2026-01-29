"""Visualization tools for FluidGym environments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from skimage import measure

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
    y = np.linspace(0, extent[1][1], 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, y = np.meshgrid(theta, y)

    x = radius * np.cos(theta) + center_x

    # Note: In the 3D plot, z is vertical axis,
    # but in general convention, y is vertical axis.
    z = radius * np.sin(theta) + center_y

    # Cylinder surface
    ax.plot_surface(  # type: ignore
        x, y, z, color=color, alpha=1.0, rstride=5, cstride=5, edgecolor="none"
    )

    theta_face = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, radius, 50)
    r, theta_face = np.meshgrid(r, theta_face)

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
    cmap = plt.cm.get_cmap(colormap)

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
            volume=np.abs(iso_field),
            level=iso_val,
            spacing=spacing,
            step_size=1,
            allow_degenerate=True,
        )

        if color_field is None:
            face_colors = cmap(iso_val)
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

        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.7,
        )
        mesh.set_facecolor(face_colors)

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
