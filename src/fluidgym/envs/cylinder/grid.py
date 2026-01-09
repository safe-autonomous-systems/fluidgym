"""Grid generation for the cylinder vortex street environment."""

import logging

import numpy as np
import torch

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym.envs.util.profiles import get_inflow_profile
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.util.output import plot_grids

logger = logging.getLogger("env.cylinder.grid")


def make_vortex_street_domain(
    ndims: int,
    viscosity: torch.Tensor,
    domain_height: float,
    domain_length: float,
    cylinder_radius: float,
    cylinder_offset_y: float,
    circle_thickness: float,
    quad_thickness_x: float,
    circle_resolution_angular: int,
    vortex_street_refinement_base: float,
    vortex_street_refinement_axes: list[str],
    cuda_device: torch.device,
    dtype: torch.dtype = torch.float32,
    debug: bool = False,
) -> PISOtorch.Domain:
    """Create a PISOtorch Domain for the cylinder vortex street environment.

    Parameters
    ----------
    ndims: int
        Number of spatial dimensions (2 or 3).

    viscosity: torch.Tensor
        Kinematic viscosity of the fluid.

    domain_height: float
        Height of the simulation domain.

    domain_length: float
        Length of the simulation domain.

    cylinder_radius: float
        Radius of the cylinder.

    cylinder_offset_y: float
        Vertical offset of the cylinder center from the domain centerline.

    circle_thickness: float
        Thickness of the circular grid around the cylinder.

    quad_thickness_x: float
        Thickness of the quadrilateral grid in the x-direction.

    circle_resolution_angular: int
        Number of grid cells in the angular direction of the circular grid.

    vortex_street_refinement_base: float
        Base for exponential grid refinement in the vortex street region.

    vortex_street_refinement_axes: list[str]
        Axes along which to apply wall refinement in the vortex street region.

    cuda_device: torch.device
        The CUDA device to use for computations.

    dtype: torch.dtype
        Data type for tensors. Defaults to torch.float32.

    debug: bool
        If True, plot the generated grids for debugging purposes. Defaults to False.

    Returns
    -------
    PISOtorch.Domain
        The created PISOtorch Domain for the simulation.
    """
    res_z = circle_resolution_angular

    quad_thickness_y = quad_thickness_x + cylinder_offset_y
    actual_domain_height = (
        2 * cylinder_radius + 2 * circle_thickness + 2 * quad_thickness_y
    )

    x_min = -(cylinder_radius + circle_thickness + quad_thickness_x)
    x_max = domain_length + x_min

    if domain_height != actual_domain_height:
        raise ValueError(
            f"domain_height ({domain_height}) does not match the calculated height "
            f"({actual_domain_height}) from cylinder_radius, circle_thickness and "
            f"quad_thickness_y"
        )

    # Inner Circle
    t_r1 = cylinder_radius
    t_r2 = t_r1 + circle_thickness

    t_r1 = cylinder_radius
    t_r2 = t_r1 + circle_thickness
    circle_top_coords = shapes.make_torus_2D(
        circle_resolution_angular,
        r1=t_r1,
        r2=t_r2,
        start_angle=135,
        angle=-90,
        dtype=dtype,
    )  # y up, x right
    # circle_top_coords = torch.flip(circle_top_coords, (-2,)) # y down, x right
    circle_right_coords = shapes.make_torus_2D(
        circle_resolution_angular,
        r1=t_r1,
        r2=t_r2,
        start_angle=45,
        angle=-90,
        dtype=dtype,
    )  # y right, x down
    circle_right_coords = torch.movedim(circle_right_coords, -1, -2)  # y down, x right
    circle_right_coords = torch.flip(circle_right_coords, (-2,))  # y up, x right

    circle_bot_coords = shapes.make_torus_2D(
        circle_resolution_angular,
        r1=t_r1,
        r2=t_r2,
        start_angle=-45,
        angle=-90,
        dtype=dtype,
    )  # y down, x left
    circle_bot_coords = torch.flip(circle_bot_coords, (-2, -1))  # y up, x right
    circle_left_coords = shapes.make_torus_2D(
        circle_resolution_angular,
        r1=t_r1,
        r2=t_r2,
        start_angle=-135,
        angle=-90,
        dtype=dtype,
    )  # y left, x up
    circle_left_coords = torch.movedim(circle_left_coords, -1, -2)  # y up, x left
    circle_left_coords = torch.flip(circle_left_coords, (-1,))  # y up, x right

    # Quad
    quad_r_outer_x = cylinder_radius + circle_thickness + quad_thickness_x
    quad_r_outer_y = cylinder_radius + circle_thickness + quad_thickness_y
    quad_r_outer_y_top = quad_r_outer_y + cylinder_offset_y
    quad_r_outer_y_bot = quad_r_outer_y - cylinder_offset_y

    quad_r_inner = np.sin(np.deg2rad(45)) * t_r2

    circle_resolution_radial = circle_top_coords.size(-2) - 1

    quad_res_angular = circle_resolution_angular + 1
    quad_res_radial = int(
        np.ceil(quad_thickness_y / circle_thickness * circle_resolution_radial)
    )

    quad_corners_top = [
        (-quad_r_inner, quad_r_inner),
        (quad_r_inner, quad_r_inner),
        (-quad_r_outer_x, quad_r_outer_y_top),
        (quad_r_outer_x, quad_r_outer_y_top),
    ]
    quad_corners_right = [
        (quad_r_inner, -quad_r_inner),
        (quad_r_outer_x, -quad_r_outer_y_bot),
        (quad_r_inner, quad_r_inner),
        (quad_r_outer_x, quad_r_outer_y_top),
    ]
    quad_corners_bot = [
        (-quad_r_outer_x, -quad_r_outer_y_bot),
        (quad_r_outer_x, -quad_r_outer_y_bot),
        (-quad_r_inner, -quad_r_inner),
        (quad_r_inner, -quad_r_inner),
    ]
    quad_corners_left = [
        (-quad_r_outer_x, -quad_r_outer_y_bot),
        (-quad_r_inner, -quad_r_inner),
        (-quad_r_outer_x, quad_r_outer_y_top),
        (-quad_r_inner, quad_r_inner),
    ]

    # borders, specify round circle borders, rest is linear interpolated from corners
    def make_border(t):
        return torch.movedim(t, 0, 1).cpu().clone().numpy().tolist()

    quad_border_top = [None, None, make_border(circle_top_coords[0, :, -1, :]), None]
    quad_border_right = [
        make_border(circle_right_coords[0, :, :, -1]),
        None,
        None,
        None,
    ]
    quad_border_bot = [None, None, None, make_border(circle_bot_coords[0, :, 0, :])]
    quad_border_left = [None, make_border(circle_left_coords[0, :, :, 0]), None, None]

    quad_coords_top = shapes.generate_grid_vertices_2D(
        [quad_res_radial, quad_res_angular],
        quad_corners_top,
        quad_border_top,
        dtype=torch.float32,
    )
    quad_coords_bot = shapes.generate_grid_vertices_2D(
        [quad_res_radial, quad_res_angular],
        quad_corners_bot,
        quad_border_bot,
        dtype=torch.float32,
    )

    x_weights = shapes.make_weights_exp(
        res=quad_res_angular - 1, base=vortex_street_refinement_base, refinement="BOTH"
    )

    quad_coords_right = shapes.generate_grid_vertices_2D(
        res=[quad_res_angular, quad_res_radial],
        corner_vertices=quad_corners_right,
        border_vertices=quad_border_right,
        x_weights=x_weights,
        dtype=torch.float32,
    )

    quad_coords_left = shapes.generate_grid_vertices_2D(
        res=[quad_res_angular, quad_res_radial],
        corner_vertices=quad_corners_left,
        border_vertices=quad_border_left,
        dtype=torch.float32,
    )

    cylinder_left_coords = torch.cat(
        [quad_coords_left[:, :, :, :-1], circle_left_coords], dim=-1
    )

    cylinder_top_coords = torch.cat(
        [
            circle_top_coords[:, :, :-1, :],
            quad_coords_top,
        ],
        dim=-2,
    )

    cylinder_right_coords = torch.cat(
        [
            circle_right_coords[:, :, :, :-1],
            quad_coords_right,
        ],
        dim=-1,
    )

    cylinder_bottom_coords = torch.cat(
        [quad_coords_bot[:, :, :-1, :], circle_bot_coords], dim=-2
    )

    vortex_street_resolution_x = int(quad_res_radial / quad_thickness_y * 18)
    vortex_street_cords = shapes.make_wall_refined_ortho_grid(
        vortex_street_resolution_x,
        circle_resolution_angular,
        corner_lower=(-1 * x_min, -quad_r_outer_y_bot),
        corner_upper=(x_max, quad_r_outer_y_top),
        wall_refinement=vortex_street_refinement_axes,
        base=vortex_street_refinement_base,
    )

    if debug:
        plot_grids(
            [
                cylinder_left_coords,
                cylinder_bottom_coords,
                cylinder_top_coords,
                cylinder_right_coords,
                vortex_street_cords,
                vortex_street_cords,
            ],
            color=["r", "b", "y", "g", "r", "k"],  # type: ignore
            path=".",
            type="pdf",
        )

    grids = [
        cylinder_left_coords,
        cylinder_bottom_coords,
        cylinder_top_coords,
        cylinder_right_coords,
        vortex_street_cords,
    ]

    if ndims == 3:
        [
            cylinder_left_coords,
            cylinder_bottom_coords,
            cylinder_top_coords,
            cylinder_right_coords,
            vortex_street_cords,
        ] = [shapes.extrude_grid_z(g, res_z=res_z, start_z=-2, end_z=2) for g in grids]

    domain = PISOtorch.Domain(
        ndims,
        viscosity,
        name="CylinderDomain",
        device=cuda_device,
        dtype=dtype,
        passiveScalarChannels=0,
    )

    inflow = get_inflow_profile(
        h=domain_height - 2 * cylinder_offset_y,
        res_y=circle_resolution_angular,
        n_dims=ndims,
        dtype=dtype,
        device=cuda_device,
        res_z=res_z if ndims == 3 else None,
    )

    cylinder_left_block = domain.CreateBlock(
        vertexCoordinates=cylinder_left_coords.to(cuda_device),
        name="BlockCylinderLeft",
    )
    cylinder_left_block.CloseBoundary("-x")  # Inflow
    cylinder_left_block.getBoundary("-x").setVelocity(inflow)
    cylinder_left_block.CloseBoundary("+x")  # Cylinder

    cylinder_top_block = domain.CreateBlock(
        vertexCoordinates=cylinder_top_coords.to(cuda_device),
        name="BlockCylinderTop",
    )
    cylinder_top_block.CloseBoundary("+y")  # Wall
    cylinder_top_block.CloseBoundary("-y")  # Cylinder

    cylinder_right_block = domain.CreateBlock(
        vertexCoordinates=cylinder_right_coords.to(cuda_device),
        name="BlockCylinderRight",
    )
    cylinder_right_block.CloseBoundary("-x")  # Cylinder

    cylinder_bottom_block = domain.CreateBlock(
        vertexCoordinates=cylinder_bottom_coords.to(cuda_device),
        name="BlockCylinderBottom",
    )
    cylinder_bottom_block.CloseBoundary("-y")  # Wall
    cylinder_bottom_block.CloseBoundary("+y")  # Cylinder

    vortex_street_block = domain.CreateBlock(
        vertexCoordinates=vortex_street_cords.to(cuda_device),
        name="BlockVortexStreet",
    )
    vortex_street_block.CloseBoundary("+y")  # Wall
    vortex_street_block.CloseBoundary("-y")  # Wall
    vortex_street_block.CloseBoundary("+x")

    vortex_street_block.getBoundary("+x").setVelocity(inflow)  # Outflow
    vortex_street_block.getBoundary("+x").makeVelocityVarying()

    if ndims == 3:
        cylinder_left_block.MakePeriodic("z")
        cylinder_top_block.MakePeriodic("z")
        cylinder_right_block.MakePeriodic("z")
        cylinder_bottom_block.MakePeriodic("z")
        vortex_street_block.MakePeriodic("z")

    # Connect blocks
    # for all blocks/grids: y up, x right
    # --------------------------------------------------------------------------------
    # directional face specification: (-x,+x,-y,+y,-z,+z) <-> [0,5]
    # => axis := face/2 ((x,y,z) <-> [0,2])
    # => direction := face%2, 0 is lower/negative side, 1 is upper/positive side
    #   for 'axisIndex' the direction indicates if the connection is inverted (0 for
    # same direction, 1 for inverted)
    #
    # Using block.ConnectBlock(faceIndex, otherBlock, otherFaceIndex, axis1Index,
    # axis2Index), faceIndex of block is connected to otherFaceIndex of otherBlock.
    # For 2D and 3D, the remaining axes are also mapped:
    #   faceIndex connects to otherFaceIndex
    #   axis[(faceIndex / 2 + 1)%ndims] of block is aligned to axis1Index of otherBlock.
    # The connection is inverted if axis1Index%2==1.
    #   axis[(faceIndex / 2 + 2)%ndims] of block is aligned to axis2Index of otherBlock.
    # The connection is inverted if axis2Index%2==1.
    # --------------------------------------------------------------------------------
    if ndims == 2:
        # The last argument (axis2Index) can be omitted for 2D
        # face +y(3) of cylinder_left_block connects to face -x(0) of cylinder_top_block
        # the next axis of cylinder_left_block, x (0=(3//2 + 1)%2), is aligned with the
        # y-axis of cylinder_top_block, +y means the axis direction is inverted
        # axis2Index -x is ignored (omitted below)
        cylinder_left_block.ConnectBlock("+y", cylinder_top_block, "-x", "+y")

        cylinder_left_block.ConnectBlock("-y", cylinder_bottom_block, "-x", "-y")

        cylinder_right_block.ConnectBlock("+y", cylinder_top_block, "+x", "-y")
        cylinder_right_block.ConnectBlock("-y", cylinder_bottom_block, "+x", "+y")

        cylinder_right_block.ConnectBlock("+x", vortex_street_block, "-x", "-y")
    else:
        # face +y(3) of cylinder_left_block connects to face -x(0) of cylinder_top_block
        # (as before). The next axis of cylinder_left_block, z (2=(3//2 + 1)%3), is
        # aligned with the z-axis of cylinder_top_block, -z means the axis direction is
        # not inverted. The next axis of cylinder_left_block, x (0=(3//2 + 2)%3), is
        # aligned with the y-axis of cylinder_top_block, +y means the axis direction is
        # inverted (as before, but specified after the z-z connection/alignment)
        cylinder_left_block.ConnectBlock("+y", cylinder_top_block, "-x", "-z", "+y")

        cylinder_left_block.ConnectBlock("-y", cylinder_bottom_block, "-x", "-z", "-y")

        cylinder_right_block.ConnectBlock("+y", cylinder_top_block, "+x", "-z", "-y")
        cylinder_right_block.ConnectBlock("-y", cylinder_bottom_block, "+x", "-z", "+y")

        # face +x(1) of cylinder_right_block connects to face -x(0) of
        # vortex_street_block. The next axis of cylinder_right_block,
        # y (1=(1//2 + 1)%3), is aligned with the y-axis of cylinder_top_block, -y means
        # the axis direction is not inverted. The next axis of cylinder_right_block,
        # z (2=(1//2 + 2)%3), is aligned with the z-axis of cylinder_top_block, -z means
        # the axis direction is not inverted
        cylinder_right_block.ConnectBlock("+x", vortex_street_block, "-x", "-y", "-z")

    return domain
