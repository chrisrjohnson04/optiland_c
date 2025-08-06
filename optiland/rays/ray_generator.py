"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays


class RayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """Generates rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays: RealRays object containing the generated rays.

        """
        field_count = be.size(Hx)
        pupil_count = be.size(Px)
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vx_broadcasted = be.reshape(vx, (field_count, 1)) 
        vy = 1 - be.array(vyf)
        vy_broadcasted = be.reshape(vy, (field_count, 1)) 
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx_broadcasted, vy_broadcasted, field_count, pupil_count)

        if self.optic.obj_space_telecentric:
            if self.optic.field_type == "angle":
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x_broadcasted = be.broadcast_to(Px, (field_count, pupil_count)) 
            x1 = be.reshape(vx_broadcasted * x_broadcasted, (-1,))
            x1 = x1 * EPD * 0.5
            # x1 = Px * EPD * vx / 2 (before chris j changes)
            y_broadcasted = be.broadcast_to(Py, (field_count, pupil_count)) 
            y1 = be.reshape(vy_broadcasted * y_broadcasted, (-1,))
            y1 = y1 * EPD * 0.5
            # y1 = Py * EPD * vy / 2 ( before chris j changes)
            z1 = be.full_like(x1, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        apodization = self.optic.apodization
        if apodization:
            x_tiled = be.tile(Px, field_count)
            y_tiled = be.tile(Py, field_count)
            intensity = apodization.get_intensity(x_tiled, y_tiled)
        else:
            intensity = be.ones_like(x1)

        wavelength = be.ones_like(x1) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx_broadcasted, vy_broadcasted, field_count, pupil_count):
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float or be.ndarray): Normalized x field coordinate.
            Hy (float or be.ndarray): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (be.ndarray): Vignetting factor in the x-direction.
            vy (be.ndarray): Vignetting factor in the y-direction.
            field_count (float): Number of fields
            pupil_count (float): Number of pupils

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.

        """
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )
            if self.optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity.",
                )
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            offset = self._get_starting_z_offset()

            # x, y, z positions of ray starting points
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            x = be.repeat(x, pupil_count)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            y = be.repeat(y, pupil_count)
            z = self.optic.surface_group.positions[1] - offset

            x_broadcasted = be.broadcast_to(Px, (field_count, pupil_count)) 
            
            x0 = be.reshape(vx_broadcasted * x_broadcasted, (-1,))
            x0 = x0 * (EPD / 2) + x
            # x0 = Px * EPD / 2 * vx + x (before changes by chris j)
            y_broadcasted = be.broadcast_to(Py, (field_count, pupil_count)) 
            y0 = be.reshape(vy_broadcasted * y_broadcasted, (-1,))
            y0 = y0 * (EPD / 2) + y
            # y0 = Py * EPD / 2 * vy + y (before changes by chris j)
            z0 = be.full_like(x0, z)
        else:
            if self.optic.field_type == "object_height":
                x0 = be.array(field_x)
                x0 = be.repeat(x0, pupil_count)
                y0 = be.array(field_y)
                y0 = be.repeat(y0, pupil_count)
                z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z

            elif self.optic.field_type == "angle":
                EPL = self.optic.paraxial.EPL()
                z0 = self.optic.surface_group.positions[0]
                x0 = -be.tan(be.radians(field_x)) * (EPL - z0)
                x0 = be.repeat(x0, pupil_count)
                y0 = -be.tan(be.radians(field_y)) * (EPL - z0)
                y0 = be.repeat(y0, pupil_count)

            if be.size(x0) == 1:
                x0 = be.repeat(x0, pupil_count*field_count)
            if be.size(y0) == 1:
                y0 = be.repeat(y0, pupil_count*field_count)
            if be.size(z0) == 1:
                z0 = be.repeat(z0, pupil_count*field_count)

        return x0, y0, z0

    def _get_starting_z_offset(self):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        z = self.optic.surface_group.positions[1:-1]
        offset = self.optic.paraxial.EPD()
        return offset - be.min(z)
