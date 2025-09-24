from vstools import vs
from ..expr import infix2postfix
from enum import IntEnum


class ZoomMode(IntEnum):
    """
    Controls the zoom behavior of `rotate_image` function

    Attributes:
        NO_ZOOM: Rotate the clip without any scaling; missing border pixels will be interpolated
        ZOOM_TO_FIT: Dynamically scale the clip to ensure the entire image fits within the frame, thus avoids additional interpolation on the borders
        CONSTANT_MAX_ZOOM: Use a constant scaling factor based on a 45° rotation to prevent dynamic changes during rotation
    """

    NO_ZOOM = 0
    ZOOM_TO_FIT = 1
    CONSTANT_MAX_ZOOM = 2


def rotate_image(
    clip: vs.VideoNode,
    angle_degrees: str,
    zoom: ZoomMode = ZoomMode.NO_ZOOM,
    bicubic_b: str = "1 / 3",
    bicubic_c: str = "1 / 3",
    center_x: str = "$width / 2",
    center_y: str = "$height / 2",
) -> vs.VideoNode:
    """
    Rotate a video clip

    This function rotates the given video clip by an angle specified in degrees. The rotation is
    performed around a defined center point and uses bicubic interpolation for resampling.

    Args:
        clip: The source video clip to rotate.
        angle_degrees: The rotation angle in degrees, can be an expression to evaluate in run-time.
        zoom: The zoom mode to apply.
        bicubic_b: The 'b' parameter for bicubic interpolation.
        bicubic_c: The 'c' parameter for bicubic interpolation.
        center_x: The x-coordinate of the rotation center, can be an expression to evaluate in run-time
        center_y: The y-coordinate of the rotation center, can be an expression to evaluate in run-time

    Returns:
        The rotated video clip.
    """
    setup = f"""
        param_b  = {bicubic_b}
        param_c = {bicubic_c}
        <global<param_b><param_c>>
        function bicubic_weight(in) {{
            ax = abs(in)
            ax2 = ax ** 2
            ax3 = ax ** 3

            term3_p1 = (12 - 9 * param_b - 6 * param_c) * ax3
            term2_p1 = (-18 + 12 * param_b + 6 * param_c) * ax2
            term0_p1 = (6 - 2 * param_b)
            part1 = (term3_p1 + term2_p1 + term0_p1) / 6
            
            term3_p2 = (-param_b - 6 * param_c) * ax3
            term2_p2 = (6 * param_b + 30 * param_c) * ax2
            term1_p2 = (-12 * param_b - 48 * param_c) * ax
            term0_p2 = (8 * param_b + 24 * param_c)
            part2 = (term3_p2 + term2_p2 + term1_p2 + term0_p2) / 6

            result = (ax < 1) ? part1 : (ax < 2) ? part2 : 0

            return result
        }}
        
        angle_degrees = {angle_degrees}
        angle_rad = angle_degrees * $pi / 180
        center_x = {center_x}
        center_y = {center_y}
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)

        out_rel_x = $X - center_x
        out_rel_y = $Y - center_y
    """

    if zoom == ZoomMode.CONSTANT_MAX_ZOOM:
        source_coord_calc = """
            qq = 0.7071067811865476
            w_bound_const = $width * qq + $height * qq
            h_bound_const = $width * qq + $height * qq
            shrink_scale = min($width / max(w_bound_const, 1e-9), $height / max(h_bound_const, 1e-9))

            src_rel_x_unscaled = out_rel_x * cos_a + out_rel_y * sin_a
            src_rel_y_unscaled = -out_rel_x * sin_a + out_rel_y * cos_a

            final_src_rel_x = src_rel_x_unscaled * shrink_scale
            final_src_rel_y = src_rel_y_unscaled * shrink_scale

            source_x = final_src_rel_x + center_x
            source_y = final_src_rel_y + center_y
        """
    elif zoom == ZoomMode.ZOOM_TO_FIT:
        source_coord_calc = """
            ca = abs(cos_a)
            sa = abs(sin_a)
            w_bound = $width * ca + $height * sa
            h_bound = $width * sa + $height * ca
            shrink_scale_x = $width / max(w_bound, 1e-9)
            shrink_scale_y = $height / max(h_bound, 1e-9)
            shrink_scale = min(shrink_scale_x, shrink_scale_y)

            src_rel_x_unscaled = out_rel_x * cos_a + out_rel_y * sin_a
            src_rel_y_unscaled = -out_rel_x * sin_a + out_rel_y * cos_a

            final_src_rel_x = src_rel_x_unscaled * shrink_scale
            final_src_rel_y = src_rel_y_unscaled * shrink_scale

            source_x = final_src_rel_x + center_x
            source_y = final_src_rel_y + center_y
        """
    else:
        source_coord_calc = """
            final_src_rel_x = out_rel_x * cos_a + out_rel_y * sin_a
            final_src_rel_y = -out_rel_x * sin_a + out_rel_y * cos_a

            source_x = final_src_rel_x + center_x
            source_y = final_src_rel_y + center_y
        """

    interpolate = """
        ix = floor(source_x)
        iy = floor(source_y)
        fx = source_x - ix
        fy = source_y - iy

        x_m1 = ix - 1
        x0 = ix 
        x1 = ix + 1 
        x2 = ix + 2
        y_m1 = iy - 1 
        y0 = iy 
        y1 = iy + 1 
        y2 = iy + 2

        wx_m1 = bicubic_weight(fx + 1) 
        wx0 = bicubic_weight(fx)
        wx1 = bicubic_weight(fx - 1) 
        wx2 = bicubic_weight(fx - 2)

        wy_m1 = bicubic_weight(fy + 1) 
        wy0 = bicubic_weight(fy)
        wy1 = bicubic_weight(fy - 1) 
        wy2 = bicubic_weight(fy - 2)

        p_m1_m1 = dyn($src0, x_m1, y_m1) 
        p0_m1 = dyn($src0, x0, y_m1)
        p1_m1 = dyn($src0, x1, y_m1)   
        p2_m1 = dyn($src0, x2, y_m1)
        p_m1_0 = dyn($src0, x_m1, y0)  
        p0_0 = dyn($src0, x0, y0)
        p1_0 = dyn($src0, x1, y0)    
        p2_0 = dyn($src0, x2, y0)
        p_m1_1 = dyn($src0, x_m1, y1)  
        p0_1 = dyn($src0, x0, y1)
        p1_1 = dyn($src0, x1, y1)    
        p2_1 = dyn($src0, x2, y1)
        p_m1_2 = dyn($src0, x_m1, y2)  
        p0_2 = dyn($src0, x0, y2)
        p1_2 = dyn($src0, x1, y2)    
        p2_2 = dyn($src0, x2, y2)

        row_m1 = p_m1_m1 * wx_m1 + p0_m1 * wx0 + p1_m1 * wx1 + p2_m1 * wx2
        row_0  = p_m1_0 * wx_m1 + p0_0 * wx0 + p1_0 * wx1 + p2_0 * wx2
        row_1  = p_m1_1 * wx_m1 + p0_1 * wx0 + p1_1 * wx1 + p2_1 * wx2
        row_2  = p_m1_2 * wx_m1 + p0_2 * wx0 + p1_2 * wx1 + p2_2 * wx2

        RESULT = row_m1 * wy_m1 + row_0 * wy0 + row_1 * wy1 + row_2 * wy2
    """

    full_expr_str = setup + source_coord_calc + interpolate

    expr_postfix = infix2postfix(full_expr_str)
    return clip.akarin.Expr(expr_postfix)
