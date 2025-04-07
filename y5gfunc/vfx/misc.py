from vstools import vs
from ..expr import infix2postfix


def rotate_image(
    clip: vs.VideoNode,
    angle_degrees: str,
    bicubic_b: str = "1 / 3",
    bicubic_c: str = "1 / 3",
    center_x: str = "width / 2",
    center_y: str = "height / 2"
) -> vs.VideoNode:

    expr = infix2postfix(f'''
            angle_degrees = {angle_degrees}
            param_b  = {bicubic_b}
            param_c = {bicubic_c}

            <global<param_b><param_c>>
            function bicubic_weight(in) {{
                ax = abs(in)
                ax2 = ax * ax
                ax3 = ax2 * ax

                term3_p1 = (12 - 9 * param_b - 6 * param_c) * ax3
                term2_p1 = (-18 + 12 * param_b + 6 * param_c) * ax2
                term0_p1 = (6 - 2 * param_b)
                part1 = (term3_p1 + term2_p1 + term0_p1) / 6

                term3_p2 = (-param_b - 6 * param_c) * ax3
                term2_p2 = (6 * param_b + 30 * param_c) * ax2
                term1_p2 = (-12 * param_b - 48 * param_c) * ax
                term0_p2 = (8 * param_b + 24 * param_c)
                part2 = (term3_p2 + term2_p2 + term1_p2 + term0_p2) / 6

                result = (ax < 2) ? part2 : 0
                result = (ax < 1) ? part1 : result

                return result
            }}

            angle_rad = angle_degrees * pi / 180
            center_x = {center_x}
            center_y = {center_y}
            cos_angle = cos(angle_rad)
            sin_angle = sin(angle_rad)

            x_relative = X - center_x
            y_relative = Y - center_y
            source_x_relative = x_relative * cos_angle + y_relative * sin_angle
            source_y_relative = -x_relative * sin_angle + y_relative * cos_angle
            source_x = source_x_relative + center_x
            source_y = source_y_relative + center_y

            ix = floor(source_x)
            iy = floor(source_y)
            fx = source_x - ix
            fy = source_y - iy

            x_m1 = ix - 1
            x0   = ix
            x1   = ix + 1
            x2   = ix + 2

            y_m1 = iy - 1
            y0   = iy
            y1   = iy + 1
            y2   = iy + 2

            wx_m1 = bicubic_weight(fx + 1)
            wx0   = bicubic_weight(fx)
            wx1   = bicubic_weight(fx - 1)
            wx2   = bicubic_weight(fx - 2)

            wy_m1 = bicubic_weight(fy + 1)
            wy0   = bicubic_weight(fy)
            wy1   = bicubic_weight(fy - 1)
            wy2   = bicubic_weight(fy - 2)

            p_m1_m1 = dyn(src0, x_m1, y_m1)
            p0_m1   = dyn(src0, x0,   y_m1)
            p1_m1   = dyn(src0, x1,   y_m1)
            p2_m1   = dyn(src0, x2,   y_m1)
            p_m1_0  = dyn(src0, x_m1, y0)
            p0_0    = dyn(src0, x0,   y0)
            p1_0    = dyn(src0, x1,   y0)
            p2_0    = dyn(src0, x2,   y0)
            p_m1_1  = dyn(src0, x_m1, y1)
            p0_1    = dyn(src0, x0,   y1)
            p1_1    = dyn(src0, x1,   y1)
            p2_1    = dyn(src0, x2,   y1)
            p_m1_2  = dyn(src0, x_m1, y2)
            p0_2    = dyn(src0, x0,   y2)
            p1_2    = dyn(src0, x1,   y2)
            p2_2    = dyn(src0, x2,   y2)

            interp_row_m1 = p_m1_m1 * wx_m1 + p0_m1 * wx0 + p1_m1 * wx1 + p2_m1 * wx2
            interp_row_0  = p_m1_0  * wx_m1 + p0_0  * wx0 + p1_0  * wx1 + p2_0  * wx2
            interp_row_1  = p_m1_1  * wx_m1 + p0_1  * wx0 + p1_1  * wx1 + p2_1  * wx2
            interp_row_2  = p_m1_2  * wx_m1 + p0_2  * wx0 + p1_2  * wx1 + p2_2  * wx2

            RESULT = interp_row_m1 * wy_m1 + interp_row_0 * wy0 + interp_row_1 * wy1 + interp_row_2 * wy2
            ''')

    return clip.akarin.Expr(expr)
