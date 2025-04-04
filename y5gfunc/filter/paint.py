import math
from vstools import core
from vstools import vs
from ..expr import infix2postfix
import sys


def draw_line(clip: vs.VideoNode, sx: str, sy: str, ex: str, ey: str, thickness: str, color: str, factor: str = "1.0") -> vs.VideoNode: 
    assert clip.format.num_planes == 1
    
    expr = infix2postfix(f'''
            sx = {sx}
            sy = {sy}
            ex = {ex}
            ey = {ey}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            dx = ex - sx
            dy = ey - sy
            L2 = (ex - sx) * dx + dy * dy
            half_thickness = thickness / 2
            half_thickness_sq = half_thickness * half_thickness
            tt = ((X - sx) * dx + (Y - sy) * dy) / L2
            tt = clamp(tt, 0, 1)
            proj_x = sx + tt * dx
            proj_y = sy + tt * dy
            d2 = (X - proj_x) * (X - proj_x) + (Y - proj_y) * (Y - proj_y)
            do = d2 <= half_thickness_sq
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            ''')
    
    return clip.akarin.Expr(expr)

def draw_circle(clip: vs.VideoNode, cx: str, cy: str, radius: str, thickness: str, color: str, factor: str = "1.0") -> vs.VideoNode:
    assert clip.format.num_planes == 1

    expr = infix2postfix(f'''
            cx = {cx}
            cy = {cy}
            radius = {radius}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            half_thickness = thickness / 2
            dx = X - cx
            dy = Y - cy
            distance_sq = dx * dx + dy * dy
            radius_minus_half = radius - half_thickness
            lower_sq = radius_minus_half * radius_minus_half
            lower_sq = max(lower_sq, 0)
            upper_sq = (radius + half_thickness) * (radius + half_thickness)
            do = distance_sq >= lower_sq && distance_sq <= upper_sq
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            ''')

    return clip.akarin.Expr(expr)

def draw_ellipse(clip: vs.VideoNode, f1x: str, f1y: str, f2x: str, f2y: str, ellipse_sum: str, thickness: str, color: str, factor: str = "1.0") -> vs.VideoNode:
    assert clip.format.num_planes == 1
    
    expr = infix2postfix(f'''
            f1x = {f1x}
            f1y = {f1y}
            f2x = {f2x}
            f2y = {f2y}
            ellipse_sum = {ellipse_sum}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            cx = (f1x + f2x) / 2
            cy = (f1y + f2y) / 2
            aa = ellipse_sum / 2
            a2 = aa * aa
            dx = f2x - f1x
            dy = f2y - f1y
            c2 = (dx * dx + dy * dy) / 4
            b2 = a2 - c2
            value = ((X - cx) * (X - cx)) / a2 + ((Y - cy) * (Y - cy)) / b2
            norm_thresh = thickness / ellipse_sum
            do = abs(value - 1) <= norm_thresh
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            ''')
    
    return clip.akarin.Expr(expr)

def draw_bezier_curve(
    clip: vs.VideoNode,
    controlPoint0X: str,
    controlPoint0Y: str,
    controlPoint1X: str,
    controlPoint1Y: str,
    controlPoint2X: str,
    controlPoint2Y: str,
    controlPoint3X: str,
    controlPoint3Y: str,
    thickness: str,
    color: str,
    sample_count: int = 100,
    factor: str = "1.0"
) -> vs.VideoNode:

    assert sample_count >= 2
    assert clip.format.num_planes == 1

    expr_lines = []

    expr_lines.append(f"controlPoint0X = {controlPoint0X}")
    expr_lines.append(f"controlPoint0Y = {controlPoint0Y}")
    expr_lines.append(f"controlPoint1X = {controlPoint1X}")
    expr_lines.append(f"controlPoint1Y = {controlPoint1Y}")
    expr_lines.append(f"controlPoint2X = {controlPoint2X}")
    expr_lines.append(f"controlPoint2Y = {controlPoint2Y}")
    expr_lines.append(f"controlPoint3X = {controlPoint3X}")
    expr_lines.append(f"controlPoint3Y = {controlPoint3Y}")
    expr_lines.append(f"thickness = {thickness}")
    expr_lines.append(f"color = {color}")
    expr_lines.append(f"factor = {factor}")
    expr_lines.append("halfThickness = thickness / 2.0")
    expr_lines.append("halfThicknessSquared = halfThickness * halfThickness")
    
    total_samples = sample_count
    for sample_index in range(total_samples):
        t_value = sample_index / (total_samples - 1)
        expr_lines.append(f"parameterT_{sample_index} = {t_value}")
        expr_lines.append(f"oneMinusT_{sample_index} = 1 - parameterT_{sample_index}")
        expr_lines.append(f"oneMinusT_squared_{sample_index} = oneMinusT_{sample_index} * oneMinusT_{sample_index}")
        expr_lines.append(f"oneMinusT_cubed_{sample_index} = oneMinusT_squared_{sample_index} * oneMinusT_{sample_index}")
        expr_lines.append(f"parameterT_squared_{sample_index} = parameterT_{sample_index} * parameterT_{sample_index}")
        expr_lines.append(f"parameterT_cubed_{sample_index} = parameterT_squared_{sample_index} * parameterT_{sample_index}")
        expr_lines.append(
            f"bezierX_{sample_index} = oneMinusT_cubed_{sample_index} * controlPoint0X + "
            f"3 * oneMinusT_squared_{sample_index} * parameterT_{sample_index} * controlPoint1X + "
            f"3 * oneMinusT_{sample_index} * parameterT_squared_{sample_index} * controlPoint2X + "
            f"parameterT_cubed_{sample_index} * controlPoint3X"
        )
        expr_lines.append(
            f"bezierY_{sample_index} = oneMinusT_cubed_{sample_index} * controlPoint0Y + "
            f"3 * oneMinusT_squared_{sample_index} * parameterT_{sample_index} * controlPoint1Y + "
            f"3 * oneMinusT_{sample_index} * parameterT_squared_{sample_index} * controlPoint2Y + "
            f"parameterT_cubed_{sample_index} * controlPoint3Y"
        )
    
    segment_distance_squared_expressions = []
    for seg_index in range(total_samples - 1):
        expr_lines.append(f"deltaX_{seg_index} = bezierX_{seg_index+1} - bezierX_{seg_index}")
        expr_lines.append(f"deltaY_{seg_index} = bezierY_{seg_index+1} - bezierY_{seg_index}")
        expr_lines.append(f"segmentLengthSquared_{seg_index} = deltaX_{seg_index} * deltaX_{seg_index} + deltaY_{seg_index} * deltaY_{seg_index}")
        expr_lines.append(f"tSegment_{seg_index} = ((X - bezierX_{seg_index}) * deltaX_{seg_index} + (Y - bezierY_{seg_index}) * deltaY_{seg_index}) / segmentLengthSquared_{seg_index}")
        expr_lines.append(f"tClamped_{seg_index} = clamp(tSegment_{seg_index}, 0, 1)")
        expr_lines.append(f"projectionX_{seg_index} = bezierX_{seg_index} + tClamped_{seg_index} * deltaX_{seg_index}")
        expr_lines.append(f"projectionY_{seg_index} = bezierY_{seg_index} + tClamped_{seg_index} * deltaY_{seg_index}" )
        expr_lines.append(f"distanceSquared_{seg_index} = (X - projectionX_{seg_index}) * (X - projectionX_{seg_index}) + (Y - projectionY_{seg_index}) * (Y - projectionY_{seg_index})")
        segment_distance_squared_expressions.append(f"distanceSquared_{seg_index}")
    
    distance_arguments = ", ".join(segment_distance_squared_expressions)
    expr_lines.append(f"finalMinDistanceSquared = nth_1({distance_arguments})")
    
    expr_lines.append("doDraw = finalMinDistanceSquared <= halfThicknessSquared")
    expr_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")
    
    full_expression = "\n".join(expr_lines)
    
    converted_expression = infix2postfix(full_expression)
    return clip.akarin.Expr(converted_expression)

def draw_mandelbrot_zoomer(
    clip: vs.VideoNode,
    centerX: str,
    centerY: str,
    color: str,
    initialZoom: str = "0.005",
    zoomSpeed: str = "0.01",
    centerMoveSpeed: str = "0.01",
    fractalC0_re: str = "-0.75",
    fractalC0_im: str = "0.1",
    fractalC1_re: str = "-0.743643887037158704",
    fractalC1_im: str = "0.131825904205311130",
    maxIter: int = 350,
    escapeRadius: str = "2.0"
) -> vs.VideoNode:
    
    sys.setrecursionlimit(3000)

    assert maxIter >= 1
    
    expr_lines = []
    
    expr_lines.append(f"centerX = {centerX}")
    expr_lines.append(f"centerY = {centerY}")
    expr_lines.append(f"color = {color}")
    expr_lines.append(f"initialZoom = {initialZoom}")
    expr_lines.append(f"zoomSpeed = {zoomSpeed}")
    expr_lines.append(f"centerMoveSpeed = {centerMoveSpeed}")
    expr_lines.append(f"maxIter = {maxIter}")
    expr_lines.append(f"escapeRadius = {escapeRadius}")
    expr_lines.append("escapeSq = escapeRadius ** 2")
    
    expr_lines.append("scale = initialZoom * exp(-zoomSpeed * N)")
    
    expr_lines.append("ff = 1 - exp(-centerMoveSpeed * N)")
    expr_lines.append(f"C0_re = {fractalC0_re}")
    expr_lines.append(f"C0_im = {fractalC0_im}")
    expr_lines.append(f"C1_re = {fractalC1_re}")
    expr_lines.append(f"C1_im = {fractalC1_im}")
    expr_lines.append("dC_re = C0_re + (C1_re - C0_re) * ff")
    expr_lines.append("dC_im = C0_im + (C1_im - C0_im) * ff")
    
    expr_lines.append("c_re = (X - centerX) * scale + dC_re")
    expr_lines.append("c_im = (Y - centerY) * scale + dC_im")
    
    expr_lines.append("z_re_0 = 0")
    expr_lines.append("z_im_0 = 0")
    for i in range(1, maxIter + 1):
        prev = i - 1
        expr_lines.append(f"z_re_{i} = (z_re_{prev} * z_re_{prev} - z_im_{prev} * z_im_{prev} + c_re)")
        expr_lines.append(f"z_im_{i} = (2 * z_re_{prev} * z_im_{prev} + c_im)")
        expr_lines.append(f"r2_{i} = (z_re_{i} * z_re_{i} + z_im_{i} * z_im_{i})")
    
    iter_expr = str(maxIter)
    for i in range(maxIter, 0, -1):
        iter_expr = f"(r2_{i} > escapeSq ? {i} : {iter_expr})"
    expr_lines.append("iterResult = " + iter_expr)
    
    expr_lines.append("RESULT = (iterResult / maxIter) * color")
    
    full_expr = "\n".join(expr_lines)
    converted_expr = infix2postfix(full_expr)
    return clip.akarin.Expr(converted_expr)

def draw_spiral(
    clip: vs.VideoNode,
    centerX: str,
    centerY: str,
    a: str,
    b: str,
    startAngle: str,
    endAngle: str,
    thickness: str,
    color: str,
    factor: str = "1.0",
    sample_count = 500
) -> vs.VideoNode:

    assert clip.format.num_planes == 1
    
    expr_lines = []
    expr_lines.append(f"centerX = {centerX}")
    expr_lines.append(f"centerY = {centerY}")
    expr_lines.append(f"aa = {a}")
    expr_lines.append(f"bb = {b}")
    expr_lines.append(f"startAngle = {startAngle}")
    expr_lines.append(f"endAngle = {endAngle}")
    expr_lines.append(f"thickness = {thickness}")
    expr_lines.append(f"color = {color}")
    expr_lines.append(f"factor = {factor}")
    expr_lines.append("halfThickness = thickness / 2.0")
    expr_lines.append("halfThicknessSquared = halfThickness ** 2")
    
    for i in range(sample_count):
        expr_lines.append(f"tt = {i} / ({sample_count} - 1)")
        expr_lines.append("theta = startAngle + tt * (endAngle - startAngle)")
        expr_lines.append("rr = aa + bb * theta")
        expr_lines.append("spiralX = centerX + rr * cos(theta)")
        expr_lines.append("spiralY = centerY + rr * sin(theta)")
        expr_lines.append(f"spiralX_{i} = spiralX")
        expr_lines.append(f"spiralY_{i} = spiralY")
    
    segment_distance_exprs = []
    for i in range(sample_count - 1):
        expr_lines.append(f"deltaX_{i} = spiralX_{i+1} - spiralX_{i}")
        expr_lines.append(f"deltaY_{i} = spiralY_{i+1} - spiralY_{i}")
        expr_lines.append(f"segmentLengthSquared_{i} = deltaX_{i} * deltaX_{i} + deltaY_{i} * deltaY_{i}")
        expr_lines.append(f"t_{i} = ((X - spiralX_{i}) * deltaX_{i} + (Y - spiralY_{i}) * deltaY_{i}) / segmentLengthSquared_{i}")
        expr_lines.append(f"tClamped_{i} = clamp(t_{i}, 0, 1)")
        expr_lines.append(f"projectionX_{i} = spiralX_{i} + tClamped_{i} * deltaX_{i}")
        expr_lines.append(f"projectionY_{i} = spiralY_{i} + tClamped_{i} * deltaY_{i}")
        expr_lines.append(f"distanceSquared_{i} = (X - projectionX_{i}) ** 2 + (Y - projectionY_{i}) ** 2")
        segment_distance_exprs.append(f"distanceSquared_{i}")
    
    expr_lines.append("finalMinDistanceSquared = nth_1(" + ", ".join(segment_distance_exprs) + ")")
    expr_lines.append("doDraw = finalMinDistanceSquared <= halfThicknessSquared")
    expr_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")
    
    full_expr = "\n".join(expr_lines)
    converted_expr = infix2postfix(full_expr)
    return clip.akarin.Expr(converted_expr)

def draw_3d_cube(
    clip: vs.VideoNode,
    centerX: str,
    centerY: str,
    cubeSize: str,
    color: str,
    rotationX: str,
    rotationY: str,
    thickness: str,
    translateZ: str = "500",
    focal: str = "500",
    factor: str = "1.0"
) -> vs.VideoNode:
    assert clip.format.num_planes == 1

    expr = infix2postfix(f'''
            centerX = {centerX}
            centerY = {centerY}
            cubeSize = {cubeSize}
            rotationX = {rotationX}
            rotationY = {rotationY}
            translateZ = {translateZ}
            focal = {focal}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            half = cubeSize / 2.0

            v0x0 = -half
            v0y0 = -half
            v0z0 = -half
            v0y1 = v0y0 * cos(rotationX) - v0z0 * sin(rotationX)
            v0z1 = v0y0 * sin(rotationX) + v0z0 * cos(rotationX)
            v0x1 = v0x0
            v0x = v0x1 * cos(rotationY) + v0z1 * sin(rotationY)
            v0z = -v0x1 * sin(rotationY) + v0z1 * cos(rotationY)
            v0y = v0y1
            v0z_final = v0z + translateZ
            v0projX = centerX + (v0x * focal) / v0z_final
            v0projY = centerY + (v0y * focal) / v0z_final

            v1x0 = half
            v1y0 = -half
            v1z0 = -half
            v1y1 = v1y0 * cos(rotationX) - v1z0 * sin(rotationX)
            v1z1 = v1y0 * sin(rotationX) + v1z0 * cos(rotationX)
            v1x1 = v1x0
            v1x = v1x1 * cos(rotationY) + v1z1 * sin(rotationY)
            v1z = -v1x1 * sin(rotationY) + v1z1 * cos(rotationY)
            v1y = v1y1
            v1z_final = v1z + translateZ
            v1projX = centerX + (v1x * focal) / v1z_final
            v1projY = centerY + (v1y * focal) / v1z_final

            v2x0 = half
            v2y0 = half
            v2z0 = -half
            v2y1 = v2y0 * cos(rotationX) - v2z0 * sin(rotationX)
            v2z1 = v2y0 * sin(rotationX) + v2z0 * cos(rotationX)
            v2x1 = v2x0
            v2x = v2x1 * cos(rotationY) + v2z1 * sin(rotationY)
            v2z = -v2x1 * sin(rotationY) + v2z1 * cos(rotationY)
            v2y = v2y1
            v2z_final = v2z + translateZ
            v2projX = centerX + (v2x * focal) / v2z_final
            v2projY = centerY + (v2y * focal) / v2z_final

            v3x0 = -half
            v3y0 = half
            v3z0 = -half
            v3y1 = v3y0 * cos(rotationX) - v3z0 * sin(rotationX)
            v3z1 = v3y0 * sin(rotationX) + v3z0 * cos(rotationX)
            v3x1 = v3x0
            v3x = v3x1 * cos(rotationY) + v3z1 * sin(rotationY)
            v3z = -v3x1 * sin(rotationY) + v3z1 * cos(rotationY)
            v3y = v3y1
            v3z_final = v3z + translateZ
            v3projX = centerX + (v3x * focal) / v3z_final
            v3projY = centerY + (v3y * focal) / v3z_final

            v4x0 = -half
            v4y0 = -half
            v4z0 = half
            v4y1 = v4y0 * cos(rotationX) - v4z0 * sin(rotationX)
            v4z1 = v4y0 * sin(rotationX) + v4z0 * cos(rotationX)
            v4x1 = v4x0
            v4x = v4x1 * cos(rotationY) + v4z1 * sin(rotationY)
            v4z = -v4x1 * sin(rotationY) + v4z1 * cos(rotationY)
            v4y = v4y1
            v4z_final = v4z + translateZ
            v4projX = centerX + (v4x * focal) / v4z_final
            v4projY = centerY + (v4y * focal) / v4z_final

            v5x0 = half
            v5y0 = -half
            v5z0 = half
            v5y1 = v5y0 * cos(rotationX) - v5z0 * sin(rotationX)
            v5z1 = v5y0 * sin(rotationX) + v5z0 * cos(rotationX)
            v5x1 = v5x0
            v5x = v5x1 * cos(rotationY) + v5z1 * sin(rotationY)
            v5z = -v5x1 * sin(rotationY) + v5z1 * cos(rotationY)
            v5y = v5y1
            v5z_final = v5z + translateZ
            v5projX = centerX + (v5x * focal) / v5z_final
            v5projY = centerY + (v5y * focal) / v5z_final

            v6x0 = half
            v6y0 = half
            v6z0 = half
            v6y1 = v6y0 * cos(rotationX) - v6z0 * sin(rotationX)
            v6z1 = v6y0 * sin(rotationX) + v6z0 * cos(rotationX)
            v6x1 = v6x0
            v6x = v6x1 * cos(rotationY) + v6z1 * sin(rotationY)
            v6z = -v6x1 * sin(rotationY) + v6z1 * cos(rotationY)
            v6y = v6y1
            v6z_final = v6z + translateZ
            v6projX = centerX + (v6x * focal) / v6z_final
            v6projY = centerY + (v6y * focal) / v6z_final

            v7x0 = -half
            v7y0 = half
            v7z0 = half
            v7y1 = v7y0 * cos(rotationX) - v7z0 * sin(rotationX)
            v7z1 = v7y0 * sin(rotationX) + v7z0 * cos(rotationX)
            v7x1 = v7x0
            v7x = v7x1 * cos(rotationY) + v7z1 * sin(rotationY)
            v7z = -v7x1 * sin(rotationY) + v7z1 * cos(rotationY)
            v7y = v7y1
            v7z_final = v7z + translateZ
            v7projX = centerX + (v7x * focal) / v7z_final
            v7projY = centerY + (v7y * focal) / v7z_final

            deltaX_0 = v1projX - v0projX
            deltaY_0 = v1projY - v0projY
            segLenSq_0 = deltaX_0 * deltaX_0 + deltaY_0 * deltaY_0
            t0 = ((X - v0projX) * deltaX_0 + (Y - v0projY) * deltaY_0) / segLenSq_0
            t0_clamped = clamp(t0, 0, 1)
            projX_0 = v0projX + t0_clamped * deltaX_0
            projY_0 = v0projY + t0_clamped * deltaY_0
            distanceSq_0 = (X - projX_0) * (X - projX_0) + (Y - projY_0) * (Y - projY_0)

            deltaX_1 = v2projX - v1projX
            deltaY_1 = v2projY - v1projY
            segLenSq_1 = deltaX_1 * deltaX_1 + deltaY_1 * deltaY_1
            t1 = ((X - v1projX) * deltaX_1 + (Y - v1projY) * deltaY_1) / segLenSq_1
            t1_clamped = clamp(t1, 0, 1)
            projX_1 = v1projX + t1_clamped * deltaX_1
            projY_1 = v1projY + t1_clamped * deltaY_1
            distanceSq_1 = (X - projX_1) * (X - projX_1) + (Y - projY_1) * (Y - projY_1)

            deltaX_2 = v3projX - v2projX
            deltaY_2 = v3projY - v2projY
            segLenSq_2 = deltaX_2 * deltaX_2 + deltaY_2 * deltaY_2
            t2 = ((X - v2projX) * deltaX_2 + (Y - v2projY) * deltaY_2) / segLenSq_2
            t2_clamped = clamp(t2, 0, 1)
            projX_2 = v2projX + t2_clamped * deltaX_2
            projY_2 = v2projY + t2_clamped * deltaY_2
            distanceSq_2 = (X - projX_2) * (X - projX_2) + (Y - projY_2) * (Y - projY_2)

            deltaX_3 = v0projX - v3projX
            deltaY_3 = v0projY - v3projY
            segLenSq_3 = deltaX_3 * deltaX_3 + deltaY_3 * deltaY_3
            t3 = ((X - v3projX) * deltaX_3 + (Y - v3projY) * deltaY_3) / segLenSq_3
            t3_clamped = clamp(t3, 0, 1)
            projX_3 = v3projX + t3_clamped * deltaX_3
            projY_3 = v3projY + t3_clamped * deltaY_3
            distanceSq_3 = (X - projX_3) * (X - projX_3) + (Y - projY_3) * (Y - projY_3)

            deltaX_4 = v5projX - v4projX
            deltaY_4 = v5projY - v4projY
            segLenSq_4 = deltaX_4 * deltaX_4 + deltaY_4 * deltaY_4
            t4 = ((X - v4projX) * deltaX_4 + (Y - v4projY) * deltaY_4) / segLenSq_4
            t4_clamped = clamp(t4, 0, 1)
            projX_4 = v4projX + t4_clamped * deltaX_4
            projY_4 = v4projY + t4_clamped * deltaY_4
            distanceSq_4 = (X - projX_4) * (X - projX_4) + (Y - projY_4) * (Y - projY_4)

            deltaX_5 = v6projX - v5projX
            deltaY_5 = v6projY - v5projY
            segLenSq_5 = deltaX_5 * deltaX_5 + deltaY_5 * deltaY_5
            t5 = ((X - v5projX) * deltaX_5 + (Y - v5projY) * deltaY_5) / segLenSq_5
            t5_clamped = clamp(t5, 0, 1)
            projX_5 = v5projX + t5_clamped * deltaX_5
            projY_5 = v5projY + t5_clamped * deltaY_5
            distanceSq_5 = (X - projX_5) * (X - projX_5) + (Y - projY_5) * (Y - projY_5)

            deltaX_6 = v7projX - v6projX
            deltaY_6 = v7projY - v6projY
            segLenSq_6 = deltaX_6 * deltaX_6 + deltaY_6 * deltaY_6
            t6 = ((X - v6projX) * deltaX_6 + (Y - v6projY) * deltaY_6) / segLenSq_6
            t6_clamped = clamp(t6, 0, 1)
            projX_6 = v6projX + t6_clamped * deltaX_6
            projY_6 = v6projY + t6_clamped * deltaY_6
            distanceSq_6 = (X - projX_6) * (X - projX_6) + (Y - projY_6) * (Y - projY_6)

            deltaX_7 = v4projX - v7projX
            deltaY_7 = v4projY - v7projY
            segLenSq_7 = deltaX_7 * deltaX_7 + deltaY_7 * deltaY_7
            t7 = ((X - v7projX) * deltaX_7 + (Y - v7projY) * deltaY_7) / segLenSq_7
            t7_clamped = clamp(t7, 0, 1)
            projX_7 = v7projX + t7_clamped * deltaX_7
            projY_7 = v7projY + t7_clamped * deltaY_7
            distanceSq_7 = (X - projX_7) * (X - projX_7) + (Y - projY_7) * (Y - projY_7)

            deltaX_8 = v4projX - v0projX
            deltaY_8 = v4projY - v0projY
            segLenSq_8 = deltaX_8 * deltaX_8 + deltaY_8 * deltaY_8
            t8 = ((X - v0projX) * deltaX_8 + (Y - v0projY) * deltaY_8) / segLenSq_8
            t8_clamped = clamp(t8, 0, 1)
            projX_8 = v0projX + t8_clamped * deltaX_8
            projY_8 = v0projY + t8_clamped * deltaY_8
            distanceSq_8 = (X - projX_8) * (X - projX_8) + (Y - projY_8) * (Y - projY_8)

            deltaX_9 = v5projX - v1projX
            deltaY_9 = v5projY - v1projY
            segLenSq_9 = deltaX_9 * deltaX_9 + deltaY_9 * deltaY_9
            t9 = ((X - v1projX) * deltaX_9 + (Y - v1projY) * deltaY_9) / segLenSq_9
            t9_clamped = clamp(t9, 0, 1)
            projX_9 = v1projX + t9_clamped * deltaX_9
            projY_9 = v1projY + t9_clamped * deltaY_9
            distanceSq_9 = (X - projX_9) * (X - projX_9) + (Y - projY_9) * (Y - projY_9)

            deltaX_10 = v6projX - v2projX
            deltaY_10 = v6projY - v2projY
            segLenSq_10 = deltaX_10 * deltaX_10 + deltaY_10 * deltaY_10
            t10 = ((X - v2projX) * deltaX_10 + (Y - v2projY) * deltaY_10) / segLenSq_10
            t10_clamped = clamp(t10, 0, 1)
            projX_10 = v2projX + t10_clamped * deltaX_10
            projY_10 = v2projY + t10_clamped * deltaY_10
            distanceSq_10 = (X - projX_10) * (X - projX_10) + (Y - projY_10) * (Y - projY_10)

            deltaX_11 = v7projX - v3projX
            deltaY_11 = v7projY - v3projY
            segLenSq_11 = deltaX_11 * deltaX_11 + deltaY_11 * deltaY_11
            t11 = ((X - v3projX) * deltaX_11 + (Y - v3projY) * deltaY_11) / segLenSq_11
            t11_clamped = clamp(t11, 0, 1)
            projX_11 = v3projX + t11_clamped * deltaX_11
            projY_11 = v3projY + t11_clamped * deltaY_11
            distanceSq_11 = (X - projX_11) * (X - projX_11) + (Y - projY_11) * (Y - projY_11)

            finalMinDistanceSquared = nth_1(distanceSq_0, distanceSq_1, distanceSq_2, distanceSq_3, distanceSq_4, distanceSq_5, distanceSq_6, distanceSq_7, distanceSq_8, distanceSq_9, distanceSq_10, distanceSq_11)
            halfThickness = thickness / 2.0
            halfThicknessSq = halfThickness * halfThickness
            doDraw = finalMinDistanceSquared <= halfThicknessSq
            RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0
            ''')

    return clip.akarin.Expr(expr)