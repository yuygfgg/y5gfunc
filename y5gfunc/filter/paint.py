from vstools import core
from vstools import vs
from ..expr import infix2postfix


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
            half_thickness = thickness / 2.0
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
            half_thickness = thickness / 2.0
            dx = X - cx
            dy = Y - cy
            distance_sq = dx * dx + dy * dy
            radius_minus_half = radius - half_thickness
            lower_sq = radius_minus_half * radius_minus_half
            lower_sq = lower_sq < 0.0 ? 0.0 : lower_sq
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
            cx = (f1x + f2x) / 2.0
            cy = (f1y + f2y) / 2.0
            aa = ellipse_sum / 2.0
            a2 = aa * aa
            dx = f2x - f1x
            dy = f2y - f1y
            c2 = (dx * dx + dy * dy) / 4.0
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
    sample_count: int,
    factor: str = "1.0"
) -> vs.VideoNode:

    assert sample_count >= 2
    assert clip.format.num_planes == 1

    expression_lines = []

    expression_lines.append(f"controlPoint0X = {controlPoint0X}")
    expression_lines.append(f"controlPoint0Y = {controlPoint0Y}")
    expression_lines.append(f"controlPoint1X = {controlPoint1X}")
    expression_lines.append(f"controlPoint1Y = {controlPoint1Y}")
    expression_lines.append(f"controlPoint2X = {controlPoint2X}")
    expression_lines.append(f"controlPoint2Y = {controlPoint2Y}")
    expression_lines.append(f"controlPoint3X = {controlPoint3X}")
    expression_lines.append(f"controlPoint3Y = {controlPoint3Y}")
    expression_lines.append(f"thickness = {thickness}")
    expression_lines.append(f"color = {color}")
    expression_lines.append(f"factor = {factor}")
    expression_lines.append("halfThickness = thickness / 2.0")
    expression_lines.append("halfThicknessSquared = halfThickness * halfThickness")
    
    total_samples = sample_count
    for sample_index in range(total_samples):
        t_value = sample_index / (total_samples - 1)
        expression_lines.append(f"parameterT_{sample_index} = {t_value}")
        expression_lines.append(f"oneMinusT_{sample_index} = 1 - parameterT_{sample_index}")
        expression_lines.append(f"oneMinusT_squared_{sample_index} = oneMinusT_{sample_index} * oneMinusT_{sample_index}")
        expression_lines.append(f"oneMinusT_cubed_{sample_index} = oneMinusT_squared_{sample_index} * oneMinusT_{sample_index}")
        expression_lines.append(f"parameterT_squared_{sample_index} = parameterT_{sample_index} * parameterT_{sample_index}")
        expression_lines.append(f"parameterT_cubed_{sample_index} = parameterT_squared_{sample_index} * parameterT_{sample_index}")
        expression_lines.append(
            f"bezierX_{sample_index} = oneMinusT_cubed_{sample_index} * controlPoint0X + "
            f"3 * oneMinusT_squared_{sample_index} * parameterT_{sample_index} * controlPoint1X + "
            f"3 * oneMinusT_{sample_index} * parameterT_squared_{sample_index} * controlPoint2X + "
            f"parameterT_cubed_{sample_index} * controlPoint3X"
        )
        expression_lines.append(
            f"bezierY_{sample_index} = oneMinusT_cubed_{sample_index} * controlPoint0Y + "
            f"3 * oneMinusT_squared_{sample_index} * parameterT_{sample_index} * controlPoint1Y + "
            f"3 * oneMinusT_{sample_index} * parameterT_squared_{sample_index} * controlPoint2Y + "
            f"parameterT_cubed_{sample_index} * controlPoint3Y"
        )
    
    segment_distance_squared_expressions = []
    for seg_index in range(total_samples - 1):
        expression_lines.append(
            f"deltaX_{seg_index} = bezierX_{seg_index+1} - bezierX_{seg_index}"
        )
        expression_lines.append(
            f"deltaY_{seg_index} = bezierY_{seg_index+1} - bezierY_{seg_index}"
        )
        expression_lines.append(
            f"segmentLengthSquared_{seg_index} = deltaX_{seg_index} * deltaX_{seg_index} + deltaY_{seg_index} * deltaY_{seg_index}"
        )
        expression_lines.append(
            f"tSegment_{seg_index} = ((X - bezierX_{seg_index}) * deltaX_{seg_index} + (Y - bezierY_{seg_index}) * deltaY_{seg_index}) / segmentLengthSquared_{seg_index}"
        )
        expression_lines.append(
            f"tClamped_{seg_index} = clamp(tSegment_{seg_index}, 0, 1)"
        )
        expression_lines.append(
            f"projectionX_{seg_index} = bezierX_{seg_index} + tClamped_{seg_index} * deltaX_{seg_index}"
        )
        expression_lines.append(
            f"projectionY_{seg_index} = bezierY_{seg_index} + tClamped_{seg_index} * deltaY_{seg_index}"
        )
        expression_lines.append(
            f"distanceSquared_{seg_index} = (X - projectionX_{seg_index}) * (X - projectionX_{seg_index}) + (Y - projectionY_{seg_index}) * (Y - projectionY_{seg_index})"
        )
        segment_distance_squared_expressions.append(f"distanceSquared_{seg_index}")
    
    distance_arguments = ", ".join(segment_distance_squared_expressions)
    expression_lines.append(f"finalMinDistanceSquared = nth_1({distance_arguments})")
    
    expression_lines.append("doDraw = finalMinDistanceSquared <= halfThicknessSquared")
    expression_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")
    
    full_expression = "\n".join(expression_lines)
    
    converted_expression = infix2postfix(full_expression)
    return clip.akarin.Expr(converted_expression)