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
            dd = sqrt(dx * dx + dy * dy)
            diff = abs(dd - radius)
            do = diff <= half_thickness
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
            half_thickness = thickness / 2.0
            dx1 = X - f1x
            dy1 = Y - f1y
            d1 = sqrt(dx1 * dx1 + dy1 * dy1)
            dx2 = X - f2x
            dy2 = Y - f2y
            d2 = sqrt(dx2 * dx2 + dy2 * dy2)
            total_d = d1 + d2
            diff = abs(total_d - ellipse_sum)
            do = diff <= half_thickness
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
    
    total_samples = sample_count
    for sample_index in range(total_samples):
        parameter_t = sample_index / (total_samples - 1)
        expression_lines.append(f"parameterT_{sample_index} = {parameter_t}")
        expression_lines.append(f"oneMinusT_{sample_index} = 1 - parameterT_{sample_index}")
        expression_lines.append(
            f"bezierX_{sample_index} = (oneMinusT_{sample_index} ** 3) * controlPoint0X + "
            f"3 * (oneMinusT_{sample_index} ** 2) * parameterT_{sample_index} * controlPoint1X + "
            f"3 * oneMinusT_{sample_index} * (parameterT_{sample_index} ** 2) * controlPoint2X + "
            f"(parameterT_{sample_index} ** 3) * controlPoint3X"
        )
        expression_lines.append(
            f"bezierY_{sample_index} = (oneMinusT_{sample_index} ** 3) * controlPoint0Y + "
            f"3 * (oneMinusT_{sample_index} ** 2) * parameterT_{sample_index} * controlPoint1Y + "
            f"3 * oneMinusT_{sample_index} * (parameterT_{sample_index} ** 2) * controlPoint2Y + "
            f"(parameterT_{sample_index} ** 3) * controlPoint3Y"
        )
    
    segment_distance_expressions = []
    for segment_index in range(total_samples - 1):
        expression_lines.append(
            f"deltaX_{segment_index} = bezierX_{segment_index+1} - bezierX_{segment_index}"
        )
        expression_lines.append(
            f"deltaY_{segment_index} = bezierY_{segment_index+1} - bezierY_{segment_index}"
        )
        expression_lines.append(
            f"segmentLengthSquared_{segment_index} = (deltaX_{segment_index} ** 2) + (deltaY_{segment_index} ** 2)"
        )
        expression_lines.append(
            f"tSegment_{segment_index} = ((X - bezierX_{segment_index}) * deltaX_{segment_index} + "
            f"(Y - bezierY_{segment_index}) * deltaY_{segment_index}) / segmentLengthSquared_{segment_index}"
        )
        expression_lines.append(
            f"tClamped_{segment_index} = clamp(tSegment_{segment_index}, 0, 1)"
        )
        expression_lines.append(
            f"projectionX_{segment_index} = bezierX_{segment_index} + tClamped_{segment_index} * deltaX_{segment_index}"
        )
        expression_lines.append(
            f"projectionY_{segment_index} = bezierY_{segment_index} + tClamped_{segment_index} * deltaY_{segment_index}"
        )
        expression_lines.append(
            f"segmentDistance_{segment_index} = sqrt((X - projectionX_{segment_index}) ** 2 + "
            f"(Y - projectionY_{segment_index}) ** 2)"
        )
        segment_distance_expressions.append(f"segmentDistance_{segment_index}")
    
    distance_arguments = ", ".join(segment_distance_expressions)
    expression_lines.append(f"finalMinDistance = nth_1({distance_arguments})")
    
    expression_lines.append("doDraw = finalMinDistance <= halfThickness")
    expression_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")
    
    full_expression = "\n".join(expression_lines)
    
    converted_expression = infix2postfix(full_expression)
    return clip.akarin.Expr(converted_expression)