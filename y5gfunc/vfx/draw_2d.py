from vstools import vs
from ..expr import compile


def draw_line(
    clip: vs.VideoNode,
    sx: str,
    sy: str,
    ex: str,
    ey: str,
    thickness: str,
    color: str,
    factor: str = "1",
) -> vs.VideoNode:
    """
    Draw a line on a video clip.

    Args:
        clip: The video clip to draw the line on.
        sx: The x-coordinate of the start point.
        sy: The y-coordinate of the start point.
        ex: The x-coordinate of the end point.
        ey: The y-coordinate of the end point.
        thickness: The thickness of the line.
        color: The color of the line.
        factor: The factor of the line.

    Returns:
        The video clip with the line drawn on it.

    Raises:
        AssertionError: If the format of the video clip has more than 1 plane.
    """
    assert clip.format.num_planes == 1

    expr = compile(
        f"""
            sx = {sx}
            sy = {sy}
            ex = {ex}
            ey = {ey}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            dx = ex - sx
            dy = ey - sy
            L2 = (ex - sx) * dx + dy ** 2
            half_thickness = thickness / 2
            half_thickness_sq = half_thickness ** 2
            tt = ((X - sx) * dx + (Y - sy) * dy) / L2
            tt = clamp(tt, 0, 1)
            proj_x = sx + tt * dx
            proj_y = sy + tt * dy
            d2 = (X - proj_x) * (X - proj_x) + (Y - proj_y) * (Y - proj_y)
            do = d2 <= half_thickness_sq
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            """
    )

    return clip.akarin.Expr(expr)


def draw_circle(
    clip: vs.VideoNode,
    cx: str,
    cy: str,
    radius: str,
    thickness: str,
    color: str,
    factor: str = "1",
) -> vs.VideoNode:
    """
    Draw a circle on a video clip.

    Args:
        clip: The video clip to draw the circle on.
        cx: The x-coordinate of the circle's center.
        cy: The y-coordinate of the circle's center.
        radius: The radius of the circle.
        thickness: The thickness of the circle's line.
        color: The color of the circle.
        factor: The blending factor for the color.

    Returns:
        The video clip with the circle drawn on it.

    Raises:
        AssertionError: If the format of the video clip has more than 1 plane.
    """
    assert clip.format.num_planes == 1

    expr = compile(
        f"""
            cx = {cx}
            cy = {cy}
            radius = {radius}
            thickness = {thickness}
            color = {color}
            factor = {factor}
            half_thickness = thickness / 2
            dx = X - cx
            dy = Y - cy
            distance_sq = dx ** 2 + dy ** 2
            radius_minus_half = radius - half_thickness
            lower_sq = radius_minus_half ** 2
            lower_sq = max(lower_sq, 0)
            upper_sq = (radius + half_thickness) ** 2
            do = distance_sq >= lower_sq && distance_sq <= upper_sq
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            """
    )

    return clip.akarin.Expr(expr)


def draw_ellipse(
    clip: vs.VideoNode,
    f1x: str,
    f1y: str,
    f2x: str,
    f2y: str,
    ellipse_sum: str,
    thickness: str,
    color: str,
    factor: str = "1",
) -> vs.VideoNode:
    """
    Draw an ellipse on a video clip, defined by its two focal points.

    Args:
        clip: The video clip to draw the ellipse on.
        f1x: The x-coordinate of the first focal point.
        f1y: The y-coordinate of the first focal point.
        f2x: The x-coordinate of the second focal point.
        f2y: The y-coordinate of the second focal point.
        ellipse_sum: The sum of the distances from any point on the ellipse to the two focal points.
        thickness: The thickness of the ellipse's line.
        color: The color of the ellipse.
        factor: The blending factor for the color.

    Returns:
        The video clip with the ellipse drawn on it.

    Raises:
        AssertionError: If the format of the video clip has more than 1 plane.
    """
    assert clip.format.num_planes == 1

    expr = compile(
        f"""
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
            a2 = aa ** 2
            dx = f2x - f1x
            dy = f2y - f1y
            c2 = (dx ** 2 + dy ** 2) / 4
            b2 = a2 - c2
            value = ((X - cx) ** 2) / a2 + ((Y - cy) ** 2) / b2
            norm_thresh = thickness / ellipse_sum
            do = abs(value - 1) <= norm_thresh
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            """
    )

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
    factor: str = "1",
) -> vs.VideoNode:
    """
    Draw a cubic Bezier curve on a video clip.

    The curve is defined by four control points.

    Args:
        clip: The video clip to draw the curve on.
        controlPoint0X: The x-coordinate of the first control point (start).
        controlPoint0Y: The y-coordinate of the first control point (start).
        controlPoint1X: The x-coordinate of the second control point.
        controlPoint1Y: The y-coordinate of the second control point.
        controlPoint2X: The x-coordinate of the third control point.
        controlPoint2Y: The y-coordinate of the third control point.
        controlPoint3X: The x-coordinate of the fourth control point (end).
        controlPoint3Y: The y-coordinate of the fourth control point (end).
        thickness: The thickness of the curve.
        color: The color of the curve.
        sample_count: The number of line segments to approximate the curve.
        factor: The blending factor for the color.

    Returns:
        The video clip with the Bezier curve drawn on it.

    Raises:
        AssertionError: If `sample_count` is less than 2 or if the clip has more than one plane.
    """
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
    expr_lines.append("halfThickness = thickness / 2")
    expr_lines.append("halfThicknessSquared = halfThickness ** 2")

    total_samples = sample_count
    for sample_index in range(total_samples):
        t_value = sample_index / (total_samples - 1)
        expr_lines.append(f"parameterT_{sample_index} = {t_value}")
        expr_lines.append(f"oneMinusT_{sample_index} = 1 - parameterT_{sample_index}")
        expr_lines.append(
            f"oneMinusT_squared_{sample_index} = oneMinusT_{sample_index} ** 2"
        )
        expr_lines.append(
            f"oneMinusT_cubed_{sample_index} = oneMinusT_{sample_index} ** 3"
        )
        expr_lines.append(
            f"parameterT_squared_{sample_index} = parameterT_{sample_index} ** 2"
        )
        expr_lines.append(
            f"parameterT_cubed_{sample_index} = parameterT_{sample_index} ** 3"
        )
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
        expr_lines.append(
            f"deltaX_{seg_index} = bezierX_{seg_index+1} - bezierX_{seg_index}"
        )
        expr_lines.append(
            f"deltaY_{seg_index} = bezierY_{seg_index+1} - bezierY_{seg_index}"
        )
        expr_lines.append(
            f"segmentLengthSquared_{seg_index} = deltaX_{seg_index} * deltaX_{seg_index} + deltaY_{seg_index} * deltaY_{seg_index}"
        )
        expr_lines.append(
            f"tSegment_{seg_index} = ((X - bezierX_{seg_index}) * deltaX_{seg_index} + (Y - bezierY_{seg_index}) * deltaY_{seg_index}) / segmentLengthSquared_{seg_index}"
        )
        expr_lines.append(f"tClamped_{seg_index} = clamp(tSegment_{seg_index}, 0, 1)")
        expr_lines.append(
            f"projectionX_{seg_index} = bezierX_{seg_index} + tClamped_{seg_index} * deltaX_{seg_index}"
        )
        expr_lines.append(
            f"projectionY_{seg_index} = bezierY_{seg_index} + tClamped_{seg_index} * deltaY_{seg_index}"
        )
        expr_lines.append(
            f"distanceSquared_{seg_index} = (X - projectionX_{seg_index}) ** 2 + (Y - projectionY_{seg_index}) ** 2)"
        )
        segment_distance_squared_expressions.append(f"distanceSquared_{seg_index}")

    distance_arguments = ", ".join(segment_distance_squared_expressions)
    expr_lines.append(f"finalMinDistanceSquared = nth_1({distance_arguments})")

    expr_lines.append("doDraw = finalMinDistanceSquared <= halfThicknessSquared")
    expr_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")

    full_expression = "\n".join(expr_lines)

    converted_expression = compile(full_expression)
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
    escapeRadius: str = "2",
) -> vs.VideoNode:
    """
    Draw a zooming animation into the Mandelbrot set.

    Args:
        clip: The video clip to draw on.
        centerX: The x-coordinate for the center of the fractal on screen.
        centerY: The y-coordinate for the center of the fractal on screen.
        color: The base color for the fractal rendering.
        initialZoom: The initial zoom level.
        zoomSpeed: The speed of zooming in.
        centerMoveSpeed: The speed at which the center of the view moves from C0 to C1.
        fractalC0_re: The real part of the starting complex number for the center.
        fractalC0_im: The imaginary part of the starting complex number for the center.
        fractalC1_re: The real part of the ending complex number for the center.
        fractalC1_im: The imaginary part of the ending complex number for the center.
        maxIter: The maximum number of iterations for the Mandelbrot calculation.
        escapeRadius: The escape radius for the Mandelbrot calculation.

    Returns:
        The video clip with the Mandelbrot animation.

    Raises:
        AssertionError: If `maxIter` is less than 1.
    """
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
        expr_lines.append(
            f"z_re_{i} = (z_re_{prev} * z_re_{prev} - z_im_{prev} * z_im_{prev} + c_re)"
        )
        expr_lines.append(f"z_im_{i} = (2 * z_re_{prev} * z_im_{prev} + c_im)")
        expr_lines.append(f"r2_{i} = (z_re_{i} * z_re_{i} + z_im_{i} * z_im_{i})")

    iter_expr = str(maxIter)
    for i in range(maxIter, 0, -1):
        iter_expr = f"(r2_{i} > escapeSq ? {i} : {iter_expr})"
    expr_lines.append("iterResult = " + iter_expr)

    expr_lines.append("RESULT = (iterResult / maxIter) * color")

    full_expr = "\n".join(expr_lines)
    converted_expr = compile(full_expr)
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
    factor: str = "1",
    sample_count=500,
) -> vs.VideoNode:
    """
    Draw a spiral on a video clip.

    The spiral is defined by the equation r = a + b * theta.

    Args:
        clip: The video clip to draw the spiral on.
        centerX: The x-coordinate of the spiral's center.
        centerY: The y-coordinate of the spiral's center.
        a: The 'a' parameter in the spiral equation, affecting the starting radius.
        b: The 'b' parameter in the spiral equation, affecting the distance between arms.
        startAngle: The starting angle of the spiral in radians.
        endAngle: The ending angle of the spiral in radians.
        thickness: The thickness of the spiral line.
        color: The color of the spiral.
        factor: The blending factor for the color.
        sample_count: The number of line segments to approximate the spiral.

    Returns:
        The video clip with the spiral drawn on it.

    Raises:
        AssertionError: If the format of the video clip has more than 1 plane.
    """
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
    expr_lines.append("halfThickness = thickness / 2")
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
        expr_lines.append(
            f"segmentLengthSquared_{i} = deltaX_{i} * deltaX_{i} + deltaY_{i} * deltaY_{i}"
        )
        expr_lines.append(
            f"t_{i} = ((X - spiralX_{i}) * deltaX_{i} + (Y - spiralY_{i}) * deltaY_{i}) / segmentLengthSquared_{i}"
        )
        expr_lines.append(f"tClamped_{i} = clamp(t_{i}, 0, 1)")
        expr_lines.append(f"projectionX_{i} = spiralX_{i} + tClamped_{i} * deltaX_{i}")
        expr_lines.append(f"projectionY_{i} = spiralY_{i} + tClamped_{i} * deltaY_{i}")
        expr_lines.append(
            f"distanceSquared_{i} = (X - projectionX_{i}) ** 2 + (Y - projectionY_{i}) ** 2"
        )
        segment_distance_exprs.append(f"distanceSquared_{i}")

    expr_lines.append(
        "finalMinDistanceSquared = nth_1(" + ", ".join(segment_distance_exprs) + ")"
    )
    expr_lines.append("doDraw = finalMinDistanceSquared <= halfThicknessSquared")
    expr_lines.append("RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0")

    full_expr = "\n".join(expr_lines)
    converted_expr = compile(full_expr)
    return clip.akarin.Expr(converted_expr)
