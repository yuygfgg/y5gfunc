from vstools import vs
from ..expr import infix2postfix
import sys


def draw_line(clip: vs.VideoNode, sx: str, sy: str, ex: str, ey: str, thickness: str, color: str, factor: str = "1") -> vs.VideoNode: 
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
            ''')
    
    return clip.akarin.Expr(expr)

def draw_circle(clip: vs.VideoNode, cx: str, cy: str, radius: str, thickness: str, color: str, factor: str = "1") -> vs.VideoNode:
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
            distance_sq = dx ** 2 + dy ** 2
            radius_minus_half = radius - half_thickness
            lower_sq = radius_minus_half ** 2
            lower_sq = max(lower_sq, 0)
            upper_sq = (radius + half_thickness) ** 2
            do = distance_sq >= lower_sq && distance_sq <= upper_sq
            RESULT = do ? ((1 - factor) * src0 + factor * color) : src0
            ''')

    return clip.akarin.Expr(expr)

def draw_ellipse(clip: vs.VideoNode, f1x: str, f1y: str, f2x: str, f2y: str, ellipse_sum: str, thickness: str, color: str, factor: str = "1") -> vs.VideoNode:
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
            c2 = (dx ** 2 + dy ** 2) / 4
            b2 = a2 - c2
            value = ((X - cx) ** 2) / a2 + ((Y - cy) ** 2) / b2
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
    factor: str = "1"
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
    expr_lines.append("halfThickness = thickness / 2")
    expr_lines.append("halfThicknessSquared = halfThickness ** 2")
    
    total_samples = sample_count
    for sample_index in range(total_samples):
        t_value = sample_index / (total_samples - 1)
        expr_lines.append(f"parameterT_{sample_index} = {t_value}")
        expr_lines.append(f"oneMinusT_{sample_index} = 1 - parameterT_{sample_index}")
        expr_lines.append(f"oneMinusT_squared_{sample_index} = oneMinusT_{sample_index} ** 2")
        expr_lines.append(f"oneMinusT_cubed_{sample_index} = oneMinusT_{sample_index} ** 3")
        expr_lines.append(f"parameterT_squared_{sample_index} = parameterT_{sample_index} ** 2")
        expr_lines.append(f"parameterT_cubed_{sample_index} = parameterT_{sample_index} ** 3")
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
        expr_lines.append(f"distanceSquared_{seg_index} = (X - projectionX_{seg_index}) ** 2 + (Y - projectionY_{seg_index}) ** 2)")
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
    escapeRadius: str = "2"
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
    factor: str = "1",
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
    factor: str = "1"
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
            
            half = cubeSize / 2
            cos_rotationX = cos(rotationX)
            sin_rotationX = sin(rotationX)
            cos_rotationY = cos(rotationY)
            sin_rotationY = sin(rotationY)
            
            <global<cos_rotationX><sin_rotationX><cos_rotationY><sin_rotationY><translateZ><centerX><focal>>
            function project3d_x(vx, vy, vz) {{
                vy1 = vy * cos_rotationX - vz * sin_rotationX
                vz1 = vy * sin_rotationX + vz * cos_rotationX
                vx1 = vx
                vx2 = vx1 * cos_rotationY + vz1 * sin_rotationY
                vz2 = -vx1 * sin_rotationY + vz1 * cos_rotationY
                vy2 = vy1
                vz_final = vz2 + translateZ
                return centerX + (vx2 * focal) / vz_final
            }}
            
            <global<cos_rotationX><sin_rotationX><cos_rotationY><sin_rotationY><translateZ><centerY><focal>>
            function project3d_y(vx, vy, vz) {{
                vy1 = vy * cos_rotationX - vz * sin_rotationX
                vz1 = vy * sin_rotationX + vz * cos_rotationX
                vx1 = vx
                vx2 = vx1 * cos_rotationY + vz1 * sin_rotationY
                vz2 = -vx1 * sin_rotationY + vz1 * cos_rotationY
                vy2 = vy1
                vz_final = vz2 + translateZ
                return centerY + (vy2 * focal) / vz_final
            }}
            
            function distSqToSegment(x0, y0, x1, y1) {{
                dx = x1 - x0
                dy = y1 - y0
                segLenSq = dx * dx + dy * dy
                tt = ((X - x0) * dx + (Y - y0) * dy) / segLenSq
                t_clamped = clamp(tt, 0, 1)
                projX = x0 + t_clamped * dx
                projY = y0 + t_clamped * dy
                return (X - projX) ** 2 + (Y - projY) ** 2
            }}
            
            v0projX = project3d_x(-half, -half, -half)
            v0projY = project3d_y(-half, -half, -half)
            
            v1projX = project3d_x(half, -half, -half)
            v1projY = project3d_y(half, -half, -half)
            
            v2projX = project3d_x(half, half, -half)
            v2projY = project3d_y(half, half, -half)
            
            v3projX = project3d_x(-half, half, -half)
            v3projY = project3d_y(-half, half, -half)
            
            v4projX = project3d_x(-half, -half, half)
            v4projY = project3d_y(-half, -half, half)
            
            v5projX = project3d_x(half, -half, half)
            v5projY = project3d_y(half, -half, half)
            
            v6projX = project3d_x(half, half, half)
            v6projY = project3d_y(half, half, half)
            
            v7projX = project3d_x(-half, half, half)
            v7projY = project3d_y(-half, half, half)
            
            d0  = distSqToSegment(v0projX, v0projY, v1projX, v1projY)
            d1  = distSqToSegment(v1projX, v1projY, v2projX, v2projY)
            d2  = distSqToSegment(v2projX, v2projY, v3projX, v3projY)
            d3  = distSqToSegment(v3projX, v3projY, v0projX, v0projY)
            d4  = distSqToSegment(v4projX, v4projY, v5projX, v5projY)
            d5  = distSqToSegment(v5projX, v5projY, v6projX, v6projY)
            d6  = distSqToSegment(v6projX, v6projY, v7projX, v7projY)
            d7  = distSqToSegment(v7projX, v7projY, v4projX, v4projY)
            d8  = distSqToSegment(v0projX, v0projY, v4projX, v4projY)
            d9  = distSqToSegment(v1projX, v1projY, v5projX, v5projY)
            d10 = distSqToSegment(v2projX, v2projY, v6projX, v6projY)
            d11 = distSqToSegment(v3projX, v3projY, v7projX, v7projY)
            
            finalMinDistanceSquared = nth_1(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)
            
            halfThickness = thickness / 2
            halfThicknessSq = halfThickness ** 2
            doDraw = finalMinDistanceSquared <= halfThicknessSq
            
            RESULT = doDraw ? ((1 - factor) * src0 + factor * color) : src0
            ''')

    return clip.akarin.Expr(expr)

def render_triangle_scene(
    clip: vs.VideoNode,
    points: list,
    faces: list,
    lights: list,
    camX: str,
    camY: str,
    camZ: str,
    rotationX: str,
    rotationY: str,
    focal: str,
    background: str = "0"
) -> vs.VideoNode:

    expr_lines = []

    expr_lines.append("<global<camX><camY><camZ><rotationX><rotationY><focal><screenCenterX><screenCenterY><epsilon>>")
    expr_lines.append("function cam_coord_x (x , y , z) {")
    expr_lines.append("    tx = x - camX")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    ty1 = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    tz1 = ty * sin(rotationX) + tz * cos(rotationX)")
    expr_lines.append("    cx = tx * cos(rotationY) + tz1 * sin(rotationY)")
    expr_lines.append("    return cx")
    expr_lines.append("}")

    expr_lines.append("<global<camX><camY><camZ><rotationX><rotationY><focal><screenCenterX><screenCenterY><epsilon>>")
    expr_lines.append("function cam_coord_y (x , y , z) {")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    cy = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    return cy")
    expr_lines.append("}")

    expr_lines.append("<global<camX><camY><camZ><rotationX><rotationY><epsilon><huge>>")
    expr_lines.append("function cam_coord_z (x , y , z) {")
    expr_lines.append("    tx = x - camX")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    ty1 = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    tz1 = ty * sin(rotationX) + tz * cos(rotationX)")
    expr_lines.append("    cz = -tx * sin(rotationY) + tz1 * cos(rotationY)")
    expr_lines.append("    return cz < 0 ? huge : cz")
    expr_lines.append("}")

    expr_lines.append("<global<focal><screenCenterX>>")
    expr_lines.append("function projectX (x , y , z) {")
    expr_lines.append("    px = cam_coord_x(x , y , z)")
    expr_lines.append("    pz = cam_coord_z(x , y , z)")
    expr_lines.append("    return screenCenterX + (px * focal) / pz")
    expr_lines.append("}")

    expr_lines.append("<global<focal><screenCenterY>>")
    expr_lines.append("function projectY (x , y , z) {")
    expr_lines.append("    py = cam_coord_y(x , y , z)")
    expr_lines.append("    pz = cam_coord_z(x , y , z)")
    expr_lines.append("    return screenCenterY + (py * focal) / pz")
    expr_lines.append("}")

    expr_lines.append(f"camX = {camX}")
    expr_lines.append(f"camY = {camY}")
    expr_lines.append(f"camZ = {camZ}")
    expr_lines.append(f"rotationX = {rotationX}")
    expr_lines.append(f"rotationY = {rotationY}")
    expr_lines.append(f"focal = {focal}")
    expr_lines.append(f"background = {background}")
    expr_lines.append("screenCenterX = width / 2")
    expr_lines.append("screenCenterY = height / 2")
    expr_lines.append("epsilon = 0.0001")
    expr_lines.append("huge = 1e9")
    expr_lines.append("ambient = 0.2")

    for idx, light in enumerate(lights):
        expr_lines.append(f"light{idx}_raw_x = {light['lx']}")
        expr_lines.append(f"light{idx}_raw_y = {light['ly']}")
        expr_lines.append(f"light{idx}_raw_z = {light['lz']}")
        expr_lines.append(f"light{idx}_intensity = {light['intensity']}")
        expr_lines.append(f"mag_{idx} = sqrt(light{idx}_raw_x ** 2 + light{idx}_raw_y ** 2 + light{idx}_raw_z ** 2)")
        expr_lines.append(f"light{idx}_nx = light{idx}_raw_x / mag_{idx}")
        expr_lines.append(f"light{idx}_ny = light{idx}_raw_y / mag_{idx}")
        expr_lines.append(f"light{idx}_nz = light{idx}_raw_z / mag_{idx}")
        expr_lines.append(f"temp_ty_{idx} = light{idx}_ny * cos(rotationX) - light{idx}_nz * sin(rotationX)")
        expr_lines.append(f"temp_tz_{idx} = light{idx}_ny * sin(rotationX) + light{idx}_nz * cos(rotationX)")
        expr_lines.append(f"light{idx}_lx = light{idx}_nx * cos(rotationY) + temp_tz_{idx} * sin(rotationY)")
        expr_lines.append(f"light{idx}_ly = temp_ty_{idx}")
        expr_lines.append(f"light{idx}_lz = -light{idx}_nx * sin(rotationY) + temp_tz_{idx} * cos(rotationY)")

    for i, pt in enumerate(points):
        expr_lines.append(f"point{i}_x = {pt['x']}")
        expr_lines.append(f"point{i}_y = {pt['y']}")
        expr_lines.append(f"point{i}_z = {pt['z']}")
        expr_lines.append(f"projX_{i} = projectX(point{i}_x , point{i}_y , point{i}_z)")
        expr_lines.append(f"projY_{i} = projectY(point{i}_x , point{i}_y , point{i}_z)")
        expr_lines.append(f"cam_x_{i} = cam_coord_x(point{i}_x , point{i}_y , point{i}_z)")
        expr_lines.append(f"cam_y_{i} = cam_coord_y(point{i}_x , point{i}_y , point{i}_z)")
        expr_lines.append(f"cam_z_{i} = cam_coord_z(point{i}_x , point{i}_y , point{i}_z)")

    face_t_names = []
    face_shading_names = []
    face_count = len(faces)
    for f_idx, face in enumerate(faces):
        a_idx = face["a"]
        b_idx = face["b"]
        c_idx = face["c"]
        face_color = face.get("color", "1")
        expr_lines.append(f"E0_{f_idx} = (X - projX_{a_idx}) * (projY_{b_idx} - projY_{a_idx}) - (Y - projY_{a_idx}) * (projX_{b_idx} - projX_{a_idx})")
        expr_lines.append(f"E1_{f_idx} = (X - projX_{b_idx}) * (projY_{c_idx} - projY_{b_idx}) - (Y - projY_{b_idx}) * (projX_{c_idx} - projX_{b_idx})")
        expr_lines.append(f"E2_{f_idx} = (X - projX_{c_idx}) * (projY_{a_idx} - projY_{c_idx}) - (Y - projY_{c_idx}) * (projX_{a_idx} - projX_{c_idx})")
        expr_lines.append(f"inside_pos_{f_idx} = E0_{f_idx} >= 0 && E1_{f_idx} >= 0 && E2_{f_idx} >= 0")
        expr_lines.append(f"inside_neg_{f_idx} = E0_{f_idx} <= 0 && E1_{f_idx} <= 0 && E2_{f_idx} <= 0")
        expr_lines.append(f"inside_{f_idx} = inside_pos_{f_idx} || inside_neg_{f_idx}")

        expr_lines.append(f"area_{f_idx} = (projX_{b_idx} - projX_{a_idx}) * (projY_{c_idx} - projY_{a_idx}) - (projX_{c_idx} - projX_{a_idx}) * (projY_{b_idx} - projY_{a_idx})")
        expr_lines.append(f"alpha_{f_idx} = ((projX_{b_idx} - X) * (projY_{c_idx} - Y) - (projX_{c_idx} - X) * (projY_{b_idx} - Y)) / area_{f_idx}")
        expr_lines.append(f"beta_{f_idx} = ((projX_{c_idx} - X) * (projY_{a_idx} - Y) - (projX_{a_idx} - X) * (projY_{c_idx} - Y)) / area_{f_idx}")
        expr_lines.append(f"gamma_{f_idx} = 1 - alpha_{f_idx} - beta_{f_idx}")
        expr_lines.append(f"depth_{f_idx} = alpha_{f_idx} * cam_z_{a_idx} + beta_{f_idx} * cam_z_{b_idx} + gamma_{f_idx} * cam_z_{c_idx}")

        expr_lines.append(f"ex1_{f_idx} = cam_x_{b_idx} - cam_x_{a_idx}")
        expr_lines.append(f"ey1_{f_idx} = cam_y_{b_idx} - cam_y_{a_idx}")
        expr_lines.append(f"ez1_{f_idx} = cam_z_{b_idx} - cam_z_{a_idx}")
        expr_lines.append(f"ex2_{f_idx} = cam_x_{c_idx} - cam_x_{a_idx}")
        expr_lines.append(f"ey2_{f_idx} = cam_y_{c_idx} - cam_y_{a_idx}")
        expr_lines.append(f"ez2_{f_idx} = cam_z_{c_idx} - cam_z_{a_idx}")
        expr_lines.append(f"nx_{f_idx} = ey1_{f_idx} * ez2_{f_idx} - ez1_{f_idx} * ey2_{f_idx}")
        expr_lines.append(f"ny_{f_idx} = ez1_{f_idx} * ex2_{f_idx} - ex1_{f_idx} * ez2_{f_idx}")
        expr_lines.append(f"nz_{f_idx} = ex1_{f_idx} * ey2_{f_idx} - ey1_{f_idx} * ex2_{f_idx}")
        expr_lines.append(f"norm_{f_idx} = sqrt(nx_{f_idx} ** 2 + ny_{f_idx} ** 2 + nz_{f_idx} ** 2)")
        expr_lines.append(f"nx_{f_idx} = nx_{f_idx} / norm_{f_idx}")
        expr_lines.append(f"ny_{f_idx} = ny_{f_idx} / norm_{f_idx}")
        expr_lines.append(f"nz_{f_idx} = nz_{f_idx} / norm_{f_idx}")

        for l_idx in range(len(lights)):
            dot_expr = f"nx_{f_idx} * light{l_idx}_lx + ny_{f_idx} * light{l_idx}_ly + nz_{f_idx} * light{l_idx}_lz"
            expr_lines.append(f"dot_{f_idx}_{l_idx} = {dot_expr}")
            expr_lines.append(f"diffuse_{f_idx}_{l_idx} = max(dot_{f_idx}_{l_idx}, 0)")
            expr_lines.append(f"contrib_{f_idx}_{l_idx} = diffuse_{f_idx}_{l_idx} * light{l_idx}_intensity")

        expr_lines.append(f"sum_diffuse_{f_idx} = ambient")
        for l_idx in range(len(lights)):
            expr_lines.append(f"sum_diffuse_{f_idx} = sum_diffuse_{f_idx} + contrib_{f_idx}_{l_idx}")

        expr_lines.append(f"lighting_{f_idx} = sum_diffuse_{f_idx} > 1 ? 1 : sum_diffuse_{f_idx}")

        expr_lines.append(f"faceColor_{f_idx} = {face_color}")
        expr_lines.append(f"shading_{f_idx} = lighting_{f_idx} * faceColor_{f_idx}")

        expr_lines.append(f"t_face_{f_idx} = inside_{f_idx} == 1 ? depth_{f_idx} : huge")
        expr_lines.append(f"shading_face_{f_idx} = inside_{f_idx} == 1 ? shading_{f_idx} : 0")

        face_t_names.append(f"t_face_{f_idx}")
        face_shading_names.append(f"shading_face_{f_idx}")

    if face_t_names:
        face_t_args = ", ".join(face_t_names)
        expr_lines.append(f"final_t = nth_1 ({face_t_args})")
    else:
        expr_lines.append("final_t = huge")
    
    select_terms = []
    for f_idx in range(face_count):
        expr_lines.append(f"select_{f_idx} = abs(t_face_{f_idx} - final_t) < epsilon ? shading_face_{f_idx} : 0")
        select_terms.append(f"select_{f_idx}")
    selects_sum = " + ".join(select_terms)
    expr_lines.append(f"final_shading = final_t < huge ? ({selects_sum}) : background")
    expr_lines.append("RESULT = final_shading")
    
    full_expr = "\n".join(expr_lines)
    converted_expr = infix2postfix(full_expr)

    return clip.akarin.Expr(converted_expr)