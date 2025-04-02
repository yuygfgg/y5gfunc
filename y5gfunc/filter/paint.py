from vstools import core
from vstools import vs
from ..expr import infix2postfix


def draw_line(clip: vs.VideoNode, sx: int, sy: int, ex: int, ey: int, thickness: float, color: float, factor: float = 1.0) -> vs.VideoNode: 
    assert all(0 <= coord <= clip.height for coord in [sy, ey])
    assert all(0 <= coord <= clip.width for coord in [sx, ex])
    assert 0 < factor <= 1
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

def draw_circle(clip: vs.VideoNode, cx: int, cy: int, radius: float, thickness: float, color: float, factor: float = 1.0) -> vs.VideoNode:
    assert 0 <= cx <= clip.width
    assert 0 <= cy <= clip.height
    assert 0 < factor <= 1
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

def draw_ellipse(clip: vs.VideoNode, f1x: int, f1y: int, f2x: int, f2y: int, ellipse_sum: float, thickness: float, color: float, factor: float = 1.0) -> vs.VideoNode:
    assert 0 <= f1x <= clip.width
    assert 0 <= f1y <= clip.height
    assert 0 <= f2x <= clip.width
    assert 0 <= f2y <= clip.height
    assert thickness > 0
    assert 0 < factor <= 1
    assert clip.format.num_planes == 1
    
    foci_distance = ((f2x - f1x) ** 2 + (f2y - f1y) ** 2) ** 0.5
    assert ellipse_sum > foci_distance
    
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