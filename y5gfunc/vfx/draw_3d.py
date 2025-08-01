import math
from typing import Literal
import trimesh
from vstools import vs
from ..expr import infix2postfix
import numpy as np


def draw_3d_polyhedron(
    clip: vs.VideoNode,
    shape: Literal["cube", "icosahedron", "tetrahedron", "octahedron", "dodecahedron"],
    centerX: str,
    centerY: str,
    size: str,
    color: str,
    rotationX: str,
    rotationY: str,
    thickness: str,
    translateZ: str = "500",
    focal: str = "500",
    factor: str = "1",
) -> vs.VideoNode:
    """
    Draw a 3D regular polyhedron on a video clip.

    Args:
        clip: The video clip to draw the polyhedron on.
        shape: The shape of the polyhedron.
        centerX: The x-coordinate of the center of the projection screen.
        centerY: The y-coordinate of the center of the projection screen.
        size: The size of the polyhedron.
        color: The color of the polyhedron's edges.
        rotationX: The rotation angle around the X-axis (in radians).
        rotationY: The rotation angle around the Y-axis (in radians).
        thickness: The thickness of the polyhedron's edges.
        translateZ: The translation along the Z-axis, affecting perspective.
        focal: The focal length for the projection.
        factor: The blending factor for the color.

    Returns:
        The video clip with the 3D polyhedron drawn on it.

    Raises:
        AssertionError: If the format of the video clip has more than 1 plane.
    """
    assert clip.format.num_planes == 1

    PHI = (1 + math.sqrt(5)) / 2
    PSI = 1 / PHI

    _POLYHEDRA_DATA = {
        "cube": {
            "vertices": [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            "edges": [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
        },
        "icosahedron": {
            "vertices": [
                (-1, "phi", 0),
                (1, "phi", 0),
                (-1, "-phi", 0),
                (1, "-phi", 0),
                (0, -1, "phi"),
                (0, 1, "phi"),
                (0, -1, "-phi"),
                (0, 1, "-phi"),
                ("phi", 0, -1),
                ("phi", 0, 1),
                ("-phi", 0, -1),
                ("-phi", 0, 1),
            ],
            "edges": [
                (0, 1),
                (0, 5),
                (0, 7),
                (0, 10),
                (0, 11),
                (1, 5),
                (1, 7),
                (1, 8),
                (1, 9),
                (2, 3),
                (2, 4),
                (2, 6),
                (2, 10),
                (2, 11),
                (3, 4),
                (3, 6),
                (3, 8),
                (3, 9),
                (4, 5),
                (4, 9),
                (4, 11),
                (5, 9),
                (5, 11),
                (6, 7),
                (6, 8),
                (6, 10),
                (7, 8),
                (7, 10),
                (8, 9),
                (10, 11),
            ],
        },
        "tetrahedron": {
            "vertices": [
                (1, 1, 1),
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
            ],
            "edges": [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            ],
        },
        "octahedron": {
            "vertices": [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ],
            "edges": [
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
            ],
        },
        "dodecahedron": {
            "vertices": [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
                (0, "psi", "phi"),
                (0, "psi", "-phi"),
                (0, "-psi", "phi"),
                (0, "-psi", "-phi"),
                ("phi", 0, "psi"),
                ("phi", 0, "-psi"),
                ("-phi", 0, "psi"),
                ("-phi", 0, "-psi"),
                ("psi", "phi", 0),
                ("psi", "-phi", 0),
                ("-psi", "phi", 0),
                ("-psi", "-phi", 0),
            ],
            "edges": [
                (0, 8),
                (0, 12),
                (0, 16),
                (1, 9),
                (1, 13),
                (1, 16),
                (2, 10),
                (2, 12),
                (2, 17),
                (3, 11),
                (3, 13),
                (3, 17),
                (4, 8),
                (4, 14),
                (4, 18),
                (5, 9),
                (5, 14),
                (5, 18),
                (6, 10),
                (6, 15),
                (6, 19),
                (7, 11),
                (7, 15),
                (7, 19),
                (8, 18),
                (9, 18),
                (10, 19),
                (11, 19),
                (12, 17),
                (13, 17),
                (14, 15),
            ],
        },
    }

    data = _POLYHEDRA_DATA[shape]
    vertices_coords = data["vertices"]
    edges = data["edges"]

    expr_parts = [
        f"""
        centerX = {centerX}
        centerY = {centerY}
        size = {size}
        rotationX = {rotationX}
        rotationY = {rotationY}
        translateZ = {translateZ}
        focal = {focal}
        thickness = {thickness}
        color = {color}
        factor = {factor}
        
        halfSize = size / 2
        phi = {PHI}
        psi = {PSI}
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
            segLenSq = dx ** 2 + dy ** 2
            tt = (($X - x0) * dx + ($Y - y0) * dy) / segLenSq
            t_clamped = clamp(tt, 0, 1)
            projX = x0 + t_clamped * dx
            projY = y0 + t_clamped * dy
            return ($X - projX) ** 2 + ($Y - projY) ** 2
        }}
        """
    ]

    for i, (x, y, z) in enumerate(vertices_coords):
        expr_parts.append(f"v{i}x = ({x}) * halfSize")
        expr_parts.append(f"v{i}y = ({y}) * halfSize")
        expr_parts.append(f"v{i}z = ({z}) * halfSize")

    for i in range(len(vertices_coords)):
        expr_parts.append(f"v{i}projX = project3d_x(v{i}x, v{i}y, v{i}z)")
        expr_parts.append(f"v{i}projY = project3d_y(v{i}x, v{i}y, v{i}z)")

    dist_vars = []
    for i, (v1, v2) in enumerate(edges):
        dist_var = f"d{i}"
        dist_vars.append(dist_var)
        expr_parts.append(
            f"{dist_var} = distSqToSegment(v{v1}projX, v{v1}projY, v{v2}projX, v{v2}projY)"
        )

    min_expr = dist_vars[0]
    for i in range(1, len(dist_vars)):
        min_expr = f"min({min_expr}, {dist_vars[i]})"
    expr_parts.append(f"finalMinDistanceSquared = {min_expr}")

    expr_parts.append(
        """
        halfThickness = thickness / 2
        halfThicknessSq = halfThickness ** 2
        doDraw = finalMinDistanceSquared <= halfThicknessSq
        
        RESULT = doDraw ? ((1 - factor) * $src0 + factor * color) : $src0
    """
    )

    full_expr = "\n".join(expr_parts)
    expr = infix2postfix(full_expr)

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
    background: str = "0",
) -> vs.VideoNode:
    """
    Renders a scene composed of triangles on a video clip.

    This function performs 3D rendering of a mesh of triangles with lighting and camera transformations.

    Args:
        clip: The video clip to render on.
        points: A list of dictionaries, each representing a vertex in the scene.
                Each dictionary should have "x", "y", and "z" keys.
        faces: A list of dictionaries, each representing a triangle. Each should
               have "a", "b", "c" keys (indices to the points list) and an
               optional "color" key.
        lights: A list of dictionaries, each representing a directional (parallel) light.
                This type of light has a direction but no position, and its
                intensity does not decay with distance. Each dictionary should have
                "lx", "ly", "lz" for direction and "intensity".
        camX: The camera's X position.
        camY: The camera's Y position.
        camZ: The camera's Z position.
        rotationX: The camera's rotation around the X-axis.
        rotationY: The camera's rotation around the Y-axis.
        focal: The camera's focal length.
        background: The background color value.

    Returns:
        A video clip with the 3D scene rendered on it.

    Example:

    ```python
    clip = core.std.BlankClip(width=640, height=480, format=vs.GRAYS, length=12000)

    orig_points = [
        { "x" : "-100", "y" : "-100", "z" : "100"  },
        { "x" : "100",  "y" : "-100", "z" : "100"  },
        { "x" : "100",  "y" : "100",  "z" : "100"  },
        { "x" : "-100", "y" : "100",  "z" : "100"  },
        { "x" : "-100", "y" : "-100", "z" : "-100" },
        { "x" : "100",  "y" : "-100", "z" : "-100" },
        { "x" : "100",  "y" : "100",  "z" : "-100" },
        { "x" : "-100", "y" : "100",  "z" : "-100" }
    ]

    transformed_points = []
    for pt in orig_points:
        new_pt = {
            "x": f"(({pt['x']}) * cos($N * 0.02) - ({pt['z']}) * sin($N * 0.02) + 20 * sin($N * 0.015))",
            "y": f"(({pt['y']}) + 20 * cos($N * 0.015))",
            "z": f"(({pt['x']}) * sin($N * 0.02) + ({pt['z']}) * cos($N * 0.02))"
        }
        transformed_points.append(new_pt)

    faces = [
        { "a" : 0, "b" : 1, "c" : 2, "color" : "1"   },
        { "a" : 0, "b" : 2, "c" : 3, "color" : "1"   },
        { "a" : 4, "b" : 6, "c" : 5, "color" : "0.8" },
        { "a" : 4, "b" : 7, "c" : 6, "color" : "0.8" },
        { "a" : 0, "b" : 3, "c" : 7, "color" : "0.9" },
        { "a" : 0, "b" : 7, "c" : 4, "color" : "0.9" },
        { "a" : 1, "b" : 5, "c" : 6, "color" : "1.0" },
        { "a" : 1, "b" : 6, "c" : 2, "color" : "1.0" },
        { "a" : 3, "b" : 2, "c" : 6, "color" : "1.1" },
        { "a" : 3, "b" : 6, "c" : 7, "color" : "1.1" },
        { "a" : 0, "b" : 4, "c" : 5, "color" : "0.7" },
        { "a" : 0, "b" : 5, "c" : 1, "color" : "0.7" }
    ]

    lights = [
        { "lx" : "cos($N * 0.02)", "ly" : "0.5", "lz" : "sin($N * 0.02)", "intensity" : "0.8" },
        { "lx" : "-0.5",          "ly" : "1",   "lz" : "0.5",           "intensity" : "0.6" }
    ]

    camX = "20 * sin(241 * 0.015) + 500 * cos(241 * 0.01)"
    camY = "20 * cos(241 * 0.015) + 200 + 4 - 2 * abs($N % 8 - 4)"
    camZ = "500 * sin(241 * 0.01) - 4 - 2 * abs($N % 8 - 4)"

    rotationX = "0.38 + (40 - 2 * abs($N % 80 - 40)) / 500"
    rotationY = "-2.41"

    focal = "500"

    clip_result = render_triangle_scene(
        clip,
        points=transformed_points,
        faces=faces,
        lights=lights,
        camX=camX,
        camY=camY,
        camZ=camZ,
        rotationX=rotationX,
        rotationY=rotationY,
        focal=focal,
        background="0"
    )
    ```

    Renders a rotating and vibrating cubic.
    """

    expr_lines = []

    expr_lines.append(
        "<global<camX><camY><camZ><rotationX><rotationY><focal><screenCenterX><screenCenterY><epsilon>>"
    )
    expr_lines.append("function cam_coord_x(x, y, z) {")
    expr_lines.append("    tx = x - camX")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    ty1 = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    tz1 = ty * sin(rotationX) + tz * cos(rotationX)")
    expr_lines.append("    cx = tx * cos(rotationY) + tz1 * sin(rotationY)")
    expr_lines.append("    return cx")
    expr_lines.append("}")

    expr_lines.append(
        "<global<camX><camY><camZ><rotationX><rotationY><focal><screenCenterX><screenCenterY><epsilon>>"
    )
    expr_lines.append("function cam_coord_y(x, y, z) {")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    cy = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    return cy")
    expr_lines.append("}")

    expr_lines.append("<global<camX><camY><camZ><rotationX><rotationY><epsilon><huge>>")
    expr_lines.append("function cam_coord_z(x, y, z) {")
    expr_lines.append("    tx = x - camX")
    expr_lines.append("    ty = y - camY")
    expr_lines.append("    tz = z - camZ")
    expr_lines.append("    ty1 = ty * cos(rotationX) - tz * sin(rotationX)")
    expr_lines.append("    tz1 = ty * sin(rotationX) + tz * cos(rotationX)")
    expr_lines.append("    cz = -tx * sin(rotationY) + tz1 * cos(rotationY)")
    expr_lines.append("    return cz < 0 ? huge : cz")
    expr_lines.append("}")

    expr_lines.append("<global<focal><screenCenterX>>")
    expr_lines.append("function projectX(x, y, z) {")
    expr_lines.append("    px = cam_coord_x(x, y, z)")
    expr_lines.append("    pz = cam_coord_z(x, y, z)")
    expr_lines.append("    return screenCenterX + (px * focal) / pz")
    expr_lines.append("}")

    expr_lines.append("<global<focal><screenCenterY>>")
    expr_lines.append("function projectY(x, y, z) {")
    expr_lines.append("    py = cam_coord_y(x, y, z)")
    expr_lines.append("    pz = cam_coord_z(x, y, z)")
    expr_lines.append("    return screenCenterY + (py * focal) / pz")
    expr_lines.append("}")

    expr_lines.append(f"camX = {camX}")
    expr_lines.append(f"camY = {camY}")
    expr_lines.append(f"camZ = {camZ}")
    expr_lines.append(f"rotationX = {rotationX}")
    expr_lines.append(f"rotationY = {rotationY}")
    expr_lines.append(f"focal = {focal}")
    expr_lines.append(f"background = {background}")
    expr_lines.append("screenCenterX = $width / 2")
    expr_lines.append("screenCenterY = $height / 2")
    expr_lines.append("epsilon = 0.0001")
    expr_lines.append("huge = 1e9")
    expr_lines.append("ambient = 0.2")

    for idx, light in enumerate(lights):
        expr_lines.append(f"light{idx}_raw_x = {light['lx']}")
        expr_lines.append(f"light{idx}_raw_y = {light['ly']}")
        expr_lines.append(f"light{idx}_raw_z = {light['lz']}")
        expr_lines.append(f"light{idx}_intensity = {light['intensity']}")
        expr_lines.append(
            f"mag_{idx} = sqrt(light{idx}_raw_x ** 2 + light{idx}_raw_y ** 2 + light{idx}_raw_z ** 2)"
        )
        expr_lines.append(f"light{idx}_nx = light{idx}_raw_x / mag_{idx}")
        expr_lines.append(f"light{idx}_ny = light{idx}_raw_y / mag_{idx}")
        expr_lines.append(f"light{idx}_nz = light{idx}_raw_z / mag_{idx}")
        expr_lines.append(
            f"temp_ty_{idx} = light{idx}_ny * cos(rotationX) - light{idx}_nz * sin(rotationX)"
        )
        expr_lines.append(
            f"temp_tz_{idx} = light{idx}_ny * sin(rotationX) + light{idx}_nz * cos(rotationX)"
        )
        expr_lines.append(
            f"light{idx}_lx = light{idx}_nx * cos(rotationY) + temp_tz_{idx} * sin(rotationY)"
        )
        expr_lines.append(f"light{idx}_ly = temp_ty_{idx}")
        expr_lines.append(
            f"light{idx}_lz = -light{idx}_nx * sin(rotationY) + temp_tz_{idx} * cos(rotationY)"
        )

    for i, pt in enumerate(points):
        expr_lines.append(f"point{i}_x = {pt['x']}")
        expr_lines.append(f"point{i}_y = {pt['y']}")
        expr_lines.append(f"point{i}_z = {pt['z']}")
        expr_lines.append(f"projX_{i} = projectX(point{i}_x, point{i}_y, point{i}_z)")
        expr_lines.append(f"projY_{i} = projectY(point{i}_x, point{i}_y, point{i}_z)")
        expr_lines.append(
            f"cam_x_{i} = cam_coord_x(point{i}_x, point{i}_y, point{i}_z)"
        )
        expr_lines.append(
            f"cam_y_{i} = cam_coord_y(point{i}_x, point{i}_y, point{i}_z)"
        )
        expr_lines.append(
            f"cam_z_{i} = cam_coord_z(point{i}_x, point{i}_y, point{i}_z)"
        )

    face_t_names = []
    face_shading_names = []
    face_count = len(faces)
    for f_idx, face in enumerate(faces):
        a_idx = face["a"]
        b_idx = face["b"]
        c_idx = face["c"]
        face_color = face.get("color", "1")
        expr_lines.append(
            f"E0_{f_idx} = ($X - projX_{a_idx}) * (projY_{b_idx} - projY_{a_idx}) - ($Y - projY_{a_idx}) * (projX_{b_idx} - projX_{a_idx})"
        )
        expr_lines.append(
            f"E1_{f_idx} = ($X - projX_{b_idx}) * (projY_{c_idx} - projY_{b_idx}) - ($Y - projY_{b_idx}) * (projX_{c_idx} - projX_{b_idx})"
        )
        expr_lines.append(
            f"E2_{f_idx} = ($X - projX_{c_idx}) * (projY_{a_idx} - projY_{c_idx}) - ($Y - projY_{c_idx}) * (projX_{a_idx} - projX_{c_idx})"
        )
        expr_lines.append(
            f"inside_pos_{f_idx} = E0_{f_idx} >= 0 && E1_{f_idx} >= 0 && E2_{f_idx} >= 0"
        )
        expr_lines.append(
            f"inside_neg_{f_idx} = E0_{f_idx} <= 0 && E1_{f_idx} <= 0 && E2_{f_idx} <= 0"
        )
        expr_lines.append(
            f"valid_{f_idx} = (cam_z_{a_idx} < huge) && (cam_z_{b_idx} < huge) && (cam_z_{c_idx} < huge)"
        )
        expr_lines.append(
            f"inside_{f_idx} = (inside_pos_{f_idx} || inside_neg_{f_idx}) && valid_{f_idx}"
        )

        expr_lines.append(
            f"area_{f_idx} = (projX_{b_idx} - projX_{a_idx}) * (projY_{c_idx} - projY_{a_idx}) - (projX_{c_idx} - projX_{a_idx}) * (projY_{b_idx} - projY_{a_idx})"
        )
        expr_lines.append(
            f"alpha_{f_idx} = ((projX_{b_idx} - $X) * (projY_{c_idx} - $Y) - (projX_{c_idx} - $X) * (projY_{b_idx} - $Y)) / area_{f_idx}"
        )
        expr_lines.append(
            f"beta_{f_idx} = ((projX_{c_idx} - $X) * (projY_{a_idx} - $Y) - (projX_{a_idx} - $X) * (projY_{c_idx} - $Y)) / area_{f_idx}"
        )
        expr_lines.append(f"gamma_{f_idx} = 1 - alpha_{f_idx} - beta_{f_idx}")
        expr_lines.append(
            f"depth_{f_idx} = alpha_{f_idx} * cam_z_{a_idx} + beta_{f_idx} * cam_z_{b_idx} + gamma_{f_idx} * cam_z_{c_idx}"
        )

        expr_lines.append(f"ex1_{f_idx} = cam_x_{b_idx} - cam_x_{a_idx}")
        expr_lines.append(f"ey1_{f_idx} = cam_y_{b_idx} - cam_y_{a_idx}")
        expr_lines.append(f"ez1_{f_idx} = cam_z_{b_idx} - cam_z_{a_idx}")
        expr_lines.append(f"ex2_{f_idx} = cam_x_{c_idx} - cam_x_{a_idx}")
        expr_lines.append(f"ey2_{f_idx} = cam_y_{c_idx} - cam_y_{a_idx}")
        expr_lines.append(f"ez2_{f_idx} = cam_z_{c_idx} - cam_z_{a_idx}")
        expr_lines.append(
            f"nx_{f_idx} = ey1_{f_idx} * ez2_{f_idx} - ez1_{f_idx} * ey2_{f_idx}"
        )
        expr_lines.append(
            f"ny_{f_idx} = ez1_{f_idx} * ex2_{f_idx} - ex1_{f_idx} * ez2_{f_idx}"
        )
        expr_lines.append(
            f"nz_{f_idx} = ex1_{f_idx} * ey2_{f_idx} - ey1_{f_idx} * ex2_{f_idx}"
        )
        expr_lines.append(
            f"norm_{f_idx} = sqrt(nx_{f_idx} ** 2 + ny_{f_idx} ** 2 + nz_{f_idx} ** 2)"
        )
        expr_lines.append(f"nx_{f_idx} = nx_{f_idx} / norm_{f_idx}")
        expr_lines.append(f"ny_{f_idx} = ny_{f_idx} / norm_{f_idx}")
        expr_lines.append(f"nz_{f_idx} = nz_{f_idx} / norm_{f_idx}")

        for l_idx in range(len(lights)):
            dot_expr = f"nx_{f_idx} * light{l_idx}_lx + ny_{f_idx} * light{l_idx}_ly + nz_{f_idx} * light{l_idx}_lz"
            expr_lines.append(f"dot_{f_idx}_{l_idx} = {dot_expr}")
            expr_lines.append(f"diffuse_{f_idx}_{l_idx} = max(dot_{f_idx}_{l_idx}, 0)")
            expr_lines.append(
                f"contrib_{f_idx}_{l_idx} = diffuse_{f_idx}_{l_idx} * light{l_idx}_intensity"
            )

        expr_lines.append(f"sum_diffuse_{f_idx} = ambient")
        for l_idx in range(len(lights)):
            expr_lines.append(
                f"sum_diffuse_{f_idx} = sum_diffuse_{f_idx} + contrib_{f_idx}_{l_idx}"
            )

        expr_lines.append(
            f"lighting_{f_idx} = sum_diffuse_{f_idx} > 1 ? 1 : sum_diffuse_{f_idx}"
        )

        expr_lines.append(f"faceColor_{f_idx} = {face_color}")
        expr_lines.append(f"shading_{f_idx} = lighting_{f_idx} * faceColor_{f_idx}")

        expr_lines.append(
            f"t_face_{f_idx} = inside_{f_idx} == 1 ? depth_{f_idx} : huge"
        )
        expr_lines.append(
            f"shading_face_{f_idx} = inside_{f_idx} == 1 ? shading_{f_idx} : 0"
        )

        face_t_names.append(f"t_face_{f_idx}")
        face_shading_names.append(f"shading_face_{f_idx}")

    if face_count > 0:
        expr_lines.append("closest_t_0 = t_face_0")
        expr_lines.append("closest_shading_0 = shading_face_0")

        for i in range(1, face_count):
            prev_i = i - 1
            expr_lines.append(f"is_closer_{i} = t_face_{i} < closest_t_{prev_i}")
            expr_lines.append(
                f"closest_t_{i} = is_closer_{i} ? t_face_{i} : closest_t_{prev_i}"
            )
            expr_lines.append(
                f"closest_shading_{i} = is_closer_{i} ? shading_face_{i} : closest_shading_{prev_i}"
            )

        final_idx = face_count - 1
        expr_lines.append(f"final_t = closest_t_{final_idx}")
        expr_lines.append(f"final_shading = closest_shading_{final_idx}")
        expr_lines.append("RESULT = final_t < huge ? final_shading : background")
    else:
        expr_lines.append("RESULT = background")

    full_expr = "\n".join(expr_lines)
    converted_expr = infix2postfix(full_expr)

    return clip.akarin.Expr(converted_expr)


def load_mesh(
    file_path: str,
    default_color: str = "1",
    axis_transform: str = "+xz-y",
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[list[dict], list[dict]]:
    """
    Load a 3D model from a file and apply transformations.

    Args:
        file_path: The path to the model file.
        default_color: The default color for the model's faces.
        axis_transform: The axis transformation to apply. Supports "+xz-y" and "xyz".
        rotation: A tuple of (rx, ry, rz) rotation angles in degrees.

    Returns:
        A tuple containing the list of points and the list of faces.
    """
    mesh = trimesh.load_mesh(file_path, force="mesh", process=True)

    if axis_transform == "+xz-y":
        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
        mesh.vertices[:, 2] *= -1
    elif axis_transform == "xyz":
        pass

    if rotation != (0.0, 0.0, 0.0):
        rx, ry, rz = [np.deg2rad(angle) for angle in rotation]
        R_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        R_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        R_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
        R = trimesh.transformations.concatenate_matrices(R_z, R_y, R_x)
        mesh.apply_transform(R)

    points = [
        {"x": f"{v[0]:.6f}", "y": f"{v[1]:.6f}", "z": f"{v[2]:.6f}"}
        for v in mesh.vertices
    ]

    faces = []
    for face in mesh.faces:
        faces.append(
            {
                "a": int(face[0]),
                "b": int(face[1]),
                "c": int(face[2]),
                "color": default_color,
            }
        )

    return points, faces


def render_model_scene(
    clip: vs.VideoNode,
    model_path: str,
    lights: list,
    camX: str,
    camY: str,
    camZ: str,
    rotationX: str,
    rotationY: str,
    focal: str,
    background: str = "0",
    **mesh_kwargs,
) -> vs.VideoNode:
    """
    Render a 3D model scene.

    Args:
        clip: The video clip to render on.
        model_path: The path to the 3D model file.
        lights: A list of dictionaries, each representing a **directional (parallel) light**.
                This type of light has a direction but no position, and its
                intensity does not decay with distance. Each dictionary should have
                "lx", "ly", "lz" for direction and "intensity".
        camX: The x-coordinate of the camera position.
        camY: The y-coordinate of the camera position.
        camZ: The z-coordinate of the camera position.
        rotationX: The camera rotation angle around the X-axis.
        rotationY: The camera rotation angle around the Y-axis.
        focal: The focal length for the projection.
        background: The background color.
        **mesh_kwargs: Additional keyword arguments for `load_mesh`.

    Returns:
        The video clip with the rendered 3D model.
    """
    points, faces = load_mesh(model_path, **mesh_kwargs)

    return render_triangle_scene(
        clip=clip,
        points=points,
        faces=faces,
        lights=lights,
        camX=camX,
        camY=camY,
        camZ=camZ,
        rotationX=rotationX,
        rotationY=rotationY,
        focal=focal,
        background=background,
    )
