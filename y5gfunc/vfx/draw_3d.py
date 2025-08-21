import math
from typing import Any, Literal, Tuple
import trimesh
from vstools import vs
from ..expr import DSLExpr, Constant, BuiltInFunc, ExprLike, infix2postfix
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
    points: list[dict[str, ExprLike]],
    faces: list[dict[str, Any]],
    lights: list[dict[str, ExprLike]],
    camX: ExprLike,
    camY: ExprLike,
    camZ: ExprLike,
    rotationX: ExprLike,
    rotationY: ExprLike,
    focal: ExprLike,
    background: ExprLike = 0,
) -> vs.VideoNode:
    """
    Renders a scene composed of triangles on a video clip using the Python expression interface.

    This function performs 3D rendering of a mesh of triangles with lighting and camera transformations.

    Args:
        clip: The video clip to render on.
        points: A list of dictionaries, each representing a vertex in the scene.
                Each dictionary should have "x", "y", and "z" keys with ExprLike values.
        faces: A list of dictionaries, each representing a triangle. Each should
                have "a", "b", "c" keys (indices to the points list) and an
                optional "color" key.
        lights: A list of dictionaries, each representing a directional (parallel) light.
                Each dictionary should have "lx", "ly", "lz" for direction and "intensity" as ExprLike values.
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
    from y5gfunc.expr import Constant, BuiltInFunc

    clip = core.std.BlankClip(width=640, height=480, format=vs.GRAYS, length=12000)

    orig_points = [
        {"x": -100, "y": -100, "z": 100},
        {"x": 100, "y": -100, "z": 100},
        {"x": 100, "y": 100, "z": 100},
        {"x": -100, "y": 100, "z": 100},
        {"x": -100, "y": -100, "z": -100},
        {"x": 100, "y": -100, "z": -100},
        {"x": 100, "y": 100, "z": -100},
        {"x": -100, "y": 100, "z": -100},
    ]

    transformed_points = []
    for pt in orig_points:
        new_pt = {
            "x": (pt['x'] * BuiltInFunc.cos(Constant.N * 0.02) - pt['z'] * BuiltInFunc.sin(Constant.N * 0.02) + 20 * BuiltInFunc.sin(Constant.N * 0.015)),
            "y": (pt['y'] + 20 * BuiltInFunc.cos(Constant.N * 0.015)),
            "z": (pt['x'] * BuiltInFunc.sin(Constant.N * 0.02) + pt['z'] * BuiltInFunc.cos(Constant.N * 0.02))
        }
        transformed_points.append(new_pt)

    faces = [
        {"a": 0, "b": 1, "c": 2, "color": 1},
        {"a": 0, "b": 2, "c": 3, "color": 1},
        {"a": 4, "b": 6, "c": 5, "color": 0.8},
        {"a": 4, "b": 7, "c": 6, "color": 0.8},
        {"a": 0, "b": 3, "c": 7, "color": 0.9},
        {"a": 0, "b": 7, "c": 4, "color": 0.9},
        {"a": 1, "b": 5, "c": 6, "color": 1.0},
        {"a": 1, "b": 6, "c": 2, "color": 1.0},
        {"a": 3, "b": 2, "c": 6, "color": 1.1},
        {"a": 3, "b": 6, "c": 7, "color": 1.1},
        {"a": 0, "b": 4, "c": 5, "color": 0.7},
        {"a": 0, "b": 5, "c": 1, "color": 0.7},
    ]

    lights = [
        {"lx": BuiltInFunc.cos(Constant.N * 0.02), "ly": 0.5, "lz": BuiltInFunc.sin(Constant.N * 0.02), "intensity": 0.8},
        {"lx": -0.5, "ly": 1, "lz": 0.5, "intensity": 0.6},
    ]

    camX_expr = 20 * BuiltInFunc.sin(241 * 0.015) + 500 * BuiltInFunc.cos(241 * 0.01)
    camY_expr = 20 * BuiltInFunc.cos(241 * 0.015) + 200 + 4 - 2 * BuiltInFunc.abs(Constant.N % 8 - 4)
    camZ_expr = 500 * BuiltInFunc.sin(241 * 0.01) - 4 - 2 * BuiltInFunc.abs(Constant.N % 8 - 4)

    rotationX_expr = 0.38 + (40 - 2 * BuiltInFunc.abs(Constant.N % 80 - 40)) / 500
    rotationY_expr = -2.41

    focal_expr = 500

    clip_result = render_triangle_scene(
        clip,
        points=transformed_points,
        faces=faces,
        lights=lights,
        camX=camX_expr,
        camY=camY_expr,
        camZ=camZ_expr,
        rotationX=rotationX_expr,
        rotationY=rotationY_expr,
        focal=focal_expr,
        background=0
    )
    ```
    """

    def _cam_coord(
        x: ExprLike,
        y: ExprLike,
        z: ExprLike,
        camX: ExprLike,
        camY: ExprLike,
        camZ: ExprLike,
        rotationX: ExprLike,
        rotationY: ExprLike,
        huge: ExprLike,
    ) -> Tuple[DSLExpr, DSLExpr, DSLExpr]:
        """Calculates the camera coordinates for a 3D point."""
        tx = x - camX
        ty = y - camY
        tz = z - camZ

        cos_rotX = BuiltInFunc.cos(rotationX)
        sin_rotX = BuiltInFunc.sin(rotationX)

        # Rotate around X-axis
        ty1 = ty * cos_rotX - tz * sin_rotX
        tz1 = ty * sin_rotX + tz * cos_rotX

        cos_rotY = BuiltInFunc.cos(rotationY)
        sin_rotY = BuiltInFunc.sin(rotationY)

        # Rotate around Y-axis
        cx = tx * cos_rotY + tz1 * sin_rotY
        cy = ty1
        cz_raw = -tx * sin_rotY + tz1 * cos_rotY

        # Clamp z to avoid points behind the camera causing issues
        cz = BuiltInFunc.if_then_else(cz_raw < 0, huge, cz_raw)

        return cx, cy, cz

    screenCenterX = Constant.width / 2
    screenCenterY = Constant.height / 2
    huge = 1e9
    ambient = 0.2

    # Transform light directions to camera space
    light_vectors = []
    for light in lights:
        raw_x, raw_y, raw_z = light["lx"], light["ly"], light["lz"]
        mag = BuiltInFunc.sqrt(raw_x**2 + raw_y**2 + raw_z**2)
        nx, ny, nz = raw_x / mag, raw_y / mag, raw_z / mag

        cos_rotX = BuiltInFunc.cos(rotationX)
        sin_rotX = BuiltInFunc.sin(rotationX)
        temp_ty = ny * cos_rotX - nz * sin_rotX
        temp_tz = ny * sin_rotX + nz * cos_rotX

        cos_rotY = BuiltInFunc.cos(rotationY)
        sin_rotY = BuiltInFunc.sin(rotationY)
        lx = nx * cos_rotY + temp_tz * sin_rotY
        ly = temp_ty
        lz = -nx * sin_rotY + temp_tz * cos_rotY
        light_vectors.append(
            {"lx": lx, "ly": ly, "lz": lz, "intensity": light["intensity"]}
        )

    # Transform points to camera and projected space
    cam_points = []
    proj_points = []
    for pt in points:
        x, y, z = pt["x"], pt["y"], pt["z"]
        cam_x, cam_y, cam_z = _cam_coord(
            x, y, z, camX, camY, camZ, rotationX, rotationY, huge
        )
        cam_points.append({"x": cam_x, "y": cam_y, "z": cam_z})

        proj_x = screenCenterX + (cam_x * focal) / cam_z
        proj_y = screenCenterY + (cam_y * focal) / cam_z
        proj_points.append({"x": proj_x, "y": proj_y})

    face_calcs = []
    for face in faces:
        a, b, c = face["a"], face["b"], face["c"]
        pA, pB, pC = proj_points[a], proj_points[b], proj_points[c]
        cA, cB, cC = cam_points[a], cam_points[b], cam_points[c]

        e0 = (Constant.X - pA["x"]) * (pB["y"] - pA["y"]) - (Constant.Y - pA["y"]) * (
            pB["x"] - pA["x"]
        )
        e1 = (Constant.X - pB["x"]) * (pC["y"] - pB["y"]) - (Constant.Y - pB["y"]) * (
            pC["x"] - pB["x"]
        )
        e2 = (Constant.X - pC["x"]) * (pA["y"] - pC["y"]) - (Constant.Y - pC["y"]) * (
            pA["x"] - pC["x"]
        )

        inside_pos = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        inside_neg = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
        valid = (cA["z"] < huge) & (cB["z"] < huge) & (cC["z"] < huge)
        inside = (inside_pos | inside_neg) & valid

        area = (pB["x"] - pA["x"]) * (pC["y"] - pA["y"]) - (pC["x"] - pA["x"]) * (
            pB["y"] - pA["y"]
        )
        alpha = (
            (pB["x"] - Constant.X) * (pC["y"] - Constant.Y)
            - (pC["x"] - Constant.X) * (pB["y"] - Constant.Y)
        ) / area
        beta = (
            (pC["x"] - Constant.X) * (pA["y"] - Constant.Y)
            - (pA["x"] - Constant.X) * (pC["y"] - Constant.Y)
        ) / area
        gamma = 1 - alpha - beta
        depth = alpha * cA["z"] + beta * cB["z"] + gamma * cC["z"]

        ex1, ey1, ez1 = cB["x"] - cA["x"], cB["y"] - cA["y"], cB["z"] - cA["z"]
        ex2, ey2, ez2 = cC["x"] - cA["x"], cC["y"] - cA["y"], cC["z"] - cA["z"]
        nx, ny, nz = ey1 * ez2 - ez1 * ey2, ez1 * ex2 - ex1 * ez2, ex1 * ey2 - ey1 * ex2
        norm = BuiltInFunc.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        sum_diffuse = ambient
        for light in light_vectors:
            dot = nx * light["lx"] + ny * light["ly"] + nz * light["lz"]
            sum_diffuse += BuiltInFunc.max(dot, 0) * light["intensity"]

        lighting = BuiltInFunc.min(sum_diffuse, 1)
        shading = lighting * face.get("color", 1)

        t_face = BuiltInFunc.if_then_else(inside, depth, huge)
        shading_face = BuiltInFunc.if_then_else(inside, shading, 0)
        face_calcs.append({"t": t_face, "shading": shading_face})

    if not faces:
        final_expr = (
            background
            if isinstance(background, DSLExpr)
            else (BuiltInFunc.abs(0) * 0 + background)
        )
    else:
        closest_t = face_calcs[0]["t"]
        closest_shading = face_calcs[0]["shading"]
        for i in range(1, len(faces)):
            is_closer = face_calcs[i]["t"] < closest_t
            closest_t = BuiltInFunc.if_then_else(
                is_closer, face_calcs[i]["t"], closest_t
            )
            closest_shading = BuiltInFunc.if_then_else(
                is_closer, face_calcs[i]["shading"], closest_shading
            )
        final_expr = BuiltInFunc.if_then_else(
            closest_t < huge, closest_shading, background
        )
    return clip.akarin.Expr(infix2postfix(final_expr.dsl))


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
    lights: list[dict[str, ExprLike]],
    camX: ExprLike,
    camY: ExprLike,
    camZ: ExprLike,
    rotationX: ExprLike,
    rotationY: ExprLike,
    focal: ExprLike,
    background: ExprLike = 0,
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
                "lx", "ly", "lz" for direction and "intensity" as ExprLike values.
        camX: The x-coordinate of the camera position as an ExprLike object.
        camY: The y-coordinate of the camera position as an ExprLike object.
        camZ: The z-coordinate of the camera position as an ExprLike object.
        rotationX: The camera rotation angle around the X-axis as an ExprLike object.
        rotationY: The camera rotation angle around the Y-axis as an ExprLike object.
        focal: The focal length for the projection as an ExprLike object.
        background: The background color as an ExprLike object.
        **mesh_kwargs: Additional keyword arguments for `load_mesh`.

    Returns:
        The video clip with the rendered 3D model.
    """
    points_str, faces_str_color = load_mesh(model_path, **mesh_kwargs)
    points: list[dict[str, ExprLike]] = [
        {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])} for p in points_str
    ]
    faces: list[dict[str, Any]] = []
    for face in faces_str_color:
        new_face = face.copy()
        if "color" in new_face:
            new_face["color"] = float(new_face["color"])
        faces.append(new_face)

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
