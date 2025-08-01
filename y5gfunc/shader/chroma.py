from .lazy_loader import LazyVariable

cfl_shader = LazyVariable(
    url="https://raw.githubusercontent.com/Artoriuz/glsl-chroma-from-luma-prediction/refs/heads/main/CfL_Prediction.glsl"
)
KrigBilateral = LazyVariable(
    url="https://raw.githubusercontent.com/awused/dotfiles/refs/heads/master/mpv/.config/mpv/shaders/KrigBilateral.glsl"
)
