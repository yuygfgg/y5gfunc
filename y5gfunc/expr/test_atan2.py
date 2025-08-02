from vapoursynth import core
import vapoursynth as vs
from math import atan2
from y5gfunc import *

def calc_atan2_expr(y, x):
    expr = f'{y} {x} dup 0 = dup2 0 < -1.5708 1.5708 ? dup2 0 = dup4 0 = and 0 dup4 0 = dup6 and dup6 0 < -1.5708 1.5708 ? dup6 0 < dup8 dup8 / 0 < dup9 dup9 / abs 1 > 1.5708 -2.02586e-05 dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.00116407 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.00918456 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0258268 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0409264 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0523923 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0666586 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.000223022 dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.00385597 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.016978 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0340678 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0467395 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0587731 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0769221 + + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.090909 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.111111 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.142857 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.2 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.333333 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? * dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? + -1 * + -2.02586e-05 dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.00116407 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.00918456 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0258268 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0409264 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0523923 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow 2 pow * -0.0666586 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.000223022 dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.00385597 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.016978 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0340678 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0467395 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0587731 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * 0.0769221 + + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.090909 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.111111 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.142857 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * 0.2 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * -0.333333 + dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? 2 pow * dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? * dup12 dup12 / abs 1 > 1 dup14 dup14 / abs / dup14 dup14 / abs ? + ? abs -1 * dup10 dup10 / abs 1 > 1.5708 -2.02586e-05 dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.00116407 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.00918456 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0258268 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0409264 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0523923 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0666586 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.000223022 dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.00385597 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.016978 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0340678 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0467395 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0587731 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0769221 + + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.090909 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.111111 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.142857 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.2 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.333333 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? * dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? + -1 * + -2.02586e-05 dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.00116407 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.00918456 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0258268 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0409264 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0523923 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow 2 pow * -0.0666586 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.000223022 dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.00385597 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.016978 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0340678 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0467395 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0587731 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * 0.0769221 + + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.090909 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.111111 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.142857 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * 0.2 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * -0.333333 + dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? 2 pow * dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? * dup13 dup13 / abs 1 > 1 dup15 dup15 / abs / dup15 dup15 / abs ? + ? abs ? dup9 0 < -3.14159 3.14159 ? + dup8 0 > dup10 dup10 / 0 < dup11 dup11 / abs 1 > 1.5708 -2.02586e-05 dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.00116407 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.00918456 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0258268 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0409264 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0523923 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0666586 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.000223022 dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.00385597 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.016978 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0340678 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0467395 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0587731 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0769221 + + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.090909 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.111111 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.142857 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.2 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.333333 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? * dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? + -1 * + -2.02586e-05 dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.00116407 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.00918456 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0258268 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0409264 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0523923 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow 2 pow * -0.0666586 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.000223022 dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.00385597 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.016978 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0340678 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0467395 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0587731 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * 0.0769221 + + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.090909 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.111111 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.142857 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * 0.2 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * -0.333333 + dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? 2 pow * dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? * dup14 dup14 / abs 1 > 1 dup16 dup16 / abs / dup16 dup16 / abs ? + ? abs -1 * dup12 dup12 / abs 1 > 1.5708 -2.02586e-05 dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.00116407 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.00918456 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0258268 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0409264 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0523923 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0666586 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.000223022 dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.00385597 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.016978 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0340678 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0467395 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0587731 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0769221 + + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.090909 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.111111 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.142857 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.2 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.333333 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? * dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? + -1 * + -2.02586e-05 dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.00116407 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.00918456 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0258268 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0409264 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0523923 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow 2 pow * -0.0666586 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.000223022 dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.00385597 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.016978 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0340678 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0467395 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0587731 + dup16 dup16 / abs 1 > 1 dup18 dup18 / abs / dup18 dup18 / abs ? 2 pow 2 pow * 0.0769221 + + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.090909 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.111111 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.142857 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * 0.2 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * -0.333333 + dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? 2 pow * dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? * dup15 dup15 / abs 1 > 1 dup17 dup17 / abs / dup17 dup17 / abs ? + ? abs ? 0 ? ? ? ? ? swap 0 * + swap 0 * +'
    akarin = get_pixel_value(core.akarin.Expr(core.std.BlankClip(format=vs.GRAYS), expr))
    std = get_pixel_value(core.std.Expr(core.std.BlankClip(format=vs.GRAYS), expr))
    py = atan2(y, x)
    max_diff = max(max(abs(akarin - std), abs(akarin - py)), abs(std - py))
    if_pass = max_diff < 1e-5
    print(f"atan({y}, {x}) => Akarin: {akarin}, Std: {std}, Python: {py}; max_diff: {max_diff} {'Pass' if if_pass else 'Fail'}")
    return if_pass, max_diff
    
    
def get_pixel_value(clip):
    frame = clip.get_frame(0)
    arr = frame[0]
    return arr[0,0]

def test_atan2():
    # Test with various y, x pairs
    success = fail = total = max_diff = 0
    test_cases = [
        (0, 1),   # atan2(0, 1) = 0
        (1, 0),   # atan2(1, 0) = π/2
        (0, -1),  # atan2(0, -1) = π
        (-1, 0),  # atan2(-1, 0) = -π/2
        (1, 1),   # atan2(1, 1) = π/4
        (-1, -1), # atan2(-1, -1) = -3π/4
        (1, -1),  # atan2(1, -1) = 3π/4
        (-1, 1),  # atan2(-1, 1) = -π/4
    ]
    
    for y, x in test_cases:
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    import random
    for _ in range(50):
        y = random.uniform(-1e8, 1e8)
        x = random.uniform(-1e8, 1e8)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(100):
        y = random.uniform(-2, 2)
        x = 0
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)
    
    for _ in range(100):
        y = 0
        x = random.uniform(-2, 2)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)
    
    for _ in range(100):
        y = random.uniform(-2, 2)
        x = random.uniform(-2, 2)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(50):
        y = random.uniform(-50, 50)
        x = random.uniform(-50, 50)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    print(f"Max difference: {max_diff}")
    print(f"Success: {success}, Fail: {fail}, Total: {total}")

test_atan2()

print("All tests completed.")