math_functions = """
function fma(var1, var2, var3) {
    return var1 * var2 + var3
}

function copysign(a1, a2) {
    absx = (a1 < 0) ? -a1 : a1
    return (a2 < 0) ? -absx : absx
}

# https://stackoverflow.com/a/23097989
function atan(xx) {
    zz = abs(xx)
    aa = (zz > 1) ? (1 / zz): zz
    ss = aa ** 2
    qq = ss ** 2
    pp =             -2.0258553044340116e-5
    tt =              2.2302240345710764e-4
    pp = fma(pp, qq, -1.1640717779912220e-3)
    tt = fma(tt, qq,  3.8559749383656407e-3)
    pp = fma(pp, qq, -9.1845592187222193e-3)
    tt = fma(tt, qq,  1.6978035834594660e-2)
    pp = fma(pp, qq, -2.5826796814492296e-2)
    tt = fma(tt, qq,  3.4067811082715810e-2)
    pp = fma(pp, qq, -4.0926382420509999e-2)
    tt = fma(tt, qq,  4.6739496199158334e-2)
    pp = fma(pp, qq, -5.2392330054601366e-2)
    tt = fma(tt, qq,  5.8773077721790683e-2)
    pp = fma(pp, qq, -6.6658603633512892e-2)
    tt = fma(tt, qq,  7.6922129305867892e-2)
    pp = fma(pp, ss,                     tt)
    pp = fma(pp, ss, -9.0909012354005267e-2)
    pp = fma(pp, ss,  1.1111110678749421e-1)
    pp = fma(pp, ss, -1.4285714271334810e-1)
    pp = fma(pp, ss,  1.9999999999755005e-1)
    pp = fma(pp, ss, -3.3333333333331838e-1)
    pp = fma(pp * ss, aa, aa)
    rr = (zz > 1) ? fma(0.93282184640716537, 1.6839188885261840, -pp): pp
    return copysign (rr, xx)
}

function atan2(var_y, var_x) {
    theta = 0
    theta = (var_x > 0.0) ? atan(var_y / var_x) : theta
    theta = (var_x < 0.0) ? (atan(var_y / var_x) + copysign(pi, var_y)) : theta
    theta = ((var_x == 0.0) && (var_y != 0.0)) ? (copysign(pi / 2.0, var_y)) : theta
    theta = ((var_x == 0.0) && (var_y == 0.0)) ? 0.0 : theta
    return theta
}

function cot(var) {
    return cos(var) / sin(var)
}

function sec(var) {
    return 1 / cos(var)
}

function csc(var) {
    return 1 / sin(var)
}

function asin(var) {
    return 2 * atan(var / (1 + sqrt(1 - var ** 2)))
}

function acos(var) {
    return 2 * atan(sqrt((1 - var) / (1 + var)))
}

function acot(var) {
    return pi / 2 - atan(var)
}

function sinh(var) {
    return (exp(var) - exp(-var)) / 2
}

function cosh(var) {
    return (exp(var) + exp(-var)) / 2
}

function tanh(var) {
    return (exp(2 * var) - 1) / (exp(2 * var) + 1)
}

function coth(var) {
    return (exp(2 * var) + 1) / (exp(2 * var) - 1)
}

function sech(var) {
    return 1 / cosh(var)
}

function csch(var) {
    return 1 / sinh(var)
}

function arsinh(var) {
    return log(var + sqrt(var ** 2 + 1))
}

function arcosh(var) {
    return log(var + sqrt(var ** 2 - 1))
}

function artanh(var) {
    return 0.5 * log((1 + var) / (1 - var))
}

function arcoth(var) {
    return 0.5 * log((var + 1) / (var - 1))
}

function arsech(var) {
    return log((1 + sqrt(1 - var ** 2)) / var)
}

function arcsch(var) {
    return log(1 / var + sqrt(1 / (var ** 2) + 1))
}

# https://stackoverflow.com/a/77465269
function cbrt(var) {
    abs_var = abs(var)
    est = (abs_var != 0.0) ? exp(log(abs_var) / 3.0) : var
    est = copysign(est, var)
    cube = est ** 3 
    return abs_var != 0.0 ? fma(-est, ((cube - var) / (2.0 * cube + var)), est) : var
}
"""
