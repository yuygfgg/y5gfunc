math_functions: str = """
# These functions works, but some are really slow......

function fma(var1, var2, var3) {
    return var1 * var2 + var3
}

function copysign(val, sign) {
    return (sign < 0) ? -abs(val) : abs(val)
}

# https://stackoverflow.com/a/23097989
function atan(var) {
    zz = abs(var)
    aa = (zz > 1) ? (1 / zz) : zz
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
    return copysign (rr, var)
}

function atan2(var_y, var_x) {
    theta = 0
    theta = (var_x > 0) ? atan(var_y / var_x) : theta
    theta = (var_x < 0) ? (atan(var_y / var_x) + copysign(pi, var_y)) : theta
    theta = ((var_x == 0) && var_y) ? (copysign(pi / 2, var_y)) : theta
    theta = ((var_x == 0) && (var_y == 0)) ? 0 : theta
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
    est = abs_var ? exp(log(abs_var) / 3) : var
    est = copysign(est, var)
    cube = est ** 3 
    return abs_var? fma(-est, ((cube - var) / (2 * cube + var)), est) : var
}

# https://github.com/kravietz/nist-sts/blob/master/erf.c
function erf(var) {
    abs_var = abs(var)
    tt = 1 / (1 + 0.3275911 * abs_var)
    yy = 1 - (((((1.061405429 * tt + -1.453152027) * tt) + 1.421413741) * tt + -0.284496736) * tt + 0.254829592) * tt * exp(-abs_var * abs_var)
    return copysign(yy, var)
}

# https://github.com/kravietz/nist-sts/blob/master/erf.c
function erfc(var) {
	zz = abs(var)
	tt = 1 / (1 + 0.5 * zz)
	ans = tt * exp(-zz * zz - 1.26551223 + tt * (1.00002368 + tt * (0.37409196 + tt * (0.09678418 + tt * (-0.18628806 + tt * (0.27886807 + tt * (-1.13520398 + tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277)))))))))
	return var >= 0 ? ans : 2 - ans
}

function gamma_sign(var_z) {
    is_pole = (var_z <= 0) && (floor(var_z) == var_z)
    use_reflection = (var_z < 0.5) && !is_pole
    sign_direct = 1
    sin_pi_var_z = sin(pi * var_z)
    sign_reflected = (sin_pi_var_z >= 0) ? 1 : -1
    calc_sign = use_reflection ? sign_reflected : sign_direct
    return is_pole ? 1 : calc_sign
}

function lgamma_val(var_z) {
    is_pole = (var_z <= 0) && (floor(var_z) == var_z)
    use_reflection = (var_z < 0.5) && !is_pole
    var_z_calc = use_reflection ? (1 - var_z) : var_z
    var_z_minus_1 = var_z_calc - 1
    tmp = var_z_minus_1 + 7 + 0.5
    series = 0.99999999999980993227684700473478
    series = series + 676.52036812188509856700919044401 / (var_z_minus_1 + 1)
    series = series + -1259.1392167224028704715607875528 / (var_z_minus_1 + 2)
    series = series + 771.32342877765307884893049653757 / (var_z_minus_1 + 3)
    series = series + -176.61502916214059906584551354002 / (var_z_minus_1 + 4)
    series = series + 12.507343278686904814458936853287 / (var_z_minus_1 + 5)
    series = series + -0.13857109526572011689554707118956 / (var_z_minus_1 + 6)
    series = series + 9.9843695780195708595638234185252e-6 / (var_z_minus_1 + 7)
    series = series + 1.5056327351493115583497230996790e-7 / (var_z_minus_1 + 8)
    log_series = log(max(series, 1e9))
    log_gamma_core = 0.5 * log(2 * pi) + (var_z_calc - 0.5) * log(max(tmp, 1e9)) - tmp + log_series
    sin_pi_var_z = sin(pi * var_z)
    log_abs_sin = log(max(abs(sin_pi_var_z), 1e9))
    reflection_adjustment = log(pi) - log_abs_sin
    final_log_gamma = use_reflection ? (reflection_adjustment - log_gamma_core) : log_gamma_core
    return is_pole ? 1e9 : final_log_gamma
}

function tgamma(var_z) {
    sign = gamma_sign(var_z)
    lgamma_val = lgamma_val(var_z)
    is_inf = (lgamma_val >= 1e9)
    abs_gamma = exp(lgamma_val)
    signed_gamma = sign * abs_gamma
    return is_inf ? (sign * 1e9) : signed_gamma
}
"""
