import unittest
import vapoursynth as vs
from vapoursynth import core
import math
import functools
import operator
import random

from y5gfunc.expr import (
    infix2postfix,
    emulate_expr,
    SourceClip,
    Constant,
    BuiltInFunc,
    python2infix,
    verify_akarin_expr,
    verify_std_expr,
    math_functions,
    OptimizeLevel,
)


def get_pixel_value(clip, x=0, y=0):
    """Helper to get a pixel value from a VapourSynth clip."""
    return clip.get_frame(0)[0][y, x]


class TestComprehensiveSuite(unittest.TestCase):
    """Master test suite for the entire y5gfunc.expr module."""

    def _evaluate_expr(self, expr_str, clips, use_std=False):
        """Evaluates a given expression string using the specified VapourSynth core."""
        if not clips:
            clips = [core.std.BlankClip()]
        if use_std:
            return get_pixel_value(core.std.Expr(clips, expr_str))
        return get_pixel_value(core.akarin.Expr(clips, expr_str))

    def _test_dsl_expression(
        self,
        infix_code,
        expected_result,
        clip_values=[0],
        use_std=False,
        optimize_level=4,
        places=5,
    ):
        """A generic, result-oriented test function for the DSL."""
        postfix_expr = infix2postfix(
            infix_code, force_std=use_std, optimize_level=OptimizeLevel(optimize_level)
        )
        self.assertTrue(
            verify_akarin_expr(postfix_expr), f"Verification failed for: {postfix_expr}"
        )
        if use_std:
            self.assertTrue(verify_std_expr(postfix_expr))

        clips = [core.std.BlankClip(format=vs.GRAYS, color=val) for val in clip_values]
        eval_result = self._evaluate_expr(postfix_expr, clips, use_std=use_std)
        self.assertTrue(
            math.isclose(eval_result, expected_result, rel_tol=1e-4, abs_tol=1e-5),
            msg=f"Failed on infix: {infix_code}, expected {expected_result}, got {eval_result}",
        )


class TestCoreDSL(TestComprehensiveSuite):
    def test_all_arithmetic_operators(self):
        self._test_dsl_expression("RESULT = 10 + 3", 13)
        self._test_dsl_expression("RESULT = 10 - 3", 7)
        self._test_dsl_expression("RESULT = 10 * 3", 30)
        self._test_dsl_expression(
            "RESULT = 10 / 4", 2.5, optimize_level=0
        )  # Test unoptimized
        self._test_dsl_expression(
            "RESULT = 10 / 4", 2.5, optimize_level=4
        )  # Test optimized
        self._test_dsl_expression("RESULT = 2 ** 8", 256)
        self._test_dsl_expression("RESULT = 10 % 3", 1)

    def test_all_logical_and_comparison_operators(self):
        self._test_dsl_expression("RESULT = (1 > 0) && (1 < 2)", 1)
        self._test_dsl_expression("RESULT = (1 > 0) || (1 > 2)", 1)
        self._test_dsl_expression("RESULT = !(1 > 2)", 1)
        self._test_dsl_expression("RESULT = 5 > 2", 1)
        self._test_dsl_expression("RESULT = 5 < 2", 0)
        self._test_dsl_expression("RESULT = 5 == 5", 1)
        self._test_dsl_expression("RESULT = 5 != 2", 1)

    def test_ternary_operator(self):
        self._test_dsl_expression("RESULT = 1 > 0 ? 42 : 99", 42)
        self._test_dsl_expression("RESULT = 1 < 0 ? 42 : 99", 99)

    def test_variable_scopes(self):
        infix = """
        <global<a><b>>
        function foo(c) {
            d = a + c
            return d
        }
        a = 10
        b = 20
        RESULT = foo(b)
        """
        self._test_dsl_expression(infix, 30)

    def test_pixel_access(self):
        # Create a clip where the value at (1,1) is 20, anywhere else is 10
        test_clip = vs.core.akarin.Expr(
            core.std.BlankClip(format=vs.GRAYS, color=10), "X 1 = Y 1 = and 20 10 ?"
        )

        self._test_dsl_expression("RESULT = $src0[1, 1]", 20, clip_values=[20])
        self._test_dsl_expression("RESULT = dyn($src0, 0, 0)", 10, [10])

        c = SourceClip(0)
        expr_static = c[1, 1]
        self.assertAlmostEqual(  # get value at pixel (0,0), so at offset (1,1) should be pixel (1,1), which is 20
            self._evaluate_expr(infix2postfix(expr_static.dsl), [test_clip]), 20
        )
        expr_static2 = c[0, 4]
        self.assertAlmostEqual(  # get value at pixel (0,0), so at offset (0,4) should be pixel (0,4), which is 10
            self._evaluate_expr(infix2postfix(expr_static2.dsl), [test_clip]), 10
        )

        expr_dyn = c.access(1, 1)
        self.assertAlmostEqual(
            self._evaluate_expr(infix2postfix(expr_dyn.dsl), [test_clip]), 20
        )

    def test_syntax_errors(self):
        with self.assertRaises(Exception, msg="Expected error for undefined variable"):
            infix2postfix("RESULT = my_undefined_var")
        with self.assertRaises(
            Exception, msg="Expected error for unbalanced parentheses"
        ):
            infix2postfix("RESULT = (1 + 2")


class TestMathFunctions(TestComprehensiveSuite):
    def test_working_math_functions(self):
        test_cases = {
            "atan2": (math.atan2, [1, 2]),  # fma, copysign, atan, atan2
            "cbrt": (math.cbrt, [8]),  # cbrt
            "erfc": (math.erfc, [1]),  # erfc
        }

        for name, (py_func, args) in test_cases.items():
            arg_str = ", ".join([f"$src{i}" for i in range(len(args))])
            infix = f"{math_functions}\nRESULT = {name}({arg_str})"
            expected = py_func(*args)
            with self.subTest(function=name):
                self._test_dsl_expression(infix, expected, args)


class TestMathFunctionsRandomized(TestComprehensiveSuite):
    """Randomized testing for math functions."""

    def _get_py_func(self, name):
        """Helper to get python equivalent function."""
        # Map to python's math module names
        py_name_map = {
            "arsinh": "asinh",
            "arcosh": "acosh",
            "artanh": "atanh",
        }
        py_name = py_name_map.get(name, name)

        if hasattr(math, py_name):  # pyright: ignore[reportArgumentType]
            return getattr(math, py_name)  # pyright: ignore[reportArgumentType]

        # Custom implementations for functions not in standard math
        py_funcs = {
            "abs": abs,
            "fma": lambda x, y, z: x * y + z,
            "cot": lambda x: 1 / math.tan(x),
            "sec": lambda x: 1 / math.cos(x),
            "csc": lambda x: 1 / math.sin(x),
            "acot": lambda x: math.pi / 2 - math.atan(x),
            "coth": lambda x: 1 / math.tanh(x),
            "sech": lambda x: 1 / math.cosh(x),
            "csch": lambda x: 1 / math.sinh(x),
            "arcoth": lambda x: 0.5 * math.log((x + 1) / (x - 1)),
            "arsech": lambda x: math.log((1 + math.sqrt(1 - x**2)) / x),
            "arcsch": lambda x: math.log(1 / x + math.sqrt(1 / x**2 + 1)),
        }
        return py_funcs.get(name)

    def _generate_args(self, name, num_args):
        """Generate random arguments with valid domains."""
        args = []
        for _ in range(num_args):
            if name in ["asin", "acos", "artanh"]:
                args.append(random.uniform(-1.0, 1.0))
            elif name == "arcosh":
                args.append(random.uniform(1.0, 100.0))
            elif name == "arcoth":
                val = random.uniform(1.0001, 100.0)
                if random.choice([True, False]):
                    val = -val
                args.append(val)
            elif name == "arsech":
                args.append(random.uniform(0.0001, 1.0))
            elif name in ["tan", "sinh", "cosh", "arsinh", "cot", "sec", "csc"]:
                args.append(random.uniform(-10.0, 10.0))
            else:
                args.append(random.uniform(-100.0, 100.0))
        return args

    def test_randomized_math_functions(self):
        """Exhaustively test math functions with random values."""
        functions_to_test = {
            # name: num_args
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
            "atan2": 2,
            "sinh": 1,
            "cosh": 1,
            "tanh": 1,
            "arsinh": 1,
            "arcosh": 1,
            "artanh": 1,
            "fma": 3,
            "copysign": 2,
            "erf": 1,
            "erfc": 1,
            "cbrt": 1,
            "cot": 1,
            "sec": 1,
            "csc": 1,
            "acot": 1,
            "coth": 1,
            "sech": 1,
            "csch": 1,
            "arcoth": 1,
            "arsech": 1,
            "arcsch": 1,
        }

        for name, num_args in functions_to_test.items():
            py_func = self._get_py_func(name)
            if not py_func:
                self.skipTest(f"Python equivalent for '{name}' not found.")
                continue

            with self.subTest(function=name):
                for _ in range(100):  # 100 random tests per function
                    args = self._generate_args(name, num_args)

                    try:
                        expected = py_func(*args)
                    except (ValueError, ZeroDivisionError):
                        print(f"Skipping test case for {name} with args {args}")
                        continue

                    arg_str = ", ".join([f"$src{i}" for i in range(len(args))])
                    infix = f"{math_functions}\nRESULT = {name}({arg_str})"

                    self._test_dsl_expression(infix, expected, args, places=3)


class TestOptimizer(TestComprehensiveSuite):
    def test_optimizer_correctness(self):
        infix = """
        a = $src0 * 2
        b = a + $src1
        c = b / 3
        RESULT = c - ($src2 + 1)
        """
        clip_values = [10, 5, 2]
        unoptimized_postfix = infix2postfix(infix, optimize_level=OptimizeLevel.O0)
        expected_result = self._evaluate_expr(
            unoptimized_postfix,
            [core.std.BlankClip(format=vs.GRAYS, color=v) for v in clip_values],
        )

        for level in range(1, 5):
            with self.subTest(level=level):
                optimized_postfix = infix2postfix(
                    infix, optimize_level=OptimizeLevel(level)
                )
                optimized_result = self._evaluate_expr(
                    optimized_postfix,
                    [core.std.BlankClip(format=vs.GRAYS, color=v) for v in clip_values],
                )
                self.assertAlmostEqual(
                    optimized_result,
                    expected_result,
                    places=5,
                    msg=f"Optimizer level {level} failed",
                )


class TestPythonInterface(TestComprehensiveSuite):
    def test_all_interface_features(self):
        x, y, z = SourceClip(0), SourceClip(1), SourceClip(2)

        expr = BuiltInFunc.sqrt(BuiltInFunc.abs((x * Constant.pi) - (y / 2.0))) + z
        expr = BuiltInFunc.clamp(expr, 0, 255)
        expr = BuiltInFunc.if_then_else(x > y, expr, z)

        clips = [core.std.BlankClip(format=vs.GRAYS, color=v) for v in [50, 20, 10]]
        postfix = infix2postfix(expr.dsl)
        result = self._evaluate_expr(postfix, clips)

        py_result = math.sqrt(abs((50 * math.pi) - (20 / 2.0))) + 10
        py_result = max(0, min(255, py_result))
        self.assertAlmostEqual(result, py_result, places=5)

    def test_pixel_access_interface(self):
        """Tests the various ways to access pixels via the Python interface."""
        c = SourceClip(0)

        with python2infix.varname_toggle(True):
            static_expr = c[1, 1]
            self.assertIn("$src0[1,1]", static_expr.dsl)

            dyn_int_expr = c.access(0, 0)
            self.assertIn("dyn($src0, 0, 0)", dyn_int_expr.dsl)

            dyn_expr_expr = c.access(Constant.X + 1, Constant.Y - 1)
            self.assertIn(
                "v_0 = ($X + 1)\nv_1 = ($Y - 1)\ndyn_expr_expr_0 = dyn($src0, v_0, v_1)\nRESULT = dyn_expr_expr_0",
                dyn_expr_expr.dsl,
            )

        with self.assertRaises(
            TypeError, msg="Static access should not allow expressions"
        ):
            _ = c[Constant.X, 1]
        with self.assertRaises(
            TypeError, msg="Static access should not allow expressions"
        ):
            _ = c[1, Constant.Y]
        with self.assertRaises(TypeError, msg="Static access requires a tuple of two"):
            _ = c[1]


class TestStdConversion(TestComprehensiveSuite):
    def test_std_conversion_correctness(self):
        self._test_dsl_expression(
            "RESULT = $src0 + $src1 - $src2", 12, [10, 5, 3], use_std=True
        )
        self._test_dsl_expression(
            "var1 = $src0 + $pi\nvar2 = $src1 / $src2\nvar3 = sin($src3)\nRESULT = nth_2(var1, var2, var3, 4442)",
            emulate_expr(
                infix2postfix(
                    "var1 = $src0 + $pi\nvar2 = $src1 / $src2\nvar3 = sin($src3)\nRESULT = nth_2(var1, var2, var3, 4442)"
                ),
                clip_value={"src0": 10, "src1": 5, "src2": 3, "src3": 8},
            ),
            [10, 5, 3, 8],
            use_std=True,
        )

    def test_std_incompatibility_errors(self):
        with self.assertRaises(Exception):
            infix2postfix("RESULT = round(1.5)", force_std=True)
        with self.assertRaises(Exception):
            infix2postfix("RESULT = 10 % 3", force_std=True)


class TestEndToEndWorkflows(TestComprehensiveSuite):
    def test_gaussian_blur_workflow(self):
        """Build a 3x3 Gaussian blur using the Python interface and evaluate it."""
        c = SourceClip(0)

        p = [[c[x, y] for x in [-1, 0, 1]] for y in [-1, 0, 1]]

        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        terms = [p[y][x] * kernel[y][x] for y in range(3) for x in range(3)]
        weighted_sum = functools.reduce(operator.add, terms)

        total_weight = sum(sum(row) for row in kernel)

        blur_expr = weighted_sum / total_weight

        test_clip = core.std.BlankClip(width=10, height=10, format=vs.GRAYS, color=100)
        postfix = infix2postfix(blur_expr.dsl)
        result_clip = core.akarin.Expr([test_clip], postfix)

        # The result at (5,5) should be 100 because all surrounding pixels are 100
        self.assertAlmostEqual(get_pixel_value(result_clip, 5, 5), 100, places=3)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCoreDSL))
    suite.addTests(loader.loadTestsFromTestCase(TestMathFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMathFunctionsRandomized))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPythonInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestStdConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflows))

    runner = unittest.TextTestRunner()
    runner.run(suite)
