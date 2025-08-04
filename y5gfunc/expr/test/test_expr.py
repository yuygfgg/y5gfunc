import unittest
import vapoursynth as vs
from vapoursynth import core
import math
import functools
import operator

from y5gfunc.expr import (
    infix2postfix,
    emulate_expr,
    SourceClip,
    Constant,
    BuiltInFunc,
    verify_akarin_expr,
    verify_std_expr,
    math_functions,
    OptimizeLevel,
)


def get_pixel_value(clip):
    """Helper to get the top-left pixel value from a VapourSynth clip."""
    return clip.get_frame(0)[0][0, 0]


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
        self.assertAlmostEqual(
            eval_result, expected_result, places=5, msg=f"Failed on infix: {infix_code}"
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
        self._test_dsl_expression("RESULT = $src0[1, 1]", 10, [10, 20])
        self._test_dsl_expression("RESULT = dyn($src0, 0, 0)", 10, [10])

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
        expr = BuiltInFunc.clamp(expr, 0, 255)  # pyright: ignore[reportArgumentType]
        expr = BuiltInFunc.if_then_else(x > y, expr, z)

        clips = [core.std.BlankClip(format=vs.GRAYS, color=v) for v in [50, 20, 10]]
        postfix = infix2postfix(expr.dsl)
        result = self._evaluate_expr(postfix, clips)

        py_result = math.sqrt(abs((50 * math.pi) - (20 / 2.0))) + 10
        py_result = max(0, min(255, py_result))
        self.assertAlmostEqual(result, py_result, places=5)


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

        p = [
            [c.access(x, y) for x in [-1, 0, 1]]  # pyright: ignore[reportArgumentType]
            for y in [-1, 0, 1]
        ]

        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        terms = [p[y][x] * kernel[y][x] for y in range(3) for x in range(3)]
        weighted_sum = functools.reduce(operator.add, terms)

        total_weight = sum(sum(row) for row in kernel)

        blur_expr = weighted_sum / total_weight

        test_clip = core.std.BlankClip(width=10, height=10, format=vs.GRAYS, color=100)
        postfix = infix2postfix(blur_expr.dsl)
        result_clip = core.akarin.Expr([test_clip], postfix)

        self.assertAlmostEqual(result_clip.get_frame(0)[0][5, 5], 100, places=3)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCoreDSL))
    suite.addTests(loader.loadTestsFromTestCase(TestMathFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPythonInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestStdConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflows))

    runner = unittest.TextTestRunner()
    runner.run(suite)
