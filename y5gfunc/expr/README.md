# VapourSynth Expr Tools

This module provides a comprehensive suite of functions and classes to facilitate the writing, conversion, and optimization of VapourSynth expressions. It introduces three distinct representations for `Expr` code, each catering to different needs and levels of abstraction.

## Core Representations

The module revolves around three representations of `Expr` code:

1.  **Postfix (Reverse Polish Notation - RPN)**: This is the native syntax for VapourSynth's `std.Expr`. It is a stack-based language that is powerful but can be difficult to write and debug by hand.

2.  **Infix**: A human-readable, C-style Domain-Specific Language (DSL). It supports variables, standard operators, and user-defined functions, offering a more intuitive way to write expressions while maintaining a close-to-the-metal relationship with the underlying Postfix code.

3.  **Python Interface**: A high-level Python interface that allows you to construct expressions using familiar Python syntax. This representation abstracts away the complexities of both Postfix and Infix, enabling the use of IDE features like type checking and autocompletion.

## Key Features

Based on these representations, the module offers the following functionalities:

### 1. Code Conversion & Transpilation

-   **Python to Infix**: Transpile expressions written in Python into the Infix DSL, allowing for a more programmatic and type-safe approach.
-   **Infix to Postfix**: Transpile the readable Infix DSL into the native Postfix RPN required by VapourSynth. Includes syntax validation to catch errors early.
-   **Postfix to Infix**: Decompile existing Postfix expressions back into the human-readable Infix DSL, making it easier to understand, modify, and debug complex RPN code.

### 2. Expression Optimization

A powerful optimizer that applies a variety of techniques to enhance `Expr` performance:
-   **Constant Folding**: Pre-calculates constant parts of the expression.
-   **Dead Assignment Elimination**: Removes variables that are assigned but never used.
-   **Dynamic to Static Pixel Access Conversion**: Converts dynamic pixel access (e.g., `x[a,b]`) to faster static access (e.g., `x[1,1]`) where possible.

### 3. Compatibility & Transformation

-   **Reduce Akarin Plugin Dependency**: Transforms expressions to emulate certain `akarin.Expr`-only features (like `sort`, `dropN`, and variable storage) using `std.Expr` compatible operations. This significantly reduces the need for external plugins.

### 4. Evaluation & Debugging

-   **Expression Emulator**: A powerful tool to test and debug expressions directly in Python, without a VapourSynth environment. It can evaluate standard Postfix expressions and fully emulate the behavior of `akarin.Expr` by allowing you to provide values for special constants (e.g., `$N`, `$X`) and mock frame properties.

### 5. Extended Functionality

-   **Infix DSL Math Library**: A collection of additional mathematical functions (e.g., `atan2`, `cbrt`, `erf`, `tgamma`) are provided as Infix user-defined functions, extending the capabilities of the standard `Expr` language.

## Documentation

For detailed information on each component, please refer to the documentation:

-   **[Infix DSL Syntax](./docs/infix.md)**: A complete guide to the C-style Infix language.
-   **[Postfix Syntax (std.Expr and akarin.Expr)](./docs/postfix.md)**: A reference for the RPN syntax used by VapourSynth.
-   **[Python Interface](./docs/python_interface.md)**: Documentation on how to use the Python-based expression builder.
