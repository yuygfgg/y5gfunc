### VapourSynth Expr Syntax Documentation

`Expr` is a powerful VapourSynth function that evaluates a per-pixel mathematical or logical expression. Its core is the `expr` string, which uses Reverse Polish Notation (RPN). This guide details the syntax of that string.

---

### **1. Core Concepts**

#### **1.1. Reverse Polish Notation (RPN)**

Instead of the conventional `A + B`, RPN places operators *after* their operands: `A B +`. The expression is evaluated using a stack. Values are pushed onto the stack, and operators pop values, perform a calculation, and push the result back onto the stack.

*   **Example:** To calculate `(5 + 3) * 2`, the RPN expression is `5 3 + 2 *`.
    1.  `5`: Push 5. Stack: `[5]`
    2.  `3`: Push 3. Stack: `[5, 3]`
    3.  `+`: Pop 3 and 5, calculate 8, push 8. Stack: `[8]`
    4.  `2`: Push 2. Stack: `[8, 2]`
    5.  `*`: Pop 2 and 8, calculate 16, push 16. Stack: `[16]`

The final value on the stack is the result for the pixel.

#### **1.2. Data Ranges**

`Expr` does not normalize input clip values. You must account for the native range of the pixel format.
*   **8-bit integer:** 0 to 255
*   **10-bit integer:** 0 to 1023
*   **16-bit integer:** 0 to 65535
*   **32-bit float:** Typically 0.0 to 1.0 for Luma (Y) and Alpha, and -0.5 to 0.5 for Chroma (U/V).

When mixing formats, scale values accordingly. For example, to add an 8-bit value to a 10-bit value, you might use `x y 4 * +` (multiplying the 8-bit value `y` by 4 to scale it to the 10-bit range).

---

### **2. `std.Expr` Base Syntax**

This section covers the standard operators and identifiers available in `std.Expr`.

#### **2.1. Clip Identifiers**

Input clips are referred to by letters. Up to 26 clips can be used.
*   `x`: The first input clip.
*   `y`: The second input clip.
*   `z`: The third input clip.
*   `a` to `w`: The 4th to 26th clips.

#### **2.2. Constants**

Numeric literals are pushed directly onto the stack.
*   **Example:** `x 128 -` (Subtracts 128 from each pixel value of the first clip).

#### **2.3. Arithmetic Operators (2 operands)**

*   `+`: Addition
*   `-`: Subtraction
*   `*`: Multiplication
*   `/`: Division

#### **2.4. Math Functions**

*   `pow`: (2 operands) `x y pow` raises `x` to the power of `y`.
*   `exp`: (1 operand) `x exp` is e^x.
*   `log`: (1 operand) `x log` is the natural logarithm of `x`.
*   `sqrt`: (1 operand) `x sqrt` is the square root of `x`.
*   `abs`: (1 operand) `x abs` is the absolute value of `x`.
*   `sin`: (1 operand) `x sin` is the sine of x.
*   `cos`: (1 operand) `x cos` is the cosine of x.

#### **2.5. Comparison & Logical Operators (2 operands)**

These operators treat any value greater than 0 as `true`. They return `1.0` for true and `0.0` for false.
*   `>`: Greater than
*   `<`: Less than
*   `=`: Equal to
*   `>=`: Greater than or equal to
*   `<=`: Less than or equal to
*   `and`: Logical AND
*   `or`: Logical OR
*   `xor`: Logical XOR
*   `not`: (1 operand) Logical NOT.

#### **2.6. Conditional Operator (3 operands)**

*   `?`: A ternary operator, `C A B ?` is equivalent to `C ? A : B`. If `C` is true (non-zero), `A` is evaluated and its result is pushed. Otherwise, `B` is evaluated and its result is pushed.
    *   **Example:** `x 128 > x 0 ?` (If the pixel value is greater than 128, keep it, otherwise set it to 0).

#### **2.7. Min/Max Operators (2 operands)**

*   `max`: Returns the larger of the two values.
*   `min`: Returns the smaller of the two values.

#### **2.8. Stack Manipulation Operators**

*   `dup`: (1 operand) Duplicates the top item on the stack. `x dup *` is equivalent to `x x *` or `x 2 pow`.
*   `swap`: (2 operands) Swaps the top two items on the stack. `x y swap -` is equivalent to `y x -`.
*   `dupN`: Duplicates the item at N positions from the top of the stack. `dup0` is `dup`.
*   `swapN`: Swaps the top item with the item N positions down the stack. `swap1` is `swap`.

---

### **3. `akarin.Expr` Extended Syntax**

`akarin.Expr` includes all `std.Expr` functionality plus the following additions.

#### **3.1. Special Variables & Constants**

These are special operators that push a specific value onto the stack.
*   `pi`: Pushes the mathematical constant π (approximately 3.14159).
*   `N`: Pushes the current frame number.
*   `X`: Pushes the current pixel's column coordinate (width).
*   `Y`: Pushes the current pixel's row coordinate (height).
*   `width`: Pushes the width of the frame.
*   `height`: Pushes the height of the frame.

#### **3.2. Frame Property Access**

*   `clip.PropertyName`: Loads a scalar numerical frame property.
    *   **Example:** `x.PlaneStatsAverage` pushes the average pixel value of the current plane from the first clip. If the property doesn't exist, the value is `NaN`.

#### **3.3. Additional Math & Logic**

*   `**`: An alias for `pow`. `x 2 **` is the same as `x 2 pow`.
*   `%`: Implements C's `fmodf`. `x 1.0 %` gives the fractional part of `x`.
*   `clip` / `clamp`: (3 operands) Clamps a value. `x min max clip` is equivalent to `x max min`.
    *   **Example:** `x 16 235 clip` clamps the pixel value to the range.
*   `trunc`: (1 operand) Truncates the value towards zero.
*   `round`: (1 operand) Rounds the value to the nearest integer.
*   `floor`: (1 operand) Rounds the value down to the nearest integer.

#### **3.4. Advanced Stack Manipulation**

*   **Named Variables:**
    *   `var!`: Pops the top value from the stack and stores it in a variable named `var`.
    *   `var@`: Pushes the value of the variable `var` onto the stack.
    *   **Example:** `x 2 / my_var! my_var@ my_var@ *` (Calculates `(x/2)^2`).
*   `dropN`: Drops the top `N` items from the stack. `drop` is an alias for `drop1`.
    *   **Example:** `1 2 3 drop2` results in a stack of `[1]`.
*   `sortN`: Sorts the top `N` items on the stack, with the smallest value ending up on top.
    *   **Example:** `3 1 2 sort3` results in a stack of `[3, 2, 1]`, with `1` at the top.

#### **3.5. Pixel Access**

*   **Static Relative Access:** `clip[relX, relY]`
    *   Accesses a pixel relative to the current coordinate. `relX` and `relY` must be constants.
    *   **Example:** `y[-1, 0]` accesses the pixel to the immediate left in the second clip (`y`).
    *   **Boundary Suffixes:**
        *   `:c`: Clamped boundary (edge pixels are repeated). This is the default. `x[relX, relY]:c`.
        *   `:m`: Mirrored boundary. `x[relX, relY]:m`.
*   **Dynamic Absolute Access:** `absX absY clip[]`
    *   Accesses a pixel at an absolute coordinate. `absX` and `absY` can be computed by expressions. This is more flexible but potentially slower.
    *   **Example:** `X 2 / Y x[]` reads the pixel at half the current X coordinate from the first clip.

> Note: `X 2 + Y 3 - x[]` is equal to `x[2,-3]`.

#### **3.6. Bitwise Operators**

These operators work on integers. Set `opt=1` in the function parameters to ensure integer evaluation for clips with more than 24 bits.
*   `bitand`: Bitwise AND.
*   `bitor`: Bitwise OR.
*   `bitxor`: Bitwise XOR.
*   `bitnot`: Bitwise NOT.

#### **3.7. Extended Numeric Formats**

*   **Hexadecimal:** `0x10` (16), `0xFF` (255).
*   **Octal:** `010` (8). Note that invalid octal numbers like `09` are parsed as floats (`9.0`).

#### **3.8. Arbitrary Number of Clips**

*   `srcN`: Accesses the N-th input clip (0-indexed).
    *   `src0` is equivalent to `x`.
    *   `src1` is equivalent to `y`.
    *   `src26` accesses the 27th clip.