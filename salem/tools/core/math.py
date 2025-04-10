# noqa: A005
# https://saturncloud.io/blog/pythonnumpyscipy-converting-string-to-mathematical-function/


from salem.tools.runtime import runtime_callable


@runtime_callable
def solve(expr: str, var: str | None = None) -> str:
  """Calculate the given math expression in the simplified Python-like notation.

  Note: arbitrary Python functions are not allowed!
    Only numbers, math operands, round brackets and symbolic operands are supported.

  Example:
  ```python
    >> solve("x**2 + 3 * x - 1", var="x")
    "x = 9"
    >> solve("x**2 + 3 * x - 1")  # no var specified
    "x * (3 + x) - 1"  # simplified expression
    >> solve("2 ** 24 / 4")
    "4194304.0"  # concrete value
    >> solve("2*x + 4*y = 10", var="y")
    "y = 0.5 * (5 - x)"
  ```

  Args:
    expr: The math expression in simple Python notation, optionally with symbolic variables
    var: Optionally the target variable to solve the expression for, multiple variables not supported.
      If expression contains a variable, but it's not passed here, the result is a simplified expression

  Raises:
    ValueError: If the expression is poorly formated, or variable is incorrect in the given context

  Returns:
    The solution for the given math expression, potentially containing symbolic variables
  """
