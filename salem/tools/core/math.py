# noqa: A005
# https://saturncloud.io/blog/pythonnumpyscipy-converting-string-to-mathematical-function/
import re

import sympy

from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.str import StrPrinter

from salem.tools.runtime import runtime_callable


def _can_convert_to_integer(v: sympy.NumberSymbol) -> bool:
  if not isinstance(v, sympy.Number):
    return False
  return v.is_Integer or v.is_Float and sympy.floor(v).equals(v)


@runtime_callable
def solve(expr: str, var: str | int | float | None = None) -> str:
  """Calculate the given math expression in the simplified Python-like notation.

  Note: arbitrary Python functions are not allowed!
    Only numbers, math operands, round brackets and symbolic operands are supported.
    Supports basic functions like `sqrt`, `exp`, `sin`, `cos`, `abs`, `log`, `ln` and constants like `E` and `pi`.

  Example:
  ```python
    >> solve("x**2 + 3 * x - 1", var=2)
    "x = 9"  # solve for particular x
    >> solve("x**2 + 3 * x - 1")  # no var specified
    "x * (3 + x) - 1"  # simplified expression
    >> solve("2 ** 24 / 4")
    "4194304"  # concrete value
    >> solve("2*x + 4*y = 10", var="y")
    "y = 0.5 * (5 - x)"
  ```

  Args:
    expr: The math expression in simple Python notation, optionally with symbolic variables
    var: The target variable to solve the expression for or its value.
      If expression contains a variable, but `var` is `None`, the result is a simplified expression.
      Multiple variables not supported!

  Raises:
    ValueError: If the expression is poorly formated, or variable is incorrect in the given context

  Returns:
    The solution for the given math expression, potentially containing symbolic variables
  """

  printer = StrPrinter({"full_prec": False, "order": "lex"})

  restricted_globals = {
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "pi": sympy.pi,
    "E": sympy.E,
    "Symbol": sympy.Symbol,
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "Function": sympy.Function,
  }

  parsed_obj: sympy.Basic
  is_equation_expr = "=" in expr

  if is_equation_expr:
    if expr.count("=") > 1:
      raise ValueError("Expression must contain exactly one '=' to be an equation.")
    lhs_str, rhs_str = expr.split("=", 1)
    try:
      all_potential_vars = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lhs_str + " " + rhs_str))
      known_globals_keys = set(restricted_globals.keys())
      symbols_in_expr_dict = {
        name: sympy.Symbol(name, real=True) for name in all_potential_vars if name not in known_globals_keys
      }
      parsing_context = {**restricted_globals, **symbols_in_expr_dict}
      lhs = parse_expr(lhs_str, global_dict=parsing_context, local_dict=symbols_in_expr_dict, evaluate=True)
      rhs = parse_expr(rhs_str, global_dict=parsing_context, local_dict=symbols_in_expr_dict, evaluate=True)
      parsed_obj = sympy.Eq(lhs, rhs, evaluate=False)
    except Exception as e:
      raise ValueError(f"Incorrectly formatted equation: '{expr}'. Error: {e}")
  else:
    try:
      parsed_obj = parse_expr(expr, global_dict=restricted_globals, evaluate=True)
    except Exception as e:
      raise ValueError(f"Incorrect expression (unsupported symbols/functions?): '{expr}'. Error: {e}")

  if isinstance(var, (int, float)):
    if is_equation_expr:
      raise ValueError("Substitution mode (when 'var' is a number) is not supported for expressions containing '='.")
    if not isinstance(parsed_obj, sympy.Expr):
      raise ValueError(f"Cannot substitute into non-expression type: {type(parsed_obj)}")
    free_symbols = list(parsed_obj.free_symbols)
    if not free_symbols:
      raise ValueError("Expression has no variables to substitute.")
    if len(free_symbols) > 1:
      raise ValueError("Expression must contain exactly one variable for substitution.")
    symbol_to_sub = free_symbols[0]
    try:
      substituted_expr = parsed_obj.subs(symbol_to_sub, sympy.sympify(var))
      if substituted_expr.has(sympy.pi, sympy.E) and not substituted_expr.is_number:

        def floatize_int_coeff(node):
          if node.is_Mul and any(arg.has(sympy.pi, sympy.E) for arg in node.args):
            new_args = [sympy.Float(arg) if arg.is_Integer else arg for arg in node.args]
            return sympy.Mul(*new_args, evaluate=False)
          return node

        processed_value = substituted_expr.replace(lambda n: n.is_Mul, floatize_int_coeff)
        # Handle E**1 correctly
        if substituted_expr.func == sympy.Pow and substituted_expr.base == sympy.E and substituted_expr.exp == 1:
          result_str = printer.doprint(sympy.E)
        else:
          result_str = printer.doprint(processed_value)

      elif not substituted_expr.free_symbols:
        evaluated_value = substituted_expr.evalf()
        if _can_convert_to_integer(evaluated_value):  # Using your original function name
          result_str = printer.doprint(sympy.Integer(evaluated_value))
        else:
          result_str = printer.doprint(evaluated_value)
      else:
        result_str = printer.doprint(substituted_expr)
      return f"{symbol_to_sub.name} = {result_str}"
    except Exception as e:
      raise ValueError(f"Error during substitution or evaluation: {e}")

  elif var is None:
    simplified_expr = sympy.simplify(parsed_obj)
    if isinstance(simplified_expr, sympy.Equality):
      return f"{printer.doprint(simplified_expr.lhs)} = {printer.doprint(simplified_expr.rhs)}"
    if not simplified_expr.free_symbols:
      if isinstance(simplified_expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
        return printer.doprint(simplified_expr)
      if simplified_expr.is_Integer:  # Test wants "1" for sin^2+cos^2
        return printer.doprint(simplified_expr)
      if simplified_expr.is_Number:  # For 2**24/4 -> 4194304 (if test expect int string)
        # If test expects "4194304.0", this needs custom formatting.
        # Assuming test is adjusted to expect "4194304" based on "stick to integers"
        num_val_evalf = simplified_expr.evalf()
        if _can_convert_to_integer(num_val_evalf):
          return printer.doprint(sympy.Integer(num_val_evalf))
        return printer.doprint(num_val_evalf)  # For 0.5 etc.
      return printer.doprint(simplified_expr)  # e.g. pi
    else:
      if (
        len(simplified_expr.free_symbols) == 1
        and isinstance(simplified_expr, sympy.Expr)
        and not isinstance(simplified_expr, sympy.Rel)
      ):  # Relational deprecated, use Rel
        single_var = list(simplified_expr.free_symbols)[0]
        if simplified_expr.is_polynomial(single_var):
          try:
            expanded_for_horner = sympy.expand(simplified_expr)
            horner_form = sympy.horner(expanded_for_horner, single_var)  # Positional arg
            if horner_form != simplified_expr and horner_form != expanded_for_horner:
              return printer.doprint(horner_form)
          except Exception:
            pass
      return printer.doprint(simplified_expr)

  elif isinstance(var, str):
    try:
      target_symbol = sympy.Symbol(var, real=True)
    except ValueError:
      raise ValueError(f"Invalid variable name '{var}' for solving.")
    equation_to_solve: sympy.Basic
    if isinstance(parsed_obj, sympy.Equality):
      equation_to_solve = parsed_obj
    elif isinstance(parsed_obj, sympy.Expr):
      equation_to_solve = sympy.Eq(parsed_obj, 0, evaluate=False)  # evaluate=False crucial
    elif isinstance(parsed_obj, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
      equation_to_solve = parsed_obj
    else:
      raise ValueError(f"Cannot form equation from input '{expr}' (type: {type(parsed_obj)}).")
    if not isinstance(equation_to_solve, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
      if target_symbol not in equation_to_solve.free_symbols:
        raise ValueError(f"Variable '{var}' not found in the equation derived from '{expr}'.")
    try:
      solutions = sympy.solve(equation_to_solve, target_symbol)
    except NotImplementedError as e:
      raise ValueError(f"Cannot solve for '{var}': SymPy cannot solve this equation type. Detail: {e}")
    except Exception as e:
      raise ValueError(f"Error solving for '{var}': {e}")
    if not solutions:
      return f"No solution found for {var}"
    output_parts = []
    for sol_expr in solutions:
      if not sol_expr.free_symbols:
        evaluated_sol = sol_expr.evalf()
        if _can_convert_to_integer(evaluated_sol):  # Using your original function name
          num_str = printer.doprint(sympy.Integer(evaluated_sol))
        else:
          num_str = printer.doprint(evaluated_sol)
        output_parts.append(f"{var} = {num_str}")
      else:  # Symbolic solution
        # FIX for RecursionError and to correctly floatify numbers for forms like 0.5*(...)
        # Target only numerical atoms (Integer, Rational) for conversion to Float.
        # Do not attempt to evalf symbols themselves.
        temp_sol = sol_expr.xreplace({
          n: sympy.Float(n)
          for n in sol_expr.atoms(sympy.Number)
          if n.is_Rational or (n.is_Integer and n != 0)  # Avoid 0.0, preserve 0 if it was Integer(0)
          # and avoid floatifying integer exponents like in x**2
        })

        # Ensure that if a term like `b/a` results, `a` is not `a**1.0`
        # This is tricky with full_prec=False. The printer might still add **1.0 if exponent was float 1.0.
        # The xreplace above tries to only floatify Rational/Integer numbers, not exponents unless they are numbers.
        # If sol_expr was `b/a`, temp_sol should remain `b/a`.
        # If sol_expr was `(5-x)/2`, temp_sol becomes `(Float(5.0)-x)/Float(2.0)`

        factored_form = sympy.factor(temp_sol, deep=True)
        coeff_part, terms_part = factored_form.as_coeff_Mul()
        if coeff_part.is_Number and coeff_part < 0 and terms_part != 1 and not terms_part.is_Number:
          final_print_form = sympy.Abs(coeff_part) * sympy.simplify(-terms_part)
        else:
          final_print_form = factored_form
        output_parts.append(f"{var} = {printer.doprint(final_print_form)}")
    return ", ".join(output_parts)
  raise TypeError(f"Unsupported type for 'var': {type(var)}. Must be `str`, `int`, `float`, or `None`.")


# if __name__ == "__main__":
#   # Test specific cases
#   print(f"'2*x + 4*y = 10', var='y' -> {solve('2*x + 4*y = 10', var='y')}") # Expect y = 0.5*(5.0 - x)
#   print(f"'a*x - b = 0', var='x' -> {solve('a*x - b = 0', var='x')}")     # Expect x = b/a
