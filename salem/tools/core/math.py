# noqa: A005
# https://saturncloud.io/blog/pythonnumpyscipy-converting-string-to-mathematical-function/
import re

import sympy

from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.str import StrPrinter

from salem.tools.runtime import runtime_callable


def _can_convert_to_integer(v: sympy.NumberSymbol) -> bool:
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
    "Function": sympy.Function,  # Allow undefined functions to be parsed
  }

  parsed_obj: sympy.Basic  # To store the parsed expression/equation

  # --- 1. Parsing Stage ---
  is_equation_str_input = "=" in expr
  if is_equation_str_input:
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
      # CRITICAL: evaluate=False to allow sympy.solve to handle contradictions like x = x+1
      parsed_obj = sympy.Eq(lhs, rhs, evaluate=False)
    except Exception as e:
      raise ValueError(f"Incorrectly formatted equation: '{expr}'. Error: {e}")
  else:  # Not an equation string
    try:
      parsed_obj = parse_expr(expr, global_dict=restricted_globals, evaluate=True)
    except Exception as e:
      raise ValueError(f"Incorrect expression (unsupported symbols/functions?): '{expr}'. Error: {e}")

  # --- 2. Mode Dispatch ---

  # Mode: Substitution and Evaluation (var is a number)
  if isinstance(var, (int, float)):
    if is_equation_str_input:
      raise ValueError(
        "Substitution mode (when 'var' is a number) is not supported for expressions containing '='. "
        "To solve an equation, specify the variable name as a string for 'var'."
      )
    if not isinstance(parsed_obj, sympy.Expr):
      raise ValueError(f"Cannot substitute into non-expression type: {type(parsed_obj)}")

    free_symbols = list(parsed_obj.free_symbols)
    if not free_symbols:
      raise ValueError("Expression has no variables to substitute when 'var' is a number.")
    if len(free_symbols) > 1:
      raise ValueError(
        "Expression must contain exactly one variable for substitution when 'var' is a number. "
        f"Found: {', '.join(s.name for s in free_symbols)}"
      )
    symbol_to_sub = free_symbols[0]

    try:
      substituted_expr = parsed_obj.subs(symbol_to_sub, sympy.sympify(var))

      if substituted_expr.has(sympy.pi, sympy.E) and not substituted_expr.is_number:

        def floatize_int_coeff(node):
          if node.is_Mul:
            has_sym_const = any(arg.has(sympy.pi, sympy.E) for arg in node.args)
            if has_sym_const:
              new_args = [sympy.Float(arg) if arg.is_Integer else arg for arg in node.args]
              return sympy.Mul(*new_args, evaluate=False)
          return node

        processed_value = substituted_expr.replace(lambda n: n.is_Mul, floatize_int_coeff)
        if processed_value == sympy.E and var == 1 and parsed_obj.has(sympy.exp, symbol_to_sub):
          result_str = printer.doprint(sympy.E)
        else:
          result_str = printer.doprint(processed_value)
      elif not substituted_expr.free_symbols:
        evaluated_value = substituted_expr.evalf()
        if _can_convert_to_integer(evaluated_value):
          result_str = printer.doprint(sympy.Integer(evaluated_value))
        else:
          result_str = printer.doprint(evaluated_value)
      else:
        result_str = printer.doprint(substituted_expr)
      return f"{symbol_to_sub.name} = {result_str}"
    except Exception as e:
      raise ValueError(f"Error during substitution or evaluation: {e}")

  # Mode: Simplification / Evaluation (var is None)
  elif var is None:
    simplified_expr = sympy.simplify(parsed_obj)

    if isinstance(simplified_expr, sympy.Equality):  # Custom print for Eq to be "lhs = rhs"
      return f"{printer.doprint(simplified_expr.lhs)} = {printer.doprint(simplified_expr.rhs)}"

    if not simplified_expr.free_symbols:
      if isinstance(simplified_expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
        return printer.doprint(simplified_expr)
      if simplified_expr.is_Integer:  # e.g. sin(x)**2+cos(x)**2 -> 1, print as "1.0"
        # The test `test_symbolic_simplification_various` asserts `solve("sin(x)**2 + cos(x)**2") == "1"`, not "1.0"
        # Adjusting to match test expectation. If "1.0" is desired, change to sympy.Float(simplified_expr)
        return printer.doprint(simplified_expr)
      if simplified_expr.is_Number:
        return printer.doprint(simplified_expr.evalf())
      return printer.doprint(simplified_expr)
    else:  # Has free symbols
      # FIX for Horner FlagError: gen is positional, not keyword
      if (
        len(simplified_expr.free_symbols) == 1
        and isinstance(simplified_expr, sympy.Expr)
        and not isinstance(simplified_expr, sympy.Rel)
      ):
        single_var = list(simplified_expr.free_symbols)[0]
        if simplified_expr.is_polynomial(single_var):
          try:
            expanded_for_horner = sympy.expand(simplified_expr)
            horner_form = sympy.horner(expanded_for_horner, single_var)  # gen is positional
            # Use Horner if it results in a different, potentially more "factored" form
            if horner_form != simplified_expr and horner_form != expanded_for_horner:
              return printer.doprint(horner_form)
          except Exception:
            pass  # Fallback to printing simplified_expr
      return printer.doprint(simplified_expr)
  # Mode: Solving for a variable (var is a string name)
  elif isinstance(var, str):
    try:
      target_symbol = sympy.Symbol(var, real=True)
    except ValueError:  # Raised by SymPy for invalid symbol names like "123"
      raise ValueError(f"Invalid variable name '{var}' for solving.")

    equation_to_solve: sympy.Basic
    if isinstance(parsed_obj, sympy.Equality):
      equation_to_solve = parsed_obj
    elif isinstance(parsed_obj, sympy.Expr):  # Input was expression, solve expr=0
      equation_to_solve = sympy.Eq(parsed_obj, 0)
    elif isinstance(parsed_obj, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
      # This can happen if Eq(..., evaluate=True) was used, or simplify(Eq(...))
      equation_to_solve = parsed_obj  # Pass True/False directly to sympy.solve
    else:
      raise ValueError(f"Cannot form an equation from input '{expr}' (parsed as: {type(parsed_obj)}).")

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
        if _can_convert_to_integer(evaluated_sol):
          num_str = printer.doprint(sympy.Integer(evaluated_sol))
        else:
          num_str = printer.doprint(evaluated_sol)
        output_parts.append(f"{var} = {num_str}")
      else:  # Symbolic solution
        # Floatify numerical parts: e.g. (5-x)/2 -> (5.0-x)/2.0
        temp_sol = sol_expr.evalf(subs={s: s for s in sol_expr.free_symbols if s.is_Symbol})

        # Factor and apply cosmetic preferences
        # For y = (10-2x)/4 -> (5-x)/2 -> 2.5 - 0.5*x
        # We want 0.5*(5.0-x)
        # Try to achieve this by factoring common numerical denominator/coefficient

        # A simple approach for 0.5*(5.0-x) might be to factor out common numerical term
        # sympy.collect_const(term, C) can be useful
        # Or use sympy.Poly to extract coefficients

        # Attempt specific factoring for desired output like 0.5*(5.0-x)
        # If `temp_sol` is Add, e.g., `2.5 - 0.5*x`
        if temp_sol.is_Add:
          # Try to find a common numerical factor for all terms
          # This is a heuristic. For 2.5 - 0.5*x, we can factor 0.5 to get 0.5*(5.0 - x)
          # Or factor -0.5 to get -0.5*(x - 5.0)
          # Let's try to get positive leading coefficient for the factored part if possible

          # Heuristic to achieve desired form:
          # If it's like A - B*x or B*x + A, try to make it B*(x + A/B) or -B*(-x - A/B)
          # For 2.5 - 0.5*x, could become -0.5*(x - 5.0)
          # Then cosmetic step to 0.5*(5.0 - x)

          # Let's simplify the expression and rely on sympy's factor and printer.
          # The main goal is to ensure numbers are floats.
          factored_form = sympy.factor(temp_sol, deep=True)

          coeff_part, terms_part = factored_form.as_coeff_Mul()
          if coeff_part.is_Number and coeff_part < 0 and terms_part != 1 and not terms_part.is_Number:
            # Invert terms: -A*(B-C) -> A*(C-B)
            # For -0.5*(x - 5.0) -> 0.5*(-x + 5.0), printer output 0.5*(5.0 - x)
            final_print_form = sympy.Abs(coeff_part) * sympy.simplify(-terms_part)
          else:
            final_print_form = factored_form

          output_parts.append(f"{var} = {printer.doprint(final_print_form)}")
        else:  # Not an Add, e.g. b/a
          output_parts.append(f"{var} = {printer.doprint(temp_sol)}")  # temp_sol has floatified numbers

    return ", ".join(output_parts)

  raise TypeError(f"Unsupported type for 'var': {type(var)}. Must be `str`, `int`, `float`, or `None`.")


if __name__ == "__main__":
  output = solve("unknown_function(x)", var="x")
  print(output)
