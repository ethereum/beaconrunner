import inspect

from IPython.display import Code
from pygments.formatters import HtmlFormatter
from IPython.core.display import HTML

def display_code(code):
    """Inspect Python modules, functions and return the syntax highlighted code
    """
    formatter = HtmlFormatter()
    display(HTML(f'<style>{formatter.get_style_defs(".highlight")}</style>'))

    return Code(inspect.getsource(code), language='python')
