from typing import Callable


def show_source(function: Callable) -> None:
    """
    Show the source code of the function, with syntax highlighting

    This is meant to be called from a Jupyter notebook.
    """
    code = inspect.getsource(function)
    lexer = PythonLexer()
    formatter = HtmlFormatter(cssclass="pygments")
    html_code = highlight(code, lexer, formatter)
    css = formatter.get_style_defs(".pygments")
    html = f"<style>{css}</style>{html_code}"
    display(HTML(html))

if __name__=="__main__":
    print(f"Empty main in : '{__file__[-12:]}'")