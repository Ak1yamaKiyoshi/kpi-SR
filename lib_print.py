import numpy as np
import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class highlight:
    indicies:List[Tuple[int, int]]
    style:str
    priority: int # 0 heighest
    description:str 

def printer(
    array:np.ndarray,
    text:str = "", 
    higlights: List[highlight] = [],
    formatting:str = "0.2f",
    separator:str = ", ",
    pre_row_str:str = "  ",
    print_description:bool = False,
    print_text: bool = True,
    default_style:str = "",
    reset_style:str = "\033[0m",
):

    higlights = sorted(higlights, key=lambda h: h.priority, reverse=True)
    if not higlights:
        higlights = [highlight([], ANSI.Styles.RESET, 0, "")]


    output = [["" for _ in range(array.shape[0])] for _ in range(array.shape[1])]    

    descriptions = ""
    for n, hlght in enumerate(higlights):
        for j, row in enumerate(array):
            for i, val in enumerate(row):
                if (i, j) in hlght.indicies:
                    output[i][j] = f"{f'{hlght.style}{val:{formatting}}'}{reset_style}"
                elif output[i][j] == "":
                    output[i][j] = f"{f'{default_style}{val:{formatting}}{reset_style}'}" 
        descriptions += f" * {n: 2d}. {hlght.style}{hlght.description}{reset_style}\n"
    mat_str = pre_row_str + f"\n{pre_row_str}".join(separator.join(map(str, row)) for row in output)
    
    out = ""
    if print_text:
        out += text + ANSI.Styles.RESET + "\n"
    if print_description:
        out += descriptions + ""
    out += mat_str
    
    return out

class pidx:
    def __init__(self):
        self.counter = 0
    
    def get(self):
        self.counter += 1
        return self.counter

    def str(self):
        return f"{self.get(): 2d}. "

class ANSI:
    class Styles:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        HIDDEN = "\033[8m"
        STRIKETHROUGH = "\033[9m"
    class FG:
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        # Bright colors
        BRIGHT_BLACK = "\033[90m"
        BRIGHT_RED = "\0ANSI.Sty33[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_WHITE = "\033[97m"

    class BG:
        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
        # Bright colors
        BRIGHT_BLACK = "\033[100m"
        BRIGHT_RED = "\033[101m"
        BRIGHT_GREEN = "\033[102m"
        BRIGHT_YELLOW = "\033[103m"
        BRIGHT_BLUE = "\033[104m"
        BRIGHT_MAGENTA = "\033[105m"
        BRIGHT_CYAN = "\033[106m"
        BRIGHT_WHITE = "\033[107m"


if __name__ == "__main__":
    mat = np.random.random((5, 5))
    h1 = highlight([(0, 0), (4, 4)], ANSI.Styles.BOLD + ANSI.BG.GREEN, 0, "aboba1")
    h2 = highlight([(4, 2), (2, 1)], ANSI.Styles.ITALIC + ANSI.BG.RED, 0, "aboba2")
    mat_str = printer(mat, "cool text phronebius id k ksk kssk sk", [h1, h2])
    print(mat_str)
