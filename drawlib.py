BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'


def printmatrix(strmatrix):
    for column in strmatrix:
        for row in column:
            print(row, end=" ")
        print()


def visualized(strmatrix, highlight0, highlight1):
    res = ""
    for column in strmatrix:
        for row in column:
            if row in highlight0:
                res += f"{GREEN}{row:5}{RESET}"
            elif row in highlight1:
                res += f"{YELLOW}{row:5}{RESET}"
            
            if row not in highlight0 and row not in highlight1: res += f"{row:5}"
        res += '\n'
    return [row.split(" ") for row in res.split("\n")]


def create_vis_matrix(shape, istr='i', jstr='j'):
    vis=[]
    for i in range(shape[0]):
        row = []
        for j in range(shape[1]): row.append(f"{istr}{i+1}{jstr}{j+1}")
        vis.append(row)
    return vis


def visualize_indicies(shape, highlight0=[],highlight1=[], istr='i', jstr='j'):
    vis = create_vis_matrix(shape, istr, jstr)
    res = visualized(vis, highlight0, highlight1)
    printmatrix(res)

