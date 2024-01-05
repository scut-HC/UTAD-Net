import os

def check_dirs(path):
    if type(path) not in (tuple, list):
        path = [path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return

def parse_image_name(name):
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, mod + name[len(mod):]
