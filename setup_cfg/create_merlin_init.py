import os
import sys
import jinja2

from .config import *

def create_merlin_init():
    # create jinja template file
    searchpath = os.path.dirname(__file__)
    template_loader = jinja2.FileSystemLoader(searchpath=searchpath)
    template_env = jinja2.Environment(loader=template_loader)
    init_template = template_env.get_template("init.txt")

    # render template
    keywords = {"platform": sys.platform,
                "libkind": MERLIN_LIBKIND,
                "binary_dir": MERLIN_BIN_DIR}
    init_file = os.path.join(os.path.abspath(os.path.join(__file__, "../..")),
                             "merlin", "__init__.py")
    with open(init_file, "w") as f:
        f.write(init_template.render(keywords))
