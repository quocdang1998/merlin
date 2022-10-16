from .build_ext import build_ext
from .create_merlin_init import create_merlin_init
from .ext_options import get_extension_options
from .config import *

ext_options = get_extension_options()
create_merlin_init()
