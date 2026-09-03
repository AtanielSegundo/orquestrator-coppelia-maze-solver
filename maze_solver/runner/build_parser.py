import argparse

from src.definitions import SIM

from runner.constants import KART_HOST, KART_PORT, CONTROLS, HELP


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coppelia_main.py",
        description="Resolve o labirinto com o kart REAL (Lafvin PiKart), "
                    "espelhado ao vivo no CoppeliaSim e nesta janela.",
        epilog=CONTROLS + HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--maze", default=SIM.DEFAULT_MAZE,
                   help="arquivo .maze (ou so o nome, ex: big_eg)")
    p.add_argument("--cell-lenght", type=float, default=0.3,
                   help="lado real da celula em metros (default: %(default)s)")
    p.add_argument("--kart-lenght", type=float, default=0.28,
                   help="maior lado do kart em metros (default: %(default)s)")
    p.add_argument("--host", default=KART_HOST,
                   help="IP do Lafvin PiKart (default: %(default)s)")
    p.add_argument("--port", type=int, default=KART_PORT,
                   help="porta do web.py do kart (default: %(default)s)")
    p.add_argument("--turn-speed", type=float, default=1.0,
                   help="'speed' enviado no comando turn (default: %(default)s)")
    p.add_argument("--drive-speed", type=float, default=0.2,
                   help="m/s enviado no comando drive_distance (default: %(default)s)")
    p.add_argument("--video", choices=("sim", "client", "off"), default="sim",
                   help="quem puxa o stream da camera -- ver camera_link/ "
                       "(default: %(default)s)")
    return p
