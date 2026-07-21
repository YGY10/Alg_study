from sim2d.gui.road_compare_extension import install as install_road_compare
from sim2d.gui.road_layer_extension import install as install_road_layers

install_road_layers()
install_road_compare()

from sim2d.gui.main_window import main  # noqa: E402


if __name__ == "__main__":
    main()
