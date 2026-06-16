CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 0.05  # 20Hz simulation target

EGO_VEHICLE_TYPE = "vehicle.lincoln.mkz_2017"
EGO_ROLE_NAME = "ego"

WINDOW_NAME = "CARLA Multi Camera View"

ENABLE_BEV_RECORDING = False
BEV_RECORD_DIR = "CarlaProject0916/output/bev_records"
BEV_RECORD_EVERY_N_FRAMES = 5

# Auto-stop controls for short simulation runs.
# Use None to keep running until 'q' or Ctrl+C.
MAX_SIMULATION_FRAMES = 500
MAX_SIMULATION_SECONDS = 60
