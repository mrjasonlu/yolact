
# Initialize the visualizer with your log file path
log_path = "./logs/yolact_plus_resnet50.log"

from utils.logger import LogVisualizer
vis = LogVisualizer()
vis.sessions(log_path)

vis.add(log_path, session=8)

vis.plot('val', 'x.data.iter', 'x.data.mask["all"]')