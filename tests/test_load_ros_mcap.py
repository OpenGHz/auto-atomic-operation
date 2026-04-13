from mcap.reader import make_reader
from mcap_ros2idl_support import Ros2DecodeFactory
import sys

factory = Ros2DecodeFactory()

path = sys.argv[1] if len(sys.argv) > 1 else "data/recording_20260401_185226.mcap"

with open(path, "rb") as f:
    reader = make_reader(f, decoder_factories=[factory])
    for decoded in reader.iter_decoded_messages(
        topics=["/robot/right_arm/joint_state", "/robot/right_gripper/distance"]
    ):
        print(decoded.channel.topic)
        print(decoded.decoded_message)
