import sys


def inspect_recording(path: str, max_messages: int = 5) -> None:
    from mcap.reader import make_reader
    from mcap_ros2idl_support import Ros2DecodeFactory

    factory = Ros2DecodeFactory()
    with open(path, "rb") as recording:
        reader = make_reader(recording, decoder_factories=[factory])
        for index, decoded in enumerate(
            reader.iter_decoded_messages(
                topics=["/robot/right_arm/joint_state", "/robot/right_gripper/distance"]
            ),
            start=1,
        ):
            print(f"Message {index}:")
            print(decoded.channel.topic)
            print(decoded.decoded_message)
            if index >= max_messages:
                break


if __name__ == "__main__":
    inspect_recording(
        sys.argv[1] if len(sys.argv) > 1 else "data/recording_20260401_185226.mcap"
    )
