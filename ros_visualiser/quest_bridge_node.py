"""ROS2 bridge node: Meta Quest → TF + PoseStamped + Joy.

Reads Meta Quest controller poses plus button states, then publishes:

  /tf                           — TransformStamped for right_controller
                                  and left_controller, in world_frame
  /quest/right_controller/pose  — geometry_msgs/PoseStamped (right hand)
  /quest/left_controller/pose   — geometry_msgs/PoseStamped (left hand)
  /quest/joy                    — sensor_msgs/Joy

Joy layout
----------
axes[0]  rightGrip      analog 0.0–1.0
axes[1]  leftGrip       analog 0.0–1.0
axes[2]  rightThumbX    joystick X  −1.0–1.0
axes[3]  rightThumbY    joystick Y  −1.0–1.0
axes[4]  leftThumbX     joystick X  −1.0–1.0
axes[5]  leftThumbY     joystick Y  −1.0–1.0
axes[6]  rightTrig      index trigger 0.0–1.0
axes[7]  leftTrig       index trigger 0.0–1.0

buttons[0]  A
buttons[1]  B
buttons[2]  X
buttons[3]  Y
buttons[4]  RJ  (right joystick click)
buttons[5]  LJ  (left joystick click)

Parameters
----------
quest_ip     (string)  Quest device IP address  default: 192.168.50.215
quest_port   (int)     ADB-forwarded port        default: 5555
world_frame  (string)  TF parent frame name      default: world
publish_rate (double)  Timer frequency in Hz     default: 50.0
"""

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Joy

from meta_quest_teleop.reader import MetaQuestReader


class QuestBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('quest_bridge_node')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('quest_ip', '192.168.50.215')
        self.declare_parameter('quest_port', 5555)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('publish_rate', 50.0)

        quest_ip = self.get_parameter('quest_ip').get_parameter_value().string_value
        quest_port = self.get_parameter('quest_port').get_parameter_value().integer_value
        self._world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # ── MetaQuestReader ───────────────────────────────────────────────────
        # The reader spawns its own background thread that continuously pulls
        # data from the Quest device via ADB logcat.  All public getter methods
        # on the reader are safe to call from our timer callback because:
        #   • last_transforms is replaced atomically (full dict swap, not
        #     in-place mutation), so CPython's GIL guarantees we always see a
        #     consistent snapshot.
        #   • get_transformations_and_buttons() deep-copies both dicts before
        #     returning, which we use for button/axis data to get a stable
        #     snapshot within a single tick.
        self.get_logger().info(f'Connecting to Meta Quest at {quest_ip}:{quest_port}')
        try:
            self._reader = MetaQuestReader(ip_address=quest_ip, port=quest_port)
        except Exception as exc:
            self.get_logger().fatal(f'MetaQuestReader init failed: {exc}')
            raise

        # ── publishers ────────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._pub_right = self.create_publisher(
            PoseStamped, '/quest/right_controller/pose', 10)
        self._pub_left = self.create_publisher(
            PoseStamped, '/quest/left_controller/pose', 10)
        self._pub_joy = self.create_publisher(Joy, '/quest/joy', 10)

        # ── timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'QuestBridgeNode ready — world_frame="{self._world_frame}", '
            f'rate={rate:.0f} Hz'
        )

    # ── conversion helpers ────────────────────────────────────────────────────

    def _to_tf(self, T: np.ndarray, stamp, child_frame: str) -> TransformStamped:
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = self._world_frame
        ts.child_frame_id = child_frame
        ts.transform.translation.x = float(T[0, 3])
        ts.transform.translation.y = float(T[1, 3])
        ts.transform.translation.z = float(T[2, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # scipy → [x, y, z, w]
        ts.transform.rotation.x = float(q[0])
        ts.transform.rotation.y = float(q[1])
        ts.transform.rotation.z = float(q[2])
        ts.transform.rotation.w = float(q[3])
        return ts

    def _to_pose(self, T: np.ndarray, stamp) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self._world_frame
        ps.pose.position.x = float(T[0, 3])
        ps.pose.position.y = float(T[1, 3])
        ps.pose.position.z = float(T[2, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        return ps

    # ── timer callback ────────────────────────────────────────────────────────

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()

        # --- controller poses: TF + PoseStamped ---
        for hand, child_frame, pub in (
            ('right', 'right_controller', self._pub_right),
            ('left',  'left_controller',  self._pub_left),
        ):
            try:
                T = self._reader.get_hand_controller_transform_ros(hand)
            except Exception as exc:
                self.get_logger().warn(
                    f'Failed to read {hand} transform: {exc}',
                    throttle_duration_sec=5.0,
                )
                continue

            if T is None:
                continue

            self._tf_broadcaster.sendTransform(self._to_tf(T, now, child_frame))
            pub.publish(self._to_pose(T, now))

        # --- buttons + axes → Joy ---
        # get_transformations_and_buttons() returns deep-copied dicts, giving
        # us a consistent snapshot of both transforms and buttons for this tick.
        try:
            _, buttons = self._reader.get_transformations_and_buttons()
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to read button state: {exc}',
                throttle_duration_sec=5.0,
            )
            return

        if buttons is None:
            return

        joy = Joy()
        joy.header.stamp = now

        def _axis(key: str, default: float = 0.0) -> float:
            v = buttons.get(key, default)
            return float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        def _btn(key: str) -> int:
            v = buttons.get(key, False)
            val = v[0] if isinstance(v, (list, tuple)) else v
            return int(bool(val))

        rjs = buttons.get('rightJS') or (0.0, 0.0)
        ljs = buttons.get('leftJS') or (0.0, 0.0)

        # axes: [rightGrip, leftGrip, rightThumbX, rightThumbY,
        #        leftThumbX, leftThumbY, rightTrig, leftTrig]
        joy.axes = [
            _axis('rightGrip'),
            _axis('leftGrip'),
            float(rjs[0]),
            float(rjs[1]),
            float(ljs[0]),
            float(ljs[1]),
            _axis('rightTrig'),
            _axis('leftTrig'),
        ]

        # buttons: [A, B, X, Y, RJ, LJ]
        joy.buttons = [
            _btn('A'),
            _btn('B'),
            _btn('X'),
            _btn('Y'),
            _btn('RJ'),
            _btn('LJ'),
        ]

        self._pub_joy.publish(joy)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QuestBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
