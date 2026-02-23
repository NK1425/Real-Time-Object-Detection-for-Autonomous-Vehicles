"""
ROS2 Detection Publisher Node

Publishes YOLOv5 + ByteTrack detections as ROS2 messages on:
  /av_detection/image_annotated   (sensor_msgs/Image)
  /av_detection/detections        (vision_msgs/Detection2DArray)
  /av_detection/markers           (visualization_msgs/MarkerArray)
  /av_detection/stats             (std_msgs/String — JSON)

Subscribes to:
  /camera/image_raw               (sensor_msgs/Image)

Usage (ROS2 Humble / Iron):
    colcon build --packages-select av_detection
    source install/setup.bash
    ros2 run av_detection detection_publisher \
        --ros-args \
        -p weights:=weights/yolov5_kitti.pt \
        -p conf_thresh:=0.25 \
        -p use_depth:=true \
        -p use_tracking:=true

Test without hardware:
    ros2 run av_detection detection_publisher \
        --ros-args -p use_sim:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
import json
import sys
import time
from pathlib import Path

# ROS2 message types
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detector import YOLOv5Detector
from models.tracker import ByteTracker
from models.depth_estimator import MonocularDepthEstimator
from inference.visualizer import Visualizer
from models.detector import KITTI_CLASSES


class AVDetectionNode(Node):
    """
    ROS2 node for real-time AV object detection and tracking.

    This node is the bridge between the perception stack and
    the rest of the AV software stack (planning, control, mapping).

    Published topics:
        /av_detection/image_annotated  — Annotated camera image
        /av_detection/detections       — vision_msgs/Detection2DArray
        /av_detection/markers          — 3D distance markers (RViz)
        /av_detection/stats            — JSON performance stats

    Parameters:
        weights (str)       — Path to YOLOv5 weights
        conf_thresh (float) — Detection confidence threshold
        iou_thresh (float)  — NMS IoU threshold
        use_depth (bool)    — Enable depth estimation
        use_tracking (bool) — Enable ByteTrack MOT
        use_sim (bool)      — Publish synthetic frames (no camera needed)
        img_size (int)      — Inference image size
        device (str)        — 'cpu' or 'cuda'
    """

    def __init__(self):
        super().__init__('av_detection_node')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('weights',      'yolov5s')
        self.declare_parameter('conf_thresh',  0.25)
        self.declare_parameter('iou_thresh',   0.45)
        self.declare_parameter('use_depth',    True)
        self.declare_parameter('use_tracking', True)
        self.declare_parameter('use_sim',      False)
        self.declare_parameter('img_size',     640)
        self.declare_parameter('device',       'auto')
        self.declare_parameter('camera_topic', '/camera/image_raw')

        weights      = self.get_parameter('weights').value
        conf_thresh  = self.get_parameter('conf_thresh').value
        iou_thresh   = self.get_parameter('iou_thresh').value
        self.use_depth    = self.get_parameter('use_depth').value
        self.use_tracking = self.get_parameter('use_tracking').value
        self.use_sim      = self.get_parameter('use_sim').value
        img_size     = self.get_parameter('img_size').value
        device       = self.get_parameter('device').value
        camera_topic = self.get_parameter('camera_topic').value

        # ── Perception stack ────────────────────────────────────────────────
        self.get_logger().info(f"Loading detector: {weights}")
        self.detector  = YOLOv5Detector(weights, device=device,
                                        img_size=img_size,
                                        conf_thresh=conf_thresh,
                                        iou_thresh=iou_thresh)
        self.tracker   = ByteTracker()
        self.depth_est = MonocularDepthEstimator()
        self.visualizer = Visualizer(show_depth=True, show_tracks=True, show_bev=True)
        self.bridge    = CvBridge()
        self.frame_id  = 0

        # ── QoS profiles ─────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self.pub_image  = self.create_publisher(
            Image, '/av_detection/image_annotated', sensor_qos)
        self.pub_dets   = self.create_publisher(
            Detection2DArray, '/av_detection/detections', reliable_qos)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/av_detection/markers', reliable_qos)
        self.pub_stats  = self.create_publisher(
            String, '/av_detection/stats', reliable_qos)

        # ── Subscriber or simulation timer ─────────────────────────────────
        if self.use_sim:
            self.get_logger().info("Running in simulation mode (synthetic frames)")
            self.sim_timer = self.create_timer(
                1.0 / 30.0,  # 30 Hz
                self._sim_callback
            )
        else:
            self.get_logger().info(f"Subscribing to: {camera_topic}")
            self.sub_camera = self.create_subscription(
                Image, camera_topic,
                self._image_callback,
                sensor_qos
            )

        # ── Stats timer (publish every 2s) ────────────────────────────────
        self.stats_timer = self.create_timer(2.0, self._publish_stats)
        self._latencies: list = []
        self._det_counts: list = []

        self.get_logger().info("AVDetectionNode ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        """Process incoming camera frame."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        self._process_frame(frame, msg.header)

    def _sim_callback(self):
        """Generate synthetic frame for testing without hardware."""
        h, w = 375, 1242
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Synthetic road
        cv2.rectangle(frame, (0, h//2), (w, h), (50, 50, 50), -1)
        # Fake car boxes (moving slightly each frame)
        t = self.frame_id * 0.02
        for i, (x0, y0, cname) in enumerate([(200, 180, 'Car'), (500, 160, 'Van'), (800, 200, 'Pedestrian')]):
            x = int(x0 + np.sin(t + i) * 30)
            cv2.rectangle(frame, (x, y0), (x + 120, y0 + 80), (100, 180, 255), -1)
            cv2.putText(frame, cname, (x + 5, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'
        self._process_frame(frame, header)

    def _process_frame(self, frame: np.ndarray, header):
        """Run full perception pipeline and publish results."""
        self.frame_id += 1
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # Detect
        detections, inf_ms = self.detector.detect(frame)

        # Track
        if self.use_tracking and detections:
            detections = self.tracker.update(detections)

        # Depth
        if self.use_depth and detections:
            self.depth_est.estimate_batch(detections, h, w)

        total_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(total_ms)
        self._det_counts.append(len(detections))

        # Publish annotated image
        annotated = self.visualizer.draw(frame.copy(), detections,
                                         frame_id=self.frame_id,
                                         fps=1000.0 / (total_ms + 1e-6))
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
            img_msg.header = header
            self.pub_image.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Image publish error: {e}")

        # Publish detection array
        det_array = self._build_detection_array(detections, header)
        self.pub_dets.publish(det_array)

        # Publish RViz markers (distance text markers)
        if self.use_depth:
            markers = self._build_depth_markers(detections, header)
            self.pub_markers.publish(markers)

    def _build_detection_array(
        self, detections: list, header
    ) -> Detection2DArray:
        """Convert Detection objects to ROS2 vision_msgs/Detection2DArray."""
        arr = Detection2DArray()
        arr.header = header

        for det in detections:
            d2 = Detection2D()
            d2.header = header

            # Bounding box center + size
            x1, y1, x2, y2 = det.bbox
            d2.bbox.center.position.x = float((x1 + x2) / 2)
            d2.bbox.center.position.y = float((y1 + y2) / 2)
            d2.bbox.size_x = float(x2 - x1)
            d2.bbox.size_y = float(y2 - y1)

            # Class hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det.class_name
            hyp.hypothesis.score    = det.confidence
            d2.results.append(hyp)

            arr.detections.append(d2)

        return arr

    def _build_depth_markers(
        self, detections: list, header
    ) -> MarkerArray:
        """Build RViz TEXT_VIEW_FACING markers showing distance per object."""
        marker_arr = MarkerArray()

        for i, det in enumerate(detections):
            if det.depth is None:
                continue

            marker = Marker()
            marker.header = header
            marker.ns     = 'av_depth'
            marker.id     = i
            marker.type   = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Approximate 3D position from depth + bbox center
            cx = (det.bbox[0] + det.bbox[2]) / 2
            marker.pose.position.x = float(det.depth)
            marker.pose.position.y = float(cx)
            marker.pose.position.z = 1.5
            marker.pose.orientation.w = 1.0

            marker.scale.z = 0.5  # text height in metres
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker.text = f"{det.class_name}\n{det.depth:.1f}m"

            # Expire after 0.2s (10 frames at 30Hz)
            marker.lifetime.nanoseconds = 200_000_000

            marker_arr.markers.append(marker)

        return marker_arr

    def _publish_stats(self):
        """Publish JSON performance stats every 2 seconds."""
        if not self._latencies:
            return

        recent = self._latencies[-60:]
        stats = {
            'avg_fps':    round(1000.0 / (np.mean(recent) + 1e-6), 1),
            'avg_latency_ms': round(float(np.mean(recent)), 2),
            'p95_latency_ms': round(float(np.percentile(recent, 95)), 2),
            'avg_objects':    round(float(np.mean(self._det_counts[-60:])), 1),
            'total_frames':   self.frame_id,
            'active_tracks':  self.tracker.get_active_track_count(),
        }
        msg = String()
        msg.data = json.dumps(stats)
        self.pub_stats.publish(msg)
        self.get_logger().info(
            f"FPS={stats['avg_fps']} | Latency={stats['avg_latency_ms']}ms | "
            f"Objects={stats['avg_objects']} | Tracks={stats['active_tracks']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AVDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
