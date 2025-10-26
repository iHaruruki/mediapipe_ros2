# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from mediapipe_ros2_msgs.msg import PoseLandmark
import csv
import os
from datetime import datetime
import threading


class LandmarkCSVWriter(Node):
    def __init__(self):
        super().__init__('landmark_csv_writer')
        
        # ==== Parameters ====
        self.declare_parameter('csv_file_path', '~/landmark_data.csv')
        self.declare_parameter('topic_name', '/holistic/pose/landmarks/csv')
        self.declare_parameter('append_mode', True)  # True: 追記, False: 上書き
        self.declare_parameter('auto_filename', True)  # True: タイムスタンプ付きファイル名自動生成
        self.declare_parameter('buffer_size', 100)  # バッファサイズ（メモリ上に保持する行数）
        self.declare_parameter('flush_interval', 1.0)  # フラッシュ間隔（秒）
        
        # ==== Read parameters ====
        csv_file_path = self.get_parameter('csv_file_path').value
        self.topic_name = self.get_parameter('topic_name').value
        self.append_mode = bool(self.get_parameter('append_mode').value)
        self.auto_filename = bool(self.get_parameter('auto_filename').value)
        self.buffer_size = int(self.get_parameter('buffer_size').value)
        self.flush_interval = float(self.get_parameter('flush_interval').value)
        
        # ==== CSV file setup ====
        if self.auto_filename:
            # タイムスタンプ付きファイル名を自動生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.expanduser(csv_file_path)
            dir_path = os.path.dirname(base_path)
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            self.csv_file_path = os.path.join(dir_path, f"{base_name}_{timestamp}.csv")
        else:
            self.csv_file_path = os.path.expanduser(csv_file_path)
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
        
        # ==== CSV file and buffer initialization ====
        self.csv_buffer = []
        self.csv_lock = threading.Lock()
        self.total_rows_written = 0
        
        # CSVファイルを開いてヘッダーを書き込み
        mode = 'a' if self.append_mode and os.path.exists(self.csv_file_path) else 'w'
        self.csv_file = open(self.csv_file_path, mode, newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # ヘッダーを書き込み（新規作成時または上書きモード時）
        if mode == 'w' or (mode == 'a' and os.path.getsize(self.csv_file_path) == 0):
            self.csv_writer.writerow([
                'timestamp_sec', 'timestamp_nanosec', 'frame_id', 
                'name', 'index', 'x', 'y'
            ])
            self.csv_file.flush()
        
        # ==== Subscriber ====
        self.landmark_sub = self.create_subscription(
            PoseLandmark,
            self.topic_name,
            self.landmark_callback,
            10
        )
        
        # ==== Timer for periodic flush ====
        self.flush_timer = self.create_timer(self.flush_interval, self.flush_buffer)
        
        self.get_logger().info(f'Landmark CSV Writer started')
        self.get_logger().info(f'  - CSV file: {self.csv_file_path}')
        self.get_logger().info(f'  - Topic: {self.topic_name}')
        self.get_logger().info(f'  - Mode: {"Append" if self.append_mode else "Overwrite"}')
        self.get_logger().info(f'  - Buffer size: {self.buffer_size}')
        self.get_logger().info(f'  - Flush interval: {self.flush_interval}s')

    def landmark_callback(self, msg: PoseLandmark):
        """PoseLandmarkメッセージを受信してバッファに追加"""
        try:
            # タイムスタンプを秒とナノ秒に分離
            timestamp_sec = msg.header.stamp.sec
            timestamp_nanosec = msg.header.stamp.nanosec
            
            # CSVの行データを作成
            row_data = [
                timestamp_sec,
                timestamp_nanosec,
                msg.header.frame_id,
                msg.name,
                msg.index,
                msg.x if not (msg.x != msg.x) else 'nan',  # NaNチェック
                msg.y if not (msg.y != msg.y) else 'nan'   # NaNチェック
            ]
            
            # バッファに追加（スレッドセーフ）
            with self.csv_lock:
                self.csv_buffer.append(row_data)
                
                # バッファがフルになったら即座にフラッシュ
                if len(self.csv_buffer) >= self.buffer_size:
                    self._write_buffer_to_file()
                    
        except Exception as e:
            self.get_logger().error(f'Error in landmark_callback: {str(e)}')

    def flush_buffer(self):
        """定期的にバッファをフラッシュ"""
        with self.csv_lock:
            if self.csv_buffer:
                self._write_buffer_to_file()

    def _write_buffer_to_file(self):
        """バッファの内容をCSVファイルに書き込み（ロック済み前提）"""
        try:
            if not self.csv_buffer:
                return
                
            # バッファの内容をファイルに書き込み
            for row in self.csv_buffer:
                self.csv_writer.writerow(row)
            
            # ファイルをフラッシュ
            self.csv_file.flush()
            
            # 統計更新
            rows_written = len(self.csv_buffer)
            self.total_rows_written += rows_written
            
            # ログ出力（大量のデータの場合は間引く）
            if self.total_rows_written % 100 == 0 or rows_written >= self.buffer_size:
                self.get_logger().info(f'Wrote {rows_written} rows to CSV (total: {self.total_rows_written})')
            
            # バッファをクリア
            self.csv_buffer.clear()
            
        except Exception as e:
            self.get_logger().error(f'Error writing to CSV file: {str(e)}')

    def destroy_node(self):
        """ノード終了時の処理"""
        # 残りのバッファをフラッシュ
        with self.csv_lock:
            if self.csv_buffer:
                self._write_buffer_to_file()
        
        # CSVファイルを閉じる
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            
        self.get_logger().info(f'CSV Writer stopped. Total rows written: {self.total_rows_written}')
        self.get_logger().info(f'CSV file saved: {self.csv_file_path}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = LandmarkCSVWriter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Received interrupt signal')
    except Exception as e:
        node.get_logger().error(f'Error in main: {str(e)}')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()