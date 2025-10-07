import cv2
import numpy as np

class Color:
    def __init__(self, name: str, hsv_lower: np.ndarray, hsv_upper: np.ndarray):
        self.name = name
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
    
    def get_mask(self, hsv_image: np.ndarray):
        return cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)

def combine_masks(colors: list[Color], hsv_image: np.ndarray):
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    #创建一个 “空白画布”，用来存放合并后的掩膜
    #hsv_image.shape确保这个空白掩膜的尺寸（高度、宽度）和输入的 HSV 图像完全一致，这样每个像素才能一一对应。
    #[:2]保证只有前两个维度(高度, 宽度)，没有第三个（三个通道）
    #dtype=np.uint8指定数组的数据类型为8位无符号整数
    #初始化为全黑掩膜（像素值都是 0），因为一开始还没有任何颜色区域需要保留。
    for color in colors:
        single_mask = color.get_mask(hsv_image)
        combined_mask = cv2.bitwise_or(combined_mask, single_mask)
    return combined_mask

class BallDetector:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.roi_coords = (120, 350, 150, 450)  
        # 默认ROI，提供后备值，确保程序始终有可用的ROI坐标
        
        # 定义颜色范围
        self.red1 = Color("红色1", np.array([0, 120, 70]), np.array([10, 255, 255]))
        self.red2 = Color("红色2", np.array([170, 120, 70]), np.array([180, 255, 255]))
        self.blue = Color("蓝色", np.array([100, 80, 80]), np.array([155, 255, 240]))
        self.purple = Color("紫色", np.array([125, 50, 50]), np.array([155, 255, 230]))
        
        self.min_threshold = 700

    def set_roi(self, y_start: int, y_end: int, x_start: int, x_end: int):
        self.roi_coords = (y_start, y_end, x_start, x_end)

    def process_frame(self, frame: np.ndarray):
        # 预处理
        frame = cv2.rotate(frame, cv2.ROTATE_180)#倒转180度
        frame = cv2.GaussianBlur(frame, (5, 5), 1)#高斯降噪
        
        # 提取ROI
        y_start, y_end, x_start, x_end = self.roi_coords #从self.roi_coords中解包
        roi = frame[y_start:y_end, x_start:x_end]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 检测颜色
        red_mask = combine_masks([self.red1, self.red2], hsv_roi)
        red_count = cv2.countNonZero(red_mask)
        
        blue_mask = self.blue.get_mask(hsv_roi)
        blue_count = cv2.countNonZero(blue_mask)
        
        purple_mask = self.purple.get_mask(hsv_roi)
        purple_count = cv2.countNonZero(purple_mask)

        # 判断主导颜色
        color_counts = {'红色': red_count, '蓝色': blue_count, '紫色': purple_count}
        dominant_color = max(color_counts, key=color_counts.get)

        # 设置显示文本
        if color_counts[dominant_color] < self.min_threshold:
            display_text = "Nothing"
            text_color = (255, 255, 255)
        else:
            if dominant_color == "红色":
                display_text = "Red"
                text_color = (0, 0, 255)
            elif dominant_color == "蓝色":
                display_text = "Blue"
                text_color = (255, 0, 0)
            elif dominant_color == "紫色":
                display_text = "Purple"
                text_color = (255, 0, 255)

        # 显示结果
        cv2.putText(frame, display_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        return frame

    def process_video(self):
        cam = cv2.VideoCapture(self.video_path)
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Ball Detection', processed_frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

def main():
    detector = BallDetector("res/output1.avi")
    detector.process_video()

if __name__ == "__main__":
    main()
       