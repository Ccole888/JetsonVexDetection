from ultralytics import YOLO
import torch

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,  # Reduced from 3024
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        name='vex_cubes_detection'
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()