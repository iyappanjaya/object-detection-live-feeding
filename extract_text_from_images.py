import os
import easyocr
import cv2

# Directory containing images
dir_path = 'content'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# List all image files in the directory
image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]


import pandas as pd

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

for img_name in image_files:
    img_path = os.path.join(dir_path, img_name)
    print(f'\nProcessing: {img_path}')
    try:
        results = reader.readtext(img_path)
        data = []
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: Could not read {img_path}. Skipping.')
            continue
        for (bbox, text, prob) in results:
            x_min = int(min([point[0] for point in bbox]))
            y_min = int(min([point[1] for point in bbox]))
            x_max = int(max([point[0] for point in bbox]))
            y_max = int(max([point[1] for point in bbox]))
            data.append({
                'Text': text,
                'Confidence': f"{prob:.2f}",
                'X_min': x_min,
                'Y_min': y_min,
                'X_max': x_max,
                'Y_max': y_max
            })
            # Draw bounding box in blue (BGR: 255,0,0), border width 1
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
            # Draw text in blue, smaller font, thickness 1
            cv2.putText(img, text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        if data:
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
        else:
            print('No text detected.')
        # Save result image
        result_img_path = os.path.join(results_dir, f"result_{img_name}")
        cv2.imwrite(result_img_path, img)
    except Exception as e:
        print(f'Error processing {img_path}: {e}')

print("\nExtract text from image process done.")
