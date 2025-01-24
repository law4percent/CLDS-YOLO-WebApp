import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
from ultralytics import YOLO
import numpy as np
import cv2


def fuzzy_severity(rla, diseased_count):
    # Define fuzzy variables
    rla_level = ctrl.Antecedent(np.arange(0, 101, 1), 'RLA')
    diseased_count_level = ctrl.Antecedent(np.arange(0, 20, 1), 'DiseasedCount')  # Assuming max count is 20
    severity = ctrl.Consequent(np.arange(0, 101, 1), 'Severity')

    # Define membership functions
    rla_level['low'] = fuzz.trapmf(rla_level.universe, [0, 0, 20, 40])
    rla_level['moderate'] = fuzz.trimf(rla_level.universe, [30, 50, 70])
    rla_level['high'] = fuzz.trapmf(rla_level.universe, [60, 80, 100, 100])

    diseased_count_level['low'] = fuzz.trapmf(diseased_count_level.universe, [0, 0, 3, 5])
    diseased_count_level['moderate'] = fuzz.trimf(diseased_count_level.universe, [4, 7, 10])
    diseased_count_level['high'] = fuzz.trapmf(diseased_count_level.universe, [8, 12, 20, 20])

    severity['low'] = fuzz.trapmf(severity.universe, [0, 0, 30, 50])
    severity['moderate'] = fuzz.trimf(severity.universe, [40, 60, 80])
    severity['high'] = fuzz.trapmf(severity.universe, [70, 90, 100, 100])

    # Define rules
    rule1 = ctrl.Rule(rla_level['low'] & diseased_count_level['low'], severity['low'])
    rule2 = ctrl.Rule(rla_level['low'] & diseased_count_level['moderate'], severity['moderate'])
    rule3 = ctrl.Rule(rla_level['low'] & diseased_count_level['high'], severity['high'])
    
    rule4 = ctrl.Rule(rla_level['moderate'] & diseased_count_level['low'], severity['moderate'])
    rule5 = ctrl.Rule(rla_level['moderate'] & diseased_count_level['moderate'], severity['high'])
    rule6 = ctrl.Rule(rla_level['moderate'] & diseased_count_level['high'], severity['high'])
    
    rule7 = ctrl.Rule(rla_level['high'] & diseased_count_level['low'], severity['high'])
    rule8 = ctrl.Rule(rla_level['high'] & diseased_count_level['moderate'], severity['high'])
    rule9 = ctrl.Rule(rla_level['high'] & diseased_count_level['high'], severity['high'])

    severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    severity_sim = ctrl.ControlSystemSimulation(severity_ctrl)

    # Compute severity
    severity_sim.input['RLA'] = rla
    severity_sim.input['DiseasedCount'] = diseased_count
    severity_sim.compute()
    return severity_sim.output['Severity'] if rla != 100 else 100.00


def RLA(total_diseased_area, total_leaf_area):
    if total_leaf_area == 0 or total_leaf_area < total_diseased_area:
         total_leaf_area = total_diseased_area
    return (total_diseased_area / total_leaf_area) * 100


def evualate_severity_level(severity):
    return "Low" if severity < 30 else "Moderate" if severity < 70 else "High"


def read_class_names(file_path: str):
        with open(file_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names


def check_exist_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def load_class_names(class_names_file_path):
    check_exist_file(class_names_file_path)
    return read_class_names(class_names_file_path)


def load_model(weights_file_path):
    check_exist_file(weights_file_path)
    return YOLO(weights_file_path, "v11")


def get_prediction_result(model, image):
    pred = model.predict(source=[image], save=False, task="segment")
    results = pred[0]
    masks = results.masks.data.numpy()
    boxes = results.boxes.data.numpy()
    return [masks, boxes]


def prediction_result(image, model, cls_names):
        masks, boxes = get_prediction_result(model, image)
        height, width, _ = image.shape  # Dynamically get image dimensions
        segmented_image = image.copy()
        total_leaf_area = 0
        total_diseased_area = 0
        diseased_count = 0
        total_leaf = 0
        
        for idx, box in enumerate(boxes):
            # Extract box coordinates and confidence
            x1, y1, x2, y2, conf_score, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf_score = "%.2f" % conf_score

            # Resize the mask to match the image dimensions
            mask = masks[idx]
            mask = cv2.resize(mask, (width, height))  # Ensure mask dimensions are (width, height)
            mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask

            # Create a 3-channel mask to match the image dimensions
            colored_mask = np.zeros_like(segmented_image)
            colored_mask[mask == 1] = [0, 255, 0] if cls == 1 else [0, 0, 255]  # Apply green for leaf, red for diseased

            # Calculate area of the mask
            area = np.sum(mask > 0.5)

            if cls == 1:  # Leaf class
                total_leaf_area += area
                total_leaf += 1
                colored_mask[mask > 0.5] = [0, 255, 0]  # Green color for leaf masks
            else:  # Diseased class
                total_diseased_area += area
                diseased_count += 1
                colored_mask[mask > 0.5] = [0, 0, 255]  # Red color for diseased masks

            segmented_image = cv2.addWeighted(segmented_image, 1, colored_mask, 0.5, 0)
            cv2.putText(segmented_image, f"{cls_names[int(cls)]} {conf_score}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

        return {
                "total leaf area": total_leaf_area,
                "total diseased area": total_diseased_area,
                "total diseased": diseased_count,
                "segmented image": segmented_image,
                "total leaf": total_leaf
            }
    