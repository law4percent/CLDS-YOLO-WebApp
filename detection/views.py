from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import os

from src.clds_lib import clds


cls_names = clds.load_class_names("src/class_names/class.names")
model = clds.load_model("src/model/best.pt")

def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded image
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(f"images/upload/{uploaded_file.name}", uploaded_file)
        image_path = fs.path(filename)
        image = cv2.imread(image_path)

        pd_result = clds.prediction_result(image, model, cls_names)
        numb_of_leaf = pd_result["total leaf"]
        segmented_img = pd_result["segmented image"]
        TDA = pd_result["total diseased area"]
        TLA = pd_result["total leaf area"]
        numb_of_diseased = pd_result["total diseased"]

        # Calculate RLA and severity
        rla = clds.RLA(TDA, TLA)
        severity = clds.fuzzy_severity(rla, numb_of_diseased)
        severity_level = clds.evualate_severity_level(severity)        

        # Save segmented image
        new_filename = f"segmented_{uploaded_file.name}"
        segmented_image_path = os.path.join(f"{fs.location}\\images\\segment", new_filename)
        cv2.imwrite(segmented_image_path, segmented_img)

        uploaded_image_url = fs.url(filename)
        segmented_image_url = fs.url(f"images/segment/{new_filename}")

        # ensure only one leaf
        severity_value = "%.2f" % severity if numb_of_diseased != 0 else "0.00"
        rla_value = "%.2f" % rla if numb_of_leaf == 1 else "error"

        if numb_of_leaf == 1:
            return render(request, 'detection/result.html', {
                'rla': rla_value,
                'severity': severity_value + "%",
                'severity_level': severity_level,
                'diseased_count': numb_of_diseased,
                'numb_of_leaf': numb_of_leaf,
                'uploaded_image_url': uploaded_image_url,
                'segmented_image_url': segmented_image_url,
            })

        return render(request, 'detection/result.html', {
            'rla': "error",
            'severity': "error",
            'severity_level': "error",
            'diseased_count': numb_of_diseased,
            'numb_of_leaf': numb_of_leaf,
            'uploaded_image_url': uploaded_image_url,
            'segmented_image_url': segmented_image_url,
        })

    return render(request, 'detection/result.html')
