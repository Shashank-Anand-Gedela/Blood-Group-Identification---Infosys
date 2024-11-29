'''import base64
from django.shortcuts import render
from io import BytesIO

def profile(request):
    img_base64 = None

    if request.method == 'POST':
        img_file = request.FILES.get('abd')

        if img_file:
            # Read the uploaded image file and convert it to Base64
            img_base64 = convert_image_to_base64(img_file)

        

    return render(request, 'profile.html', {'img': img_base64})

def convert_image_to_base64(image_file):
    # Read the uploaded image file and encode it to Base64
    img_data = image_file.read()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"
'''
'''

import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    img_base64 = None
    contour_length = None

    if request.method == 'POST':
        img_file = request.FILES.get('abd')

        if img_file:
            # Read image and calculate contour length, then encode to Base64
            img_base64, contour_length = process_image_and_calculate_contour(img_file)

    return render(request, 'profile.html', {
        'img': img_base64,
        'contour_length': contour_length
    })

def process_image_and_calculate_contour(image_file):
    # Read the image data
    img_data = image_file.read()

    # Convert image data to numpy array for OpenCV processing
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total contour length
    contour_length = sum(cv2.arcLength(contour, False) for contour in contours)

    # Encode image to Base64 for display
    img_base64 = convert_image_to_base64(img_data)

    return img_base64, contour_length

def convert_image_to_base64(img_data):
    # Encode image data to Base64 for display in HTML
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

'''
'''
import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    img_base64 = None
    morphology_result = None

    if request.method == 'POST':
        img_file = request.FILES.get('abd')

        if img_file:
            # Read image and calculate morphology, then encode to Base64
            img_base64, morphology_result = process_image_and_morphology(img_file)

    return render(request, 'profile.html', {
        'img': img_base64,
        'morphology_result': morphology_result
    })

def process_image_and_morphology(image_file):
    # Read the image data
    img_data = image_file.read()

    # Convert image data to numpy array for OpenCV processing
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological operation (e.g., closing)
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel for the morphological operation
    morph_result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Count non-zero pixels (which are part of the morphological result)
    morphology_result = np.count_nonzero(morph_result)

    # Encode image to Base64 for display
    img_base64 = convert_image_to_base64(img_data)

    return img_base64, morphology_result

def convert_image_to_base64(img_data):
    # Encode image data to Base64 for display in HTML
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    img_base64 = None
    morphological_output = None

    if request.method == 'POST':
        img_file = request.FILES.get('abd')

        if img_file:
            # Process image and perform morphology, then encode to Base64
            img_base64, morphological_output = process_image_and_perform_morphology(img_file)

    return render(request, 'profile.html', {
        'img': img_base64,
        'morphological_output': morphological_output
    })

def process_image_and_perform_morphology(image_file):
    # Read the image data
    img_data = image_file.read()

    # Convert image data to numpy array for OpenCV processing
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological operations: Dilation and Erosion
    kernel = np.ones((5,5), np.uint8)  # Define a kernel for morphology

    # Dilation
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Erosion
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Stack the original, dilated, and eroded images for display
    morphological_result = np.hstack((thresh, dilated, eroded))

    # Encode the morphological result image to Base64
    img_base64 = convert_image_to_base64(morphological_result)

    # Create a textual representation of the morphological result (optional)
    morphological_output = "Morphological operations (Dilation and Erosion) applied successfully."

    return img_base64, morphological_output

def convert_image_to_base64(img_data):
    # Encode image data to Base64 for display in HTML
    _, img_encoded = cv2.imencode('.jpg', img_data)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    img_base64 = None
    morphological_output = None

    if request.method == 'POST':
        img_file = request.FILES.get('abd')  # Get the uploaded image file

        if img_file:
            # Process image and perform morphology, then encode to Base64
            img_base64, morphological_output = process_image_and_perform_morphology(img_file)

    return render(request, 'profile.html', {
        'img': img_base64,
        'morphological_output': morphological_output
    })

def process_image_and_perform_morphology(image_file):
    # Read the image data
    img_data = image_file.read()

    # Convert image data to numpy array for OpenCV processing
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological operations: Dilation and Erosion
    kernel = np.ones((5,5), np.uint8)  # Define a kernel for morphology

    # Dilation
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Erosion
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Stack the original, dilated, and eroded images for display
    morphological_result = np.hstack((thresh, dilated, eroded))

    # Resize the image to make it bigger before converting to Base64
    morphological_result_resized = cv2.resize(morphological_result, (1200, 400))  # Adjust size as needed

    # Encode the morphological result image to Base64
    img_base64 = convert_image_to_base64(morphological_result_resized)

    # Create a textual representation of the morphological result (optional)
    morphological_output = "Morphological operations (Dilation and Erosion) applied successfully."

    return img_base64, morphological_output

def convert_image_to_base64(img_data):
    # Encode image data to Base64 for display in HTML
    _, img_encoded = cv2.imencode('.jpg', img_data)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    # Initialize default values for both GET and POST requests
    img_base64 = None
    mor_img_url = None
    obj = "No image uploaded."
    img_url = None  # Initialize img_url to avoid the 'UnboundLocalError'

    if request.method == 'POST':
        img_file = request.FILES.get('abd')  # Get the uploaded image file

        if img_file:
            try:
                # Read the image directly from the uploaded file in memory
                img_data = img_file.read()

                # Convert image data to numpy array for OpenCV processing
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

                # Convert the image into grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

                # Enhance contrast of the image
                enhance_img = cv2.equalizeHist(blur_img)

                # Threshold calculation (Otsu's thresholding)
                _, bin_img = cv2.threshold(enhance_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Perform morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

                # Divide the image into three regions
                hei, wid = bin_img.shape
                mid_wid = wid // 3
                region_A = bin_img[0:mid_wid]
                region_B = bin_img[mid_wid:2 * mid_wid]
                region_D = bin_img[2 * mid_wid:]

                # Function to calculate agglutination in each region
                def cal_agg(region):
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
                    return num_labels - 1

                # Calculate agglutination for each region
                num_region_A = cal_agg(region_A)
                num_region_B = cal_agg(region_B)
                num_region_D = cal_agg(region_D)

                # Assign object result (can be extended to return actual blood group info)
                obj = f"Region A: {num_region_A}, Region B: {num_region_B}, Region D: {num_region_D}"

                # Convert the morphological binary image to base64
                _, mor_img_encoded = cv2.imencode('.jpg', bin_img)
                mor_img_base64 = base64.b64encode(mor_img_encoded).decode('utf-8')
                mor_img_url = f"data:image/jpeg;base64,{mor_img_base64}"

                # Convert the original uploaded image to base64 for display
                _, img_encoded = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                img_url = f"data:image/jpeg;base64,{img_base64}"

            except Exception as e:
                obj = f"Error processing the image: {str(e)}"
        else:
            obj = "No image uploaded. Please upload an image for analysis."

    # Return the rendered template with the necessary context
    return render(request, 'profile.html', {
        'img': img_url,
        'mor_img': mor_img_url,
        'obj': obj
    })

'''

import base64
import cv2
import numpy as np
from django.shortcuts import render
from io import BytesIO

def profile(request):
    # Initialize default values for both GET and POST requests
    img_base64 = None
    mor_img_url = None
    obj = "No image uploaded."
    img_url = None  # Initialize img_url to avoid the 'UnboundLocalError'

    if request.method == 'POST':
        img_file = request.FILES.get('abd')  # Get the uploaded image file

        if img_file:
            try:
                # Read the image directly from the uploaded file in memory
                img_data = img_file.read()

                # Convert image data to numpy array for OpenCV processing
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

                # Convert the image into grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

                # Enhance contrast of the image
                enhance_img = cv2.equalizeHist(blur_img)

                # Threshold calculation (Otsu's thresholding)
                _, bin_img = cv2.threshold(enhance_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Perform morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

                # Divide the image into three regions
                hei, wid = bin_img.shape
                mid_wid = wid // 3
                region_A = bin_img[:, :mid_wid]
                region_B = bin_img[:, mid_wid:2 * mid_wid]
                region_D = bin_img[:, 2 * mid_wid:]

                # Function to calculate agglutination in each region
                def cal_agg(region):
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
                    return num_labels - 1

                # Calculate agglutination for each region
                num_region_A = cal_agg(region_A)
                num_region_B = cal_agg(region_B)
                num_region_D = cal_agg(region_D)

                # Determine blood group based on conditions
                if num_region_A > 0 and num_region_B == 0:
                    obj = "Blood Group: A"
                elif num_region_A == 0 and num_region_B > 0:
                    obj = "Blood Group: B"
                elif num_region_A > 0 and num_region_B > 0:
                    obj = "Blood Group: AB"
                elif num_region_A == 0 and num_region_B == 0:
                    obj = "Blood Group: O"
                else:
                    obj = "Blood Group: Unknown"

                # Determine Rh factor
                if num_region_D > 0:
                    obj += " Positive"
                else:
                    obj += " Negative"

                # Convert the morphological binary image to base64
                _, mor_img_encoded = cv2.imencode('.jpg', bin_img)
                mor_img_base64 = base64.b64encode(mor_img_encoded).decode('utf-8')
                mor_img_url = f"data:image/jpeg;base64,{mor_img_base64}"

                # Convert the original uploaded image to base64 for display
                _, img_encoded = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                img_url = f"data:image/jpeg;base64,{img_base64}"

            except Exception as e:
                obj = f"Error processing the image: {str(e)}"
        else:
            obj = "No image uploaded. Please upload an image for analysis."

    # Return the rendered template with the necessary context
    return render(request, 'profile.html', {
        'img': img_url,
        'mor_img': mor_img_url,
        'obj': obj
    })

