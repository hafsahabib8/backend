import gdown
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime, timedelta
import random
import base64
import cv2
import tempfile
import jwt
import os
import tensorflow as tf
import requests
import shutil
import zipfile
import io
import json
from datetime import datetime
import pydicom  # Make sure you have this installed
from io import BytesIO




app = Flask(__name__)
CORS(app)

# Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "15u2GY0686FkJ0Dmzu6d8BMxMAB7haBGN"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Check if model exists, otherwise download
if not os.path.exists(MODEL_PATH):
    try:
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    except Exception as download_err:
        print(f"❌ Failed to download model: {download_err}")

# Attempt to load the model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print("✅ Model loaded successfully!")
    except Exception as load_err:
        print(f"❌ Model loading failed: {load_err}")
        model = None
else:
    print("❌ Model file still not found after attempting download.")
    model = None

# Store image results
scanned_images = {modality: {} for modality in ["T1N", "T1C", "T2W", "T2F"]}

# PACS/Orthanc Configuration
ORTHANC_URL = "http://localhost:8042"  # Default Orthanc server URL
ORTHANC_AUTH = ('orthanc', 'orthanc')  # Default Orthanc credentials
PACS_ENABLED = True  # Flag to enable/disable PACS functionality


SECRET_KEY = "your_secret_key"  # Replace with your actual secret key

def decode_jwt_token(auth_token):
    try:
        # Split the token from the header
        token_parts = auth_token.split()
        if len(token_parts) == 2 and token_parts[0].lower() == 'bearer':
            token = token_parts[1]
            
            # Decode and verify the token using the secret key
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return decoded.get('id')  # Assuming the 'id' is in the payload
    except jwt.ExpiredSignatureError:
        print("❌ Token has expired")
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
    except Exception as e:
        print(f"❌ Error decoding token: {e}")
    
    return None

def preprocess_nii(file_path, modality):
    try:
        nii_image = nib.load(file_path)
        img_data = nii_image.get_fdata()

        axial_slices = [img_data[:, :, i] for i in range(img_data.shape[2])]
        coronal_slices = [img_data[:, i, :] for i in range(img_data.shape[1])]
        sagittal_slices = [img_data[i, :, :] for i in range(img_data.shape[0])]

        def encode_images(img_list):
            encoded = []
            for img in img_list:
                if np.max(img) > 0:
                    img = (img / np.max(img)) * 255
                img_resized = cv2.resize(img.astype(np.uint8), (128, 128))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

                if model:
                    input_img = img_resized / 255.0
                    input_img = np.expand_dims(input_img, axis=(0, -1))  # (1, 128, 128, 1)
                    input_img = np.repeat(input_img, 4, axis=-1)

                    pred = model.predict(input_img)[0]
                    pred_mask = np.argmax(pred, axis=-1)
                    pred_mask = (pred_mask / np.max(pred_mask) if np.max(pred_mask) > 0 else pred_mask) * 255
                    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (128, 128))

                    mask_rgb = np.zeros_like(img_rgb)
                    mask_rgb[:, :, 0] = pred_mask * 0.5
                    mask_rgb[:, :, 2] = pred_mask

                    img_overlay = cv2.addWeighted(img_rgb, 1, mask_rgb, 0.5, 0)
                else:
                    img_overlay = img_rgb

                _, buffer = cv2.imencode(".png", img_overlay)
                encoded.append(base64.b64encode(buffer).decode("utf-8"))

            return encoded

        scanned_images[modality] = {
            "axial": encode_images(axial_slices),
            "coronal": encode_images(coronal_slices),
            "sagittal": encode_images(sagittal_slices),
        }

    except Exception as e:
        print(f"❌ Error processing NIfTI file: {e}")


@app.route("/upload-scans", methods=["POST"])
def upload_scans():
    uploaded = {}
    
    # Get user information from the request
    user_id = request.form.get('userId')
    auth_token = request.headers.get('Authorization')
    
    if not user_id and auth_token:
        try:
            # Extract user ID from token
            token_parts = auth_token.split()
            if len(token_parts) == 2 and token_parts[0].lower() == 'bearer':
              
                decoded = jwt.decode(token_parts[1], options={"verify_signature": False})
                user_id = decoded.get('id')
                print(f"Extracted user ID from token: {user_id}")
        except Exception as e:
            print(f"❌ Error extracting user ID from token: {e}")
    
    if not user_id:
        return jsonify({"error": "Missing user ID"}), 401
    
    for modality in ["T1N", "T1C", "T2W", "T2F"]:
        file = request.files.get(modality)
        if file and file.filename.endswith(".nii") or (file and file.filename.endswith(".nii.gz")):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
            try:
                file.save(temp.name)
                temp.close()  # Ensure the file is closed before processing
                preprocess_nii(temp.name, modality)
                
                # Save file info to Node.js backend
                if auth_token:
                    try:
                        # Save original file name
                        file_name = file.filename
                        
                        # Save the file to Node.js uploads directory
                        try:
                            node_uploads_dir = r"C:\Users\PMYLS\Desktop\new\server\uploads"  # Use the absolute path
                            if not os.path.exists(node_uploads_dir):
                                os.makedirs(node_uploads_dir)
    
                            dest_path = os.path.join(node_uploads_dir, file_name)
                            shutil.copy2(temp.name, dest_path)
                            print(f"✅ File copied to Node.js uploads: {dest_path}")
                        except Exception as e:
                            print(f"❌ Error copying file to Node.js: {str(e)}")
                        
                        payload = {
                            "userId": user_id,
                            "modality": modality,
                            "fileName": file_name
                        }
                        
                        node_response = requests.post(
                            "http://localhost:5001/register-scan",
                            json=payload,
                            headers={"Authorization": auth_token}
                        )
                        
                        if node_response.status_code == 200:
                            uploaded[modality] = "Processed and saved to database"
                        else:
                            uploaded[modality] = f"Processed but DB save failed: {node_response.text}"
                    except Exception as e:
                        uploaded[modality] = f"Processed but DB error: {str(e)}"
                else:
                    uploaded[modality] = "Processed but not saved (no auth token)"
            except Exception as e:
                uploaded[modality] = f"Error: {str(e)}"
            finally:
                try:
                    os.remove(temp.name)
                except Exception as e:
                    print(f"⚠️ Error deleting temp file: {e}")
        else:
            if file:
                uploaded[modality] = "Invalid file format (must be .nii or .nii.gz)"
            else:
                uploaded[modality] = "Missing file"
    
    return jsonify({"message": "Scans processed", "result": uploaded})

@app.route("/get-segmentation-results/<modality>", methods=["GET"])
def get_segmentation_results(modality):
    if modality not in scanned_images or not scanned_images[modality]:
        return jsonify({"error": "No results found"}), 404
    return jsonify(scanned_images[modality])

@app.route('/check-tumor-existence', methods=['GET'])
def check_tumor_existence():
    global scanned_images

    if not scanned_images["T1N"]["axial"]:
        return jsonify({"error": "No scans available for analysis"}), 400

    try:
        image_base64 = scanned_images["T1N"]["axial"][78]
        image_data = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        image_resized = cv2.resize(image, (128, 128)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=[0, -1])
        image_resized = np.repeat(image_resized, 4, axis=-1)

        prediction = model.predict(image_resized)
        print(f"⚡ Model Prediction: {prediction}")

        predicted_class = np.argmax(prediction[0])
        tumor_exists = predicted_class > 0

        return jsonify({"tumor_exists": bool(tumor_exists), "predicted_class": int(predicted_class)})

    except Exception as e:
        print(f"❌ Error in tumor detection: {e}")
        return jsonify({"error": "Failed to process the image"}), 500
@app.route('/')
def home():
    return "Welcome!"


@app.route("/create-segmentation-zip/<modality>", methods=["GET"])
def create_segmentation_zip(modality):
    if modality not in scanned_images or not scanned_images[modality]:
        return jsonify({"error": "No results found for this modality"}), 404
    
    try:
        # Create in-memory ZIP file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add the segmentation results as a JSON file
            zf.writestr(f"{modality}_segmentation_data.json", json.dumps(scanned_images[modality]))
            
            # Add individual PNG images
            for view in ["axial", "coronal", "sagittal"]:
                for i, img_base64 in enumerate(scanned_images[modality][view]):
                    img_data = base64.b64decode(img_base64)
                    zf.writestr(f"{modality}/{view}/slice_{i:03d}.png", img_data)
        
        # Return the zip file
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{modality}_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
    except Exception as e:
        print(f"❌ Error creating ZIP file: {e}")
        return jsonify({"error": f"Failed to create ZIP file: {str(e)}"}), 500


@app.route("/save-results", methods=["POST"])
def save_results():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        modality = data.get('modality')
        
        if not user_id or not modality:
            return jsonify({"error": "Missing user ID or modality"}), 400
            
        print(f"📤 Saving segmentation results for user {user_id}, modality: {modality}")
        
        # Create ZIP file in memory
        memory_file = io.BytesIO()
        try:
            with zipfile.ZipFile(memory_file, 'w') as zf:
                # Add metadata information
                metadata = {
                    "userId": user_id,
                    "modality": modality,
                    "createdAt": datetime.now().isoformat(),
                    "version": "1.0"
                }
                zf.writestr("metadata.json", json.dumps(metadata))
                
                # Add the segmentation results as a JSON file
                if modality in scanned_images and scanned_images[modality]:
                    # Add individual PNG images
                    for view in ["axial", "coronal", "sagittal"]:
                        for i, img_base64 in enumerate(scanned_images.get(modality, {}).get(view, [])):
                            img_data = base64.b64decode(img_base64)
                            zf.writestr(f"{modality}/{view}/slice_{i:03d}.png", img_data)
            
            memory_file.seek(0)
            zip_data = base64.b64encode(memory_file.read()).decode('utf-8')
            
            # Check zip data size
            zip_size_mb = len(zip_data) / (1024 * 1024)
            print(f"📦 ZIP data size: {zip_size_mb:.2f} MB")
            
            # Current timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f"{modality}_segmentation_{timestamp}.zip"
            
        except Exception as e:
            print(f"❌ Error creating ZIP file: {str(e)}")
            return jsonify({"error": f"ZIP creation failed: {str(e)}"}), 500
        
        # Pass auth token if available
        headers = {}
        auth_token = request.headers.get('Authorization')
        if auth_token:
            headers['Authorization'] = auth_token
            print(f"📤 Using auth token: {auth_token[:15]}...")
        else:
            print("⚠️ No auth token provided")
            return jsonify({"error": "Authentication required"}), 401

        # Send data to Node.js backend - now we only send the ZIP data, not the results JSON
        try:
            print(f"📤 Sending data to Node.js backend")
            response = requests.post(
                "http://localhost:5001/save-results", 
                json={
                    "userId": user_id,
                    "modality": modality,
                    "zipData": zip_data,
                    "zipFilename": zip_filename
                },
                headers=headers
            )

            print(f"📥 Node.js response status: {response.status_code}")
            if response.status_code == 200:
                resp_data = response.json()
                print(f"📥 Node.js response data: {resp_data}")
                return jsonify({
                    "message": "Data saved to database", 
                    "zipFilename": zip_filename,
                    "downloadUrl": resp_data.get("zipFileUrl")
                })
            else:
                print(f"❌ Node.js error: {response.text}")
                return jsonify({"error": "Node.js error", "details": response.text}), response.status_code
        except Exception as e:
            print(f"❌ Error sending data to Node.js: {str(e)}")
            return jsonify({"error": f"Communication error: {str(e)}"}), 500
    except Exception as e:
        print("❌ Exception saving results:", str(e))
        return jsonify({"error": str(e)}), 500

# Add this endpoint to your existing Flask app.py file:

@app.route("/download-segmentation/<filename>", methods=["GET"])
def download_segmentation(filename):
    """
    Downloads a previously generated segmentation ZIP file
    """
    try:
        # Path where segmentation ZIP files are stored
        # You might need to adjust this path based on your file storage location
        uploads_dir = os.path.join(os.path.dirname(__file__), 'segmentation_files')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        file_path = os.path.join(uploads_dir, filename)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            # If file doesn't exist, check if it's stored in memory
            # This is a fallback since the save-results endpoint may have stored it
            auth_token = request.headers.get('Authorization')
            if auth_token:
                # Try to retrieve data from Node.js backend
                response = requests.get(
                    f"http://localhost:5001/segmentation-file/{filename}",
                    headers={"Authorization": auth_token}
                )
                
                if response.status_code == 200:
                    memory_file = io.BytesIO(response.content)
                    return send_file(
                        memory_file,
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name=filename
                    )
            
            return jsonify({"error": "File not found"}), 404
        
        return send_file(
            file_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"❌ Error downloading segmentation file: {e}")
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 500


@app.route("/generate-combined-segmentation", methods=["POST"])
def generate_combined_segmentation():
    """
    Generates a combined segmentation ZIP file from multiple modalities
    """
    try:
        data = request.get_json()
        modalities = data.get('modalities', [])
        user_id = data.get('userId')
        
        if not modalities:
            return jsonify({"error": "No modalities specified"}), 400
        
        if not user_id:
            # Try to get user ID from token
            auth_token = request.headers.get('Authorization')
            if auth_token:
                try:
                    token_parts = auth_token.split()
                    if len(token_parts) == 2 and token_parts[0].lower() == 'bearer':
                        import jwt
                        decoded = jwt.decode(token_parts[1], options={"verify_signature": False})
                        user_id = decoded.get('id')
                except Exception as e:
                    print(f"❌ Error extracting user ID from token: {e}")
            
            if not user_id:
                return jsonify({"error": "User ID not provided"}), 400
        
        # Create a combined ZIP file with all specified modalities
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add metadata
            metadata = {
                "userId": user_id,
                "generatedAt": datetime.now().isoformat(),
                "modalities": modalities
            }
            zf.writestr("metadata.json", json.dumps(metadata))
            
            # Add each modality's data
            for modality in modalities:
                if modality in scanned_images and scanned_images[modality]:
                    # Add the segmentation results as a JSON file
                    zf.writestr(f"{modality}/data.json", json.dumps(scanned_images[modality]))
                    
                    # Add individual PNG images
                    for view in ["axial", "coronal", "sagittal"]:
                        for i, img_base64 in enumerate(scanned_images[modality].get(view, [])):
                            img_data = base64.b64decode(img_base64)
                            zf.writestr(f"{modality}/{view}/slice_{i:03d}.png", img_data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"combined_segmentation_{timestamp}.zip"
        
        # Save file for future access
        uploads_dir = os.path.join(os.path.dirname(__file__), 'segmentation_files')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        file_path = os.path.join(uploads_dir, filename)
        memory_file.seek(0)
        with open(file_path, 'wb') as f:
            f.write(memory_file.getbuffer())
        
        # Return the zip file
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"❌ Error creating combined segmentation: {e}")
        return jsonify({"error": f"Failed to generate combined segmentation: {str(e)}"}), 500
    
    # Add these endpoints to your existing Flask app.py file:

@app.route("/generate-segmentation-images/<modality>", methods=["GET"])
def generate_segmentation_images(modality):
    """
    Generates a ZIP file containing only the segmentation images for a specific modality
    """
    if modality not in scanned_images or not scanned_images[modality]:
        return jsonify({"error": "No results found for this modality"}), 404
    
    try:
        # Create in-memory ZIP file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add individual PNG images for each view
            for view in ["axial", "coronal", "sagittal"]:
                view_images = scanned_images[modality].get(view, [])
                if not view_images:
                    continue
                    
                for i, img_base64 in enumerate(view_images):
                    # Decode base64 image
                    img_data = base64.b64decode(img_base64)
                    
                    # Save each image with a meaningful filename
                    image_filename = f"{modality}_{view}_slice_{i:03d}.png"
                    zf.writestr(image_filename, img_data)
        
        # Create timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_filename = f"{modality}_segmentation_{timestamp}.zip"
        
        # Return the zip file
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_filename
        )
    except Exception as e:
        print(f"❌ Error creating segmentation images: {e}")
        return jsonify({"error": f"Failed to generate segmentation images: {str(e)}"}), 500


@app.route("/generate-combined-segmentation-images", methods=["POST"])
def generate_combined_segmentation_images():
    """
    Generates a ZIP file containing segmentation images from multiple modalities
    """
    try:
        data = request.get_json()
        modalities = data.get('modalities', [])
        
        if not modalities:
            return jsonify({"error": "No modalities specified"}), 400
        
        # Create in-memory ZIP file for combined segmentation images
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Process each modality
            for modality in modalities:
                if modality not in scanned_images or not scanned_images[modality]:
                    continue
                
                # Create a subdirectory for each modality
                for view in ["axial", "coronal", "sagittal"]:
                    view_images = scanned_images[modality].get(view, [])
                    if not view_images:
                        continue
                    
                    # Add images for this view
                    for i, img_base64 in enumerate(view_images):
                        # Only save every 5th image to reduce file size (adjust as needed)
                        if i % 5 == 0 or i == len(view_images) - 1:  # Include last slice always
                            img_data = base64.b64decode(img_base64)
                            image_path = f"{modality}/{view}/slice_{i:03d}.png"
                            zf.writestr(image_path, img_data)
                
                # Include a preview image for each modality (middle slice of axial view)
                axial_images = scanned_images[modality].get("axial", [])
                if axial_images:
                    middle_idx = len(axial_images) // 2
                    middle_img_data = base64.b64decode(axial_images[middle_idx])
                    zf.writestr(f"{modality}_preview.png", middle_img_data)
            
            # Generate an HTML viewer for easy navigation (optional)
            html_content = generate_html_viewer(modalities)
            if html_content:
                zf.writestr("viewer.html", html_content)
        
        # Create timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_filename = f"combined_segmentation_{timestamp}.zip"
        
        # Return the zip file
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_filename
        )
    except Exception as e:
        print(f"❌ Error creating combined segmentation images: {e}")
        return jsonify({"error": f"Failed to generate combined segmentation images: {str(e)}"}), 500

# Example function in app.py for sending results
def save_segmentation_results(user_id, modality, results, zip_buffer):
    # Encode the zip file as base64
    try:
        zip_data = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
        zip_filename = f"segmentation_{modality}_{user_id}_{int(time.time())}.zip"
        
        print("✅ Sending zipFilename:", zip_filename)
        print("✅ Zip data size (chars):", len(zip_data))
        
        # Send data to Node.js backend
        response = requests.post(
            "http://localhost:5001/save-results",
            headers={"Authorization": f"Bearer {session['token']}"},
            json={
                "userId": user_id,
                "modality": modality,
                "results": results,
                "zipData": zip_data,
                "zipFilename": zip_filename
            }
        )
        
        return response.json()
    except Exception as e:
        print(f"❌ Error in save_segmentation_results: {e}")
        return {"error": str(e)}

def generate_html_viewer(modalities):
    """
    Generate a simple HTML viewer for the segmentation images
    """
    try:
        # Create a simple HTML file to view the images
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Segmentation Viewer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .modality-section { background-color: white; margin-bottom: 30px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .view-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px; }
                .view-box { flex: 1; min-width: 300px; }
                .preview-img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .image-list { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; }
                .image-item { padding: 5px; cursor: pointer; }
                .image-item:hover { background-color: #e9e9e9; }
                #image-display { width: 100%; border: 2px solid #333; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Brain Segmentation Results</h1>
                <p>This file contains segmentation results for brain MRI scans.</p>
        """
        
        for modality in modalities:
            if modality not in scanned_images or not scanned_images[modality]:
                continue
                
            html += f"""
                <div class="modality-section">
                    <h2>{modality} Modality</h2>
                    <div class="view-container">
            """
            
            # Add sections for each view
            for view in ["axial", "coronal", "sagittal"]:
                view_images = scanned_images[modality].get(view, [])
                if not view_images:
                    continue
                    
                html += f"""
                        <div class="view-box">
                            <h3>{view.capitalize()} View</h3>
                            <div class="image-list">
                """
                
                # Add links to each image
                for i in range(0, len(view_images), 5):  # List every 5th image
                    if i < len(view_images):
                        html += f'                <div class="image-item" onclick="displayImage(\'{modality}/{view}/slice_{i:03d}.png\')">{view.capitalize()} Slice {i}</div>\n'
                
                html += """
                            </div>
                        </div>
                """
                
            html += """
                    </div>
                </div>
            """
        
        # Add JavaScript to display selected images
        html += """
                <div id="selected-image">
                    <h2>Selected Image:</h2>
                    <img id="image-display" src="" alt="Select an image to view" style="display: none;">
                </div>
                
                <script>
                function displayImage(imagePath) {
                    const img = document.getElementById('image-display');
                    img.src = imagePath;
                    img.style.display = 'block';
                }
                </script>
            </div>
        </body>
        </html>
        """
        
        return html
    except Exception as e:
        print(f"❌ Error generating HTML viewer: {e}")
        return None
    # Mock data generator
def generate_model_performance():
    # Load Excel file
    df = pd.read_csv('training_log.csv')  # Update filename if needed

    epochs = df['epoch'].tolist()

    return {
        "epochs": epochs,
        "history": {
            "loss": df['loss'].tolist(),
            "val_loss": df['val_loss'].tolist(),
            "accuracy": df['accuracy'].tolist(),
            "val_accuracy": df['val_accuracy'].tolist(),
            "dice_coefficient": df['dice_coef'].tolist(),  # or val_dice_coef
        },
        "current_metrics": {
            "precision": float(df['val_precision'].iloc[-1]),
            "recall": float(df['val_sensitivity'].iloc[-1]),  # sensitivity == recall
            "f1_score": float(df['val_dice_coef'].iloc[-1]),  # Dice ≈ F1
            "iou": float(df['val_mean_io_u'].iloc[-1]),
        },
        "model_info": {
            "name": "3D-UNet Brain Tumor Segmentation",
            "framework": "TensorFlow/Keras",
            "last_updated": datetime.fromtimestamp(os.path.getmtime("model.h5")).isoformat()
        }
    }

@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    return jsonify(generate_model_performance())

@app.route('/orthanc/studies', methods=['GET'])
def list_orthanc_studies():
    try:
        response = requests.get(f"{ORTHANC_URL}/studies", auth=ORTHANC_AUTH)
        study_ids = response.json()
        studies = []

        for sid in study_ids:
            meta = requests.get(f"{ORTHANC_URL}/studies/{sid}", auth=ORTHANC_AUTH).json()
            studies.append({
                "StudyID": sid,
                "PatientName": meta.get("MainDicomTags", {}).get("PatientName", "Unknown"),
                "StudyDate": meta.get("MainDicomTags", {}).get("StudyDate", "N/A"),
            })

        return jsonify(studies)

    except Exception as e:
        return jsonify({"error": f"Failed to fetch studies: {str(e)}"}), 500

@app.route('/orthanc/segment/<instance_id>', methods=['GET'])
def segment_dicom_from_orthanc(instance_id):
    try:
        dicom_url = f"{ORTHANC_URL}/instances/{instance_id}/file"
        dicom_response = requests.get(dicom_url, auth=ORTHANC_AUTH)

        # Read the DICOM image
        dicom_data = pydicom.dcmread(BytesIO(dicom_response.content))
        image = dicom_data.pixel_array.astype(np.float32)

        # Preprocess the image
        image_resized = cv2.resize(image, (128, 128)) / 255.0  # Normalize to [0, 1]
        image_resized = np.expand_dims(image_resized, axis=[0, -1])    # Shape becomes (1, 128, 128, 1)
        image_resized = np.repeat(image_resized, 4, axis=-1)           # Changed from 2 to 4 channels as expected by the model

        # Predict using the model
        prediction = model.predict(image_resized)
        predicted_class = int(np.argmax(prediction[0]))
        tumor_exists = predicted_class > 0

        # Convert processed image to base64 so it can be displayed in the frontend
        _, buffer = cv2.imencode('.png', (image_resized[0, :, :, 0] * 255).astype(np.uint8))
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Return both image and prediction results
        return jsonify({
            "tumor_exists": tumor_exists,
            "predicted_class": predicted_class,
            "image": base64_image
        })

    except Exception as e:
        print(f"Error processing DICOM: {e}")
        return jsonify({"error": f"Failed to process DICOM image: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

