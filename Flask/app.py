from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from tensorflow.keras.utils import CustomObjectScope
import os
import numpy as np
import tensorflow as tf
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import random
from patchify import patchify
import subprocess
import time

""" Global parameters """
H = 256
W = 256

cf = {}
cf["image_size"] = 256
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)

hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 8
hp["lr"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 3
hp["class_names"] = ["Benign", "Malignant", "Normal"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 2304 #3072
hp["num_heads"] = 6
hp["dropout_rate"] = 0.1

@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # Add **kwargs here
        super().__init__(**kwargs)  # Pass **kwargs to super().__init__

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

# Define a small smoothing constant
smooth = 1e-15

UNREAL_PATH = r"D:/UE_5.5/Engine/Binaries/Win64/UnrealEditor.exe"

# Replace this with the path to your Unreal project file (.uproject)
UNET_PATH = r"C:/Users/Admin/Downloads/images2d/images2d.uproject"
UNETR_PATH = r"C:/Users/Admin/Downloads/UNETR/UNETR.uproject"
UNETR3D_PATH = r"C:/Users/Admin/Downloads/Udaan/unetr3d/unetr.uproject"

file1_path = None
file2_path = None

def open_unreal(path):
    try:
        # Open Unreal Editor with the specified project
        subprocess.Popen([UNREAL_PATH, path,"-game"])
        return True
    except Exception as e:
        return False

# Custom metrics and loss functions
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
     #, supports_credentials=True, resources={r"/*": {"origins": "http://172.16.20.125:3000"}})

# Load the pre-trained model
unet_path = os.path.join("", "model.keras")  # Update this path to your model file
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(unet_path)

unetr_path = os.path.join("", "unetrmodel.keras")
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    unetr_model = tf.keras.models.load_model(unetr_path)

# Create a directory if it does not exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_image_and_mask(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        # mask = cv2.flip(mask, 1)
    
    # Random vertical flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        # mask = cv2.flip(mask, 0)
    
    # Random rotation (90, 180, 270 degrees)
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    # mask = np.rot90(mask, k)

    return image

def save_results_3d(image, y_pred, save_image_path):
    # Convert the ground truth mask to RGB for visualization
    # mask = np.expand_dims(mask, axis=-1)
    # mask_rgb = np.concatenate([mask, mask, mask], axis=-1)

    # Convert prediction to RGB (for visualization purposes) and scale to 255
    y_pred_rgb = np.expand_dims(y_pred, axis=-1)
    y_pred_rgb = np.concatenate([y_pred_rgb, y_pred_rgb, y_pred_rgb], axis=-1) * 255

    # Create a red overlay where the predicted mask is 1
    overlay = image.copy()  # Start with the original image
    red_color = (0, 0, 255)  # Red color for the overlay

    # Apply the red overlay to the mask region where prediction is 1
    overlay[y_pred == 1] = red_color  # Apply red where prediction is 1

    # Blend the original image and the overlay with some transparency
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)  # Alpha blending: 0.7 original, 0.3 overlay

    # Create a white line for separation
    # line = np.ones((H, 10, 3)) * 255

    # Concatenate the original + overlay, ground truth mask, and predicted mask side by side
    cat_images = np.concatenate([blended], axis=1)

    # Add labels to the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color for text
    thickness = 1

    # Save the final concatenated image
    cv2.imwrite(save_image_path, cat_images)

def convert_nifti_to_png(nifti_file, output_folder,output_subfolder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    os.makedirs(output_subfolder, exist_ok=True)

    # Load the NIFTI file
    img = nib.load(nifti_file)
    img_data = img.get_fdata()

    # Save each slice along the x-axis as a PNG
    for i in range(img_data.shape[0]):
        slice_data = img_data[i, :, :]

        # Normalize to 0-255 for PNG format
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = (slice_data * 255).astype(np.uint8)

        # Save the slice as PNG
        output_path = os.path.join(output_subfolder, f'slice_{i+1}.png')
        plt.imsave(output_path, slice_data, cmap='gray')

    print(f'Converted {nifti_file} to PNGs in {output_subfolder}')

# # Input and output folder paths
def convert_color_png_folder_to_nifti(png_folder, output_nifti_path,affine):
    # Get all PNG files in the folder and sort them
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])

    if not png_files:
        print(f"No PNG files found in {png_folder}.")
        return

    # Load the first PNG to determine image size
    first_img = Image.open(os.path.join(png_folder, png_files[0]))
    img_shape = first_img.size  # (width, height)

    # Create an empty 3D volume to store the red channel slices
    volume = np.zeros((len(png_files), img_shape[1], img_shape[0]), dtype=np.uint8)

    # Load each PNG and extract the red channel
    for i, png_file in enumerate(png_files):
        img = Image.open(os.path.join(png_folder, png_file)).convert('RGB')
        red_channel = np.array(img)[:, :, 0]  # Extract the red channel
        volume[i, :, :] = red_channel

    # Create a NIFTI image from the 3D volume
    nifti_img = nib.Nifti1Image(volume, affine=affine)

    # Save the NIFTI file to the specified output path
    nib.save(nifti_img, output_nifti_path)
    print(f"Saved NIFTI file: {output_nifti_path}")


# Define the range for detecting red color
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# # Process each PNG file in the input folder


# Function to save the result image
def save_results(image, y_pred, save_image_path, mask=None):
    if mask is not None:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
    
    # Convert prediction to RGB for visualization
    y_pred_rgb = np.expand_dims(y_pred, axis=-1)
    y_pred_rgb = np.concatenate([y_pred_rgb, y_pred_rgb, y_pred_rgb], axis=-1) * 255
    
    # Ensure the image is in the correct range for display/saving (0–255)
    if image.max() <= 1.0:  # This means the image is normalized between 0 and 1
        image = (image * 255).astype(np.uint8)  # Convert to 0-255 and cast to uint8

    # Create a red overlay where the predicted mask is 1
    overlay = image.copy()  # Start with the original image
    red_color = (0, 0, 255)  # Red color for the overlay

    # Apply the red overlay to the mask region where prediction is 1
    overlay[y_pred == 1] = red_color  # Apply red where prediction is 1

    # Blend the original image and the overlay with some transparency
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)  # Alpha blending

    # Create a white line for separation
    line = np.ones((H, 10, 3)) * 255
    # Concatenate the original image and prediction overlay
    # cat_images = np.concatenate([image, line, y_pred_rgb, line, blended], axis=1)
    if mask is not None:
        cat_images = np.concatenate([image,line,mask,line,y_pred_rgb], axis=1)
    else:
        cat_images=np.concatenate([y_pred_rgb,line,blended],axis=1)

    # Add labels to the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color for text
    thickness = 1

    # Label positions
    # cv2.putText(cat_images, "Original Image", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    if mask is not None:
        cv2.putText(cat_images, "Original Image", (20, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(cat_images, "Ground Truth", ( H + 30, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(cat_images, "Predicted Mask", ( 2*H+30, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(cat_images, "Predicted Mask", (20, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(cat_images, "Overlay Image", ( H + 30, 30), font, font_scale, color, thickness, cv2.LINE_AA)

    # Save the final concatenated image
    cv2.imwrite(save_image_path, cat_images)

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (W, H))
    # mask = cv2.resize(mask, (W, H))

    return image

# Preprocess 3D image
def preprocess_image_3d(image_path):
    img = nib.load(image_path)
    image_data = img.get_fdata()

    # Normalize the image volume
    image_data_normalized = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Resize each slice and stack
    image_resized = np.zeros((image_data.shape[0], H, W))
    for i in range(image_data.shape[0]):
        slice_image = image_data[i, :, :]
        resized_image = cv2.resize(slice_image, (W, H))
        image_resized[i, :, :] = resized_image
    
    return image_resized

# Endpoint for image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create directory for saving results
    results_dir = "results"  # Update this path if needed
    create_dir(results_dir)
    output_path=""
    base, ext = os.path.splitext(file.filename)
    if ext == '.gz' and base.endswith('.nii'):
        file_extension = '.nii.gz'
    else:
        file_extension = ext
    
    if file_extension==".nii.gz":
        # Save the uploaded image temporarily
        file_path = os.path.join(results_dir, file.filename)
        file.save(file_path)

        img=nib.load(file_path)
        image_data = img.get_fdata()

        slice_output_folder = os.path.join(results_dir,f"{file.filename}_slices")
        os.makedirs(slice_output_folder,exist_ok=True)

        original_shape=image_data.shape[1:3]

        slice_images=[]

        for i in range(image_data.shape[0]):          
           slice_image = image_data[i, :, :]  # Extract 2D slice
        
           # Normalize the image slice (scale between 0 and 1)
           slice_image_normalized = (slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image))
        
           # Resize to match input shape for the model
           resized_image = cv2.resize(slice_image_normalized, (W, H))  # Resize to (256, 256) or any required size
        
           # Convert to 3-channel (RGB) format, necessary for some models expecting RGB images
           resized_image_rgb = np.stack([resized_image] * 3, axis=-1)  # [H, W, 3]
        
           # Expand dimensions to match model input shape (batch size of 1)
           x = np.expand_dims(resized_image_rgb, axis=0)  # [1, H, W, 3]
        
           """ Prediction """
           y_pred = model.predict(x, verbose=0)[0]
           y_pred = np.squeeze(y_pred, axis=-1)
           y_pred = y_pred >= 0.5
           y_pred = y_pred.astype(np.int32)

           """ Saving the prediction """
           save_image_path = os.path.join(slice_output_folder, f"slice{i+1}.png")
           save_results(resized_image_rgb, y_pred, save_image_path)


        png_files = sorted([f for f in os.listdir("results/IMG_0002.nii.gz_slices") if f.endswith('.png')])

        # Load each slice and stack them into a 3D volume
        reconstructed_slices = []

        for png_file in png_files:
           img = Image.open(f"results/IMG_0002.nii.gz_slices/{png_file}")
           slice_2d = np.array(img)  # Convert the image back to a numpy array
           resized_slice = cv2.resize(slice_2d,original_shape)
           reconstructed_slices.append(resized_slice)

        # Stack the 2D slices into a 3D array
        reconstructed_3d_image = np.stack(reconstructed_slices, axis=0)
        new_img = nib.Nifti1Image(reconstructed_3d_image, affine=np.eye(4))
        nib.save(new_img, 'results/reconstructed_image.nii.gz')
        print(reconstructed_3d_image.shape)
        recon_data = reconstructed_3d_image[:,:,:,0]
        output_path=f"http://172.20.10.4:5000/results/{file.filename}_slices/slice125.png"
        # mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))

        # # Render the 3D volume (image)
        # src_image = mlab.pipeline.scalar_field(recon_data)
        # mlab.pipeline.iso_surface(src_image, contours=[recon_data.min() + 0.1 * recon_data.ptp()], opacity=0.3, color=(0.8, 0.8, 0.8))  # Set a slightly transparent gray for the background

        # mlab.view(azimuth=180, elevation=80)
        # mlab.show()
    elif file_extension==".png":
        image_file=request.files['file']
        image_path = os.path.join(results_dir, image_file.filename)
        image_file.save(image_path)
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        x_input = image / 255.0
        x_input = np.expand_dims(x_input, axis=0)

        # Make the prediction
        y_pred = model.predict(x_input, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)  # Remove extra dimension [H, W]
        y_pred = (y_pred >= 0.5).astype(np.int32)  # Threshold to binary mask

        # Save results
        save_image_path = os.path.join(results_dir, "unreal.png")
        save_results(image, y_pred, save_image_path)
        save_image_path = os.path.join(results_dir, image_file.filename)
        save_results(image, y_pred, save_image_path)
        output_path= f"http://172.20.10.4:5000/results/{image_file.filename}"
        open_unreal(UNET_PATH)
        
    return jsonify({
        'prediction':'success',
        'output_path': output_path
    })

# Endpoint for image and mask upload with augmentation
@app.route('/upload_augmented', methods=['POST'])
def upload_augmented_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No image selected'}), 400
    
    print(request.files)
    file = request.files['file']
    # mask_file = request.files['mask']

    if file.filename == '' :
        return jsonify({'error': 'No selected image'}), 400
    # Preprocess image and mask
    # Create directory for saving results
    results_dir = "results"  # Update this path if needed
    create_dir(results_dir)

    """ Reading and preprocessing the image """
    # x = image / 255.0  # Normalize the image
    # x = (image - np.min(image)) / (np.max(image) - np.min(image))


    
    # output_path=""
    base, ext = os.path.splitext(file.filename)
    if base.endswith('.nii'):
        file_extension = '.nii'
    else:
        file_extension = ext
    
    if file_extension==".nii":
        # Save the uploaded image temporarily
        file_path = os.path.join(results_dir, file.filename)
        file.save(file_path)
        # image=preprocess_image_3d(file_path)

        # img=nib.load(file_path)
        # image_data = img.get_fdata()
        
        # output_subfolder = os.path.join(results_dir, base)
        # #slices
        # convert_nifti_to_png(file_path, results_dir,output_subfolder)
        # # original_shape=image_data.shape[1:3]

        # output_fol = os.path.join("red",file.filename)
        # os.makedirs(output_fol,exist_ok=True)
        # for image_name in os.listdir(output_subfolder):
        #     print(image_name)
        #     x = os.path.join(output_subfolder, image_name)
        #     print(x)

        #     # Process and predict as before
        #     image = cv2.imread(x, cv2.IMREAD_COLOR)
        #     original_shape = image.shape
        #     image = cv2.resize(image, (W, H))
        #     x = image / 255.0
        #     print(x.shape)
        #     # Extract patches and predict the mask
        #     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        #     patches = patchify(x, patch_shape, cf["patch_size"])

        #     print(patches.shape)
        #     patches = np.reshape(patches, cf["flat_patches_shape"])
        #     patches = patches.astype(np.float32)
        #     patches = np.expand_dims(patches, axis=0)
            
        #     y_pred = unetr_model.predict(patches, verbose=0)[0]
        #     y_pred = np.squeeze(y_pred, axis=-1)
        #     y_pred = (y_pred >= 0.5).astype(np.uint8)
        #     y_pred = cv2.resize(y_pred,(original_shape[1],original_shape[0]))
        #     image=cv2.resize(image,(original_shape[1],original_shape[0]))

        #     # Save the result
        #     save_image_path = os.path.join(f"results/{base}", image_name)
        #     save_results_3d(image, y_pred, save_image_path)
        
        # #red overlay
        # for file_name in os.listdir(f"results/{base}"):
        #     if file_name.endswith(".png"):
        #         # Construct the full file path
        #         input_path = os.path.join(f"results/{base}", file_name)
        #         output_path = os.path.join(output_fol, file_name)

        #         # Load the image
        #         image = cv2.imread(input_path)
        #         # Resize the image to 512x512 (if necessary)
        #         if image.shape[:2] != (512, 512):
        #             image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

        #         # Convert the image to HSV color space
        #         hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #         # Create masks for red color
        #         mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        #         mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        #         red_mask = cv2.bitwise_or(mask1, mask2)

        #         # Apply the mask to the original image
        #         red_overlay = cv2.bitwise_and(image, image, mask=red_mask)

        #         # Save the processed image to the output folder
        #         cv2.imwrite(output_path, red_overlay)
        # # Example usage
        # png_folder = f'red/{file.filename}'  # Folder containing PNGs
        # output_nifti_path = f'data/finalnifti/{file.filename}'  # Output NIFTI file path
        # convert_color_png_folder_to_nifti(png_folder, output_nifti_path,img.affine)

        global file1_path,file2_path
        file1_path = f"data/images/{file.filename}"
        file2_path = f"data/finalnifti/{file.filename}"
        time.sleep(10)
        open_unreal(UNETR3D_PATH)
        
        return jsonify({
            'prediction':'success',
            'augmented_result_path':f'http://172.20.10.4:5000/results/{base}/slice_125.png',
            'threed':True
        })
        
    elif file_extension==".png":
        # Save the uploaded image and mask temporarily
        image_path = os.path.join(results_dir, file.filename)
        file.save(image_path)

        # Preprocess image and mask
        image = preprocess_image(image_path)

        """ Reading and preprocessing the image """
        x = image / 255.0  # Normalize the image

        """ Extract patches from the image """
        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])  # Shape for patches (16, 16, 3)
        patches = patchify(x, patch_shape, cf["patch_size"])  # Split the image into patches
        patches = np.reshape(patches, cf["flat_patches_shape"])  # Flatten the patches to shape (256, 768)
        patches = patches.astype(np.float32)  # Convert to float32
        patches = np.expand_dims(patches, axis=0)  # Add a batch dimension (1, 256, 768)

        """ Predict the mask """
        y_pred = unetr_model.predict(patches, verbose=0)[0]  # Predict the mask
        y_pred = np.squeeze(y_pred, axis=-1)  # Remove the last dimension for consistency
        y_pred = (y_pred >= 0.5).astype(np.uint8)  # Threshold the predicted mask

        """ Save the prediction result with overlay """
        save_image_path = os.path.join(results_dir, "unreal.png")
        save_results(image, y_pred, save_image_path)
        save_image_path = os.path.join(results_dir, file.filename)
        save_results(image, y_pred, save_image_path)
        open_unreal(UNETR_PATH)

    # Return path to saved augmented result
    return jsonify({
        'prediction': 'success',
        'augmented_image_path': f"http://172.20.10.4:5000/results/{file.filename}",
        'augmented_result_path': f"http://172.20.10.4:5000/results/{file.filename}",
        'threed':False
    })
    
@app.route('/vit_classify', methods=['POST'])
def classify():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'File not found'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error' : 'No file selected'}), 400
    

    result_dir = "vit_results"
    os.makedirs(result_dir,exist_ok=True)

    image_file = request.files['file']
    image_path = os.path.join(result_dir, image_file.filename)
    image_file.save(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image ,(hp["image_size"], hp["image_size"]))
    image = image / 255.0

    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])
    
    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)
    
    try:
        # vit_path = os.path.join("", "C:/Users/srimayi/Downloads/vit_classifier_model.keras")
        vit_path = "vit_classifier.keras"
        vit = tf.keras.models.load_model(vit_path)
        p = vit.predict(patches)
        print(p,type(max(p)))
        p = list(map(float,max(p)))
        print(p)
        prediction = p.index(max(p))
        print(prediction)
        if(prediction == 0):
            output = "Benign"
        elif(prediction == 1):
            output = "Malignant"
        else:
            output = "Normal"

        return jsonify(
            {
                'prediction' : "success",
                'output' : output
            }
        )
    except Exception as e:
        return jsonify({'error' : e})


@app.route('/papaya',methods=["GET"])
def papaya_viewer():
    global file1_path,file2_path
    return render_template("index.html", file1_path=file1_path, file2_path=file2_path)

@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)

# Endpoint to serve the augmented result images
@app.route('/results_augmented/<path:filepath>', methods=['GET'])
def get_augmented_result_image(filepath):
    return send_from_directory('results_augmented', filepath)


# Endpoint to serve the result images
@app.route('/results/<path:filepath>', methods=['GET'])
def get_result_image(filepath):
    return send_from_directory('results', filepath)

# Run the app
if __name__ == "__main__":
    import os
    app.run(host='0.0.0.0', port=5000)

