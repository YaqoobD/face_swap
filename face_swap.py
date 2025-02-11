
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import matplotlib.pyplot as plt
import os

def detect_and_swap_faces():
    # Initialize the FaceAnalysis application with CPU provider
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Use absolute path for the model
    model_path = '/home/yaqoob/.insightface/models/inswapper_128.onnx'
    
    # Initialize face swapper with absolute path
    swapper = model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
    
    # Read the images
    target_img = cv2.imread('/home/yaqoob/git/face_swap/tom.jpg') # this is place
    source_img = cv2.imread('/home/yaqoob/git/face_swap/images.jpeg')  # Replace with your source face image path

    if target_img is None or source_img is None:
        print("Error: Could not read image")
        return
    
    # Convert BGR to RGB for matplotlib
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in both images
    target_faces = app.get(target_img)
    source_faces = app.get(source_img)
    
    if len(source_faces) == 0:
        print("No face detected in source image")
        return
        
    source_face = source_faces[0]
    n_faces = len(target_faces)
    
    # Display original image with detected faces
    result_img = target_img_rgb.copy()
    face_images = []
    
    # Draw rectangles and collect face images
    for i, face in enumerate(target_faces):
        bbox = face.bbox.astype(int)
        cv2.rectangle(result_img, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     (0, 255, 0), 2)
        
        face_img = target_img_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_images.append(face_img)
        
        cv2.putText(result_img, f'Face {i+1}: {face.det_score:.2f}', 
                   (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0), 2)
    
    # Display detected faces
    plt.figure(figsize=(15, 10))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title('Detected Faces')
    plt.show()
    
    # Display individual faces
    plt.figure(figsize=(15, 3))
    for i, face_img in enumerate(face_images):
        plt.subplot(1, n_faces, i+1)
        plt.imshow(face_img)
        plt.axis('off')
        plt.title(f'Face {i+1}')
    plt.tight_layout()
    plt.show()
    
    print(f"Found {n_faces} faces!")
    
    # Perform face swapping
    swapped_img = target_img_rgb.copy()
    for target_face in target_faces:
        swapped_img = swapper.get(swapped_img, target_face, source_face, paste_back=True)
    
    # Display final result
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(source_img_rgb)
    plt.title('Source Face')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(target_img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(swapped_img)
    plt.title('Swapped Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        detect_and_swap_faces()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


#------------------------------------------------------------------------------------------------------

# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from insightface.model_zoo import model_zoo
# import os

# def process_video():
#     # Initialize the FaceAnalysis application with CPU provider
#     app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#     app.prepare(ctx_id=-1, det_size=(640, 640))
    
#     # Initialize face swapper
#     model_path = '/home/yaqoob/.insightface/models/inswapper_128.onnx'
#     swapper = model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
    
#     # Read the source face image (the face to swap onto the video)
#     source_img = cv2.imread('/home/yaqoob/face_swap/face/face_swap/shah.png')
#     if source_img is None:
#         print("Error: Could not read source image")
#         return
        
#     # Detect face in source image
#     source_faces = app.get(source_img)
#     if len(source_faces) == 0:
#         print("No face detected in source image")
#         return
#     source_face = source_faces[0]
    
#     # Open the video file
#     video_path = '/home/yaqoob/face_swap/face/face_swap/test1.mp4'  # Replace with your video path
#     cap = cv2.VideoCapture(video_path)
    
#     # Get video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Create video writer
#     output_path = 'output_swapped.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
#         print(f"Processing frame {frame_count}")
        
#         # Detect faces in the frame
#         target_faces = app.get(frame)
        
#         # If faces detected, perform swap
#         if len(target_faces) > 0:
#             # Swap all detected faces
#             result = frame.copy()
#             for target_face in target_faces:
#                 result = swapper.get(result, target_face, source_face, paste_back=True)
            
#             # Write the frame
#             out.write(result)
#         else:
#             # If no faces detected, write original frame
#             out.write(frame)
    
#     # Release everything
#     cap.release()
#     out.release()
#     print(f"Processing complete! Output saved to {output_path}")

# if __name__ == "__main__":
#     try:
#         process_video()
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import mediapipe as mp


# def segment_person(image_path, background_path):
#     # Initialize MediaPipe Pose
#     mp_selfie_segmentation = mp.solutions.selfie_segmentation
#     selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
#     # Read images
#     original_img = cv2.imread(image_path)
#     background_img = cv2.imread(background_path)
    
#     if original_img is None or background_img is None:
#         print("Error: Could not read images")
#         return

#     # Convert to RGB
#     original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#     background_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    
#     # Resize background to match original image size
#     background_rgb = cv2.resize(background_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
    
#     # Get person segmentation mask using MediaPipe
#     results = selfie_segmentation.process(original_rgb)
#     person_mask = results.segmentation_mask > 0.1
#     person_mask = person_mask.astype(np.uint8)
    
#     # Refine the mask
#     kernel = np.ones((5,5), np.uint8)
#     person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     # Apply edge-aware refinement
#     person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
    
#     # Create foreground and background masks with correct shape
#     foreground_mask = np.stack([person_mask] * 3, axis=-1)  # Make it 3-channel
#     background_mask = 1 - foreground_mask
    
#     # Apply masks with alpha blending
#     foreground = (original_rgb * foreground_mask).astype(np.uint8)
#     background_portion = (background_rgb * background_mask).astype(np.uint8)
    
#     # Blend images
#     final_image = cv2.addWeighted(
#         foreground, 1,
#         background_portion, 1,
#         0
#     )
    
#     # Add edge refinement
#     edge_kernel = np.ones((3,3), np.uint8)
#     edge_mask = np.stack([cv2.dilate(person_mask, edge_kernel, iterations=1) - 
#                          cv2.erode(person_mask, edge_kernel, iterations=1)] * 3, axis=-1)
#     final_image = cv2.addWeighted(final_image, 0.95, 
#                                  (original_rgb * edge_mask).astype(np.uint8), 0.05, 0)
    
#     # Display results
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(231)
#     plt.imshow(original_rgb)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     plt.subplot(232)
#     plt.imshow(person_mask, cmap='gray')
#     plt.title('Person Mask')
#     plt.axis('off')
    
#     plt.subplot(233)
#     plt.imshow(1 - person_mask, cmap='gray')
#     plt.title('Background Mask')
#     plt.axis('off')
    
#     plt.subplot(234)
#     plt.imshow(foreground)
#     plt.title('Extracted Person')
#     plt.axis('off')
    
#     plt.subplot(235)
#     plt.imshow(background_rgb)
#     plt.title('New Background')
#     plt.axis('off')
    
#     plt.subplot(236)
#     plt.imshow(final_image)
#     plt.title('Final Result')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Save the final image
#     final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite('final_result.jpg', final_image_bgr)
    
#     return final_image

# if __name__ == "__main__":
#     try:
#         image_path = '/home/yaqoob/face_swap/face/face_swap/anna.jpeg'
#         background_path = '/home/yaqoob/face_swap/face/face_swap/ddd.jpg'

        
#         final_image = segment_person(image_path, background_path)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

