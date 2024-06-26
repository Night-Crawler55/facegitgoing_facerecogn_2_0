import Image_Enhance
import Face_Detection
import Face_Encoding
import Face_Recognition
import CSV_Generation


def main():
    # 1. Enhancing Group Photos

    # Input and output directories
    group_photos = r"pth"  # Input Path of Group photos for Enhancement
    enhanced_group_photos = r"pth" # Output Path of Enhnaced photos

    # Create output folder if it doesn't exist
    if not os.path.exists(enhanced_group_photos):
        os.makedirs(enhanced_group_photos)

    # Iterate over each file in the input folder
    for filename in os.listdir(group_photos):
            # Read image
            image_path = os.path.join(group_photos, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Process image
            processed_image = Image_Enhance.a(image) # Image Enhnacement Function

            # Save processed image to output folder
            output_path = os.path.join(enhanced_group_photos, filename)
            cv2.imwrite(output_path, processed_image)

    # 2. g