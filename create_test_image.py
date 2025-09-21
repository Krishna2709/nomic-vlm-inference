#!/usr/bin/env python3
"""
Create test images for the ColPali API testing.
"""

import base64
import io
from PIL import Image, ImageDraw, ImageFont

def create_test_image(width=64, height=64, color='red', text=None):
    """Create a test image with specified dimensions and color."""
    # Create image
    img = Image.new('RGB', (width, height), color=color)
    
    # Add text if provided
    if text:
        try:
            draw = ImageDraw.Draw(img)
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position (center)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
        except Exception as e:
            print(f"Warning: Could not add text to image: {e}")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_b64

def main():
    """Create various test images."""
    print("Creating test images for ColPali API...")
    
    # Test image 1: Simple red square
    img1 = create_test_image(64, 64, 'red', 'TEST')
    print(f"Test Image 1 (64x64 red): {img1[:50]}...")
    
    # Test image 2: Blue rectangle
    img2 = create_test_image(128, 64, 'blue', 'BLUE')
    print(f"Test Image 2 (128x64 blue): {img2[:50]}...")
    
    # Test image 3: Green square
    img3 = create_test_image(32, 32, 'green', 'OK')
    print(f"Test Image 3 (32x32 green): {img3[:50]}...")
    
    # Test image 4: Larger image
    img4 = create_test_image(256, 256, 'purple', 'LARGE')
    print(f"Test Image 4 (256x256 purple): {img4[:50]}...")
    
    print("\nTest images created successfully!")
    print("You can use these base64 strings to test the image embedding API.")

if __name__ == "__main__":
    main()
