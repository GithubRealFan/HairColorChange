# Hair Color Chagine:

## Original Pictures:

<div>
  <img src="https://user-images.githubusercontent.com/121934188/234773811-b48f02f5-3f12-4db6-b332-c74f62bb4c58.jpg" title="Man" alt="Man" width="205" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234773956-4349fec3-89fa-4f85-b468-6f7271a9add8.JPG" title="Girl" alt="Girl" width="205" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234774078-fdedac5a-b3b2-4c38-93c9-2be316dac7b0.jpg" title="Man2" alt="Man2" width="190" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234774195-bdb2f46f-cde9-45c3-ab8e-74be02106a55.jpg" title="Girl2" alt="Girl2" width="190" height="205"/>&nbsp;
</div>

---------------------------------------------------------------------
## Changed Pictures:
  
<div>
  <img src="https://user-images.githubusercontent.com/121934188/234774666-440d3cba-80d7-48b4-9877-e61246b9d612.jpg" title="Man" alt="Man" width="205" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234774757-75dbe911-8b70-4e6f-85a9-54892963ac68.jpg" title="Girl" alt="Girl" width="205" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234774817-2baeaff8-36f5-478a-8a33-3abf991b1fcb.jpg" title="Man2" alt="Man2" width="190" height="205"/>&nbsp;
  <img src="https://user-images.githubusercontent.com/121934188/234774875-7b53b4eb-265d-4b00-8d92-3ecc0276f92c.jpg" title="Girl2" alt="Girl2" width="190" height="205"/>&nbsp;
</div>


## requirments:
    pip install torch
    pip install numpy
    pip install Pillow
    pip install opencv-python
    pip install torchvision
## run.py
    1. You can change the last line of the run.py as you want.
        For example : evaluate(input_path='files/1.JPG', output_path='files/1_black.jpg', mod='gold')
        input_path : input image path
        output_path : output image path
        mode : color of the hair (There are 3 types : black, gold, red)
    2. Run run.py
        python run.py
## How to run this file?
    This code is used for facial image parsing, which involves dividing an input image into different regions based on the different features present in the image. 
    The code starts by importing necessary libraries including PyTorch, Numpy, PIL (Python Imaging Library), and OpenCV.

    The similar function calculates the similarity of two colors by comparing their Green (G), Blue (B), and Red (R) values. 
    If the difference between G, B or R value of both colors exceeds a certain threshold, then the function returns false, indicating that the colors are not similar. 
    Otherwise, it calculates the ratio between each color component (G/B, B/R, and G/R) and appends them to an array ar. If the length of this array is less than 1, 
    it means that the function was not able to calculate any ratios and returns False. 
    Otherwise, it checks if the minimum value in the array is equal to zero, if yes, it returns False since division by zero is not defined. 
    Finally, it calculates the brightness ratio (br) between the two colors and checks if the maximum ratio in the array divided by the minimum ratio is less than 1.05
    and the brightness ratio is between 0.7 and 1.4, if yes, it returns true, indicating that the colors are similar.

    The CFAR function determines the hair region from the face based on brightness and color percentage. 
    The function takes six arguments- G, B, and R, which are the average hair color values, g, b, and r, which are the color values to be considered, 
    pro, which is the hair percentage from the face, and bri, which is the hair brightness. 
    It first creates an empty array ar and appends the ratio between each color component (G/g, B/b, and R/r) to this array, 
    only if the g, b, and r values are greater than a certain threshold. If the length of this array is equal to 0, 
    it means that the function was not able to calculate any ratios and returns True, 
    indicating that the current pixel belongs to the hair region. Otherwise, it checks the brightness value and based on different parameters, returns True or False.

    The vis_parsing_maps function takes input image, origin image, parsing annotation, stride as arguments and performs facial parsing. 
    vis_parsing_anno does face segmentation using face segmentation model. For example, if vis_parsing_anno[_x][_y] = 1 then it will be face region, 
    if vis_parsing_anno[_x][_y] = 17 then it will be hair region. But that is not exact answer. 
    The hair region will contain all of the hair region but it will contain another region including facial area around the scalp. 
    So we need to apply hair detection on the hair region. So we applied CFAR function for detection exact hair region for that.
    Finally, it changes the color of the detected hair region and saves the output image.

    The evaluate function uses the pre-trained BiSeNet model loaded from a file and passes an input image to it for parsing using the vis_parsing_maps function. 
    The output image is then saved to the specified output path. The mode argument specifies the type of colored mask applied to the output image- gold, red, or black.
