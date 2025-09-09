classdef ImageProcessing < handle
  
    properties
        Image
    end

    methods
        function obj = ImageProcessing(inputImage)
            if(ischar(inputImage) || isstring(inputImage)) && isfile(inputImage)
                obj.Image = imread(inputImage);
            else
                error("Unsupported input type");
            end
        end
        
        %__ RGB to GrayScale
        % Funcția RGB2GrayScale transforma imaginea RGB
        % intr-o imagine alb-negru.
        
        function new = RGB2GrayScale(obj)
            try
                img = im2double(obj.Image);

                R = img(:,:,1);
                G = img(:,:,2);
                B = img(:,:,3);
    
                gamma = 1.04;
                r_const =  0.2126;
                g_const =  0.7152;
                b_const =  0.0722;

                new.Image = (r_const * (R .^ gamma)) + (g_const * (G .^ gamma)) + (b_const * (B .^ gamma));
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ RGB to HSV
        %Not Implemented

        %__ RGB to LAB
        %Not Implemented

        %__ RGB to BGR
        %Not Implemented

        %__ Flip
        % Funcția Flip inversează orientarea unei imagini, fie pe orizontală, 
        % fie pe verticală, prin rearanjarea poziției pixelilor corespunzator 
        % direcției specificate.

        function Flip(obj, Direction)
            switch lower(Direction)
                case "horizontal"
                    obj.Image = obj.Image(1:end, end:-1:1, 1:3);
                case "vertical"
                    obj.Image = obj.Image(end:-1:1, 1:end, 1:3);
                otherwise
                    error("Direction can be 'horizontal' or 'vertical'")
            end
        end
        
        %__ Rotate
        %Not Implemented
        
        %__ Resizing
        %Not Implemented

        %__ Scaling
        %Not Implemented

        %__ Interpolation
        %____ Nearest
        %____ Linear
        %____ Cubic
        %____ Area
        %____ Lanczos4
        %Not Implemented

        %__ Stacking
        % Funcția Stacking combină două imagini prin alăturare, 
        % fie pe verticală, fie pe orizontală, rezultând o 
        % singură imagine compusă.

        function Stacking(obj, img, Direction)
            try
                firstImage = obj.Image;
                secondImage = img.Image;

                [first_rows, first_cols, ~] = size(firstImage);
                [second_rows, second_cols, ~] = size(secondImage);

                if lower(Direction) == "horizontal"
                    height = max(first_rows, second_rows);
                    width = first_cols + second_cols;

                    newImage = zeros(height, width, 3, 'like', firstImage);
                    
                    newImage(1:first_rows, 1:first_cols, :) = firstImage;

                    newImage(1:second_rows, first_cols+1:width, :) = secondImage;
        
                elseif lower(Direction) == "vertical"
                    height = first_rows + second_rows;
                    width = max(first_cols, second_cols);

                    newImage = zeros(height, width, 3, 'like', firstImage);
        
                    newImage(1:first_rows, 1:first_cols, :) = firstImage;
                    newImage(first_rows+1:height, 1:second_cols, :) = secondImage;
        
                else
                    error("Direction can be 'horizontal' or 'vertical'")
                end
                
                obj.Image = newImage;

            catch Er
                disp("Error: " + Er.message);
            end
        end
        
        %__ Crop
        %Not Implemented

        %__ Shape
        % Funcția Shape decupează o imagine conform unei 
        % forme geometrice specificate (pătrat sau cerc).

        function Shape(obj, r, shape)
            try
                [rows, cols, ~] = size(obj.Image);
                newImage = zeros(rows, cols, 3, 'like', obj.Image);
        
                Ox = round(rows/2);
                Oy = round(cols/2);
        
                switch lower(shape)
                    case "square"
                        newImage(Ox - r:r + Ox, Oy - r:r + Oy, 1:3) = obj.Image(Ox - r:r + Ox, Oy - r:r + Oy, 1:3);
                    case "circle"
                        for i = 1:rows
                            for j = 1:cols
                                if sqrt((Ox - i)^2 + (Oy - j)^2) <= r
                                    newImage(i,j,1:3) = obj.Image(i,j,1:3);
                                end
                            end
                        end
                    otherwise
                        error("The shape does not exist!");
                end
                
                obj.Image = newImage;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Brightness_Contrast
        % Funcția Brightness_Contrast ajustează valorile de intensitate ale 
        % pixelilor unei imagini, fie prin adăugarea unei constante, 
        % fie prin inmultirea cu un factor.

        function Brightness_Contrast(obj, value_Brightness, value_Contrast)
            try
                if((value_Brightness >= -100) && (value_Brightness <= 100))
                    double_image = double(obj.Image);
                    percentage = (value_Brightness / 100) * 255;
                    double_image = double_image + percentage;
        
                    if ~(value_Contrast >= 0 && value_Contrast <= 5)
                        error("Contrast value must be between 0 and 5");
                    end
        
                    double_image = (double_image - 128) * value_Contrast + 128;
                    obj.Image = uint8(min(max(double_image,0),255));
                else
                    error("Brightness value must be between -100 and 100.");
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Compression
        % Funcția Compression reduce dimensiunea fișierului prin 
        % eliminatrea redundanței sau a detaliilor greu perceptibile, 
        % păstrând cât mai mult din calitatea vizuală originală.

        function Compression(obj, percentage)
            try
                if(percentage >= 0 && percentage <= 100)
                    double_img = im2double(obj.Image);
        
                    R = double_img(:,:,1);
                    G = double_img(:,:,2);
                    B = double_img(:,:,3);
                
                    [U_R, S_R, V_R] = svd(R);
                    [U_G, S_G, V_G] = svd(G);
                    [U_B, S_B, V_B] = svd(B);
                
                    com_R = U_R(:,1:percentage) * S_R(1:percentage, 1:percentage) * V_R(:,1:percentage)';
                    com_G = U_G(:,1:percentage) * S_G(1:percentage, 1:percentage) * V_G(:,1:percentage)';
                    com_B = U_B(:,1:percentage) * S_B(1:percentage, 1:percentage) * V_B(:,1:percentage)';
                    
                    obj.Image = cat(3, com_R, com_G, com_B);
                else
                    error("Percentage value must be between 0 and 100.");
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Estompare
        % Funcția Estompare aplică un filtru care reduce variațiile bruste de 
        % intensitatea dintre pixeli, netezind 
        % tranzițiile si diminuarea detaliilor fine pentru a obține un aspect 
        % mai uniform sau pentru a reduce zgomotul vizual din imagine.

        function Estompare(obj, value_Blur)
            try
                [rows, cols, channels] = size(obj.Image);
                new = zeros(rows,cols, channels, 'like', obj.Image);
        
                if ~(value_Blur >= 1 && value_Blur <= min(rows,cols))
                    error("'value_Blur' must be between 0 and " + min(rows,cols) + ".");
                else
        
                    for c = 1:channels
                        for i = 1:rows
                            top = max(i - value_Blur, 1);
                            bottom = min(rows, i + value_Blur);
                            for j = 1:cols
                                left = max(j - value_Blur, 1);
                                right = min(cols, j + value_Blur);
            
                                submatrix = obj.Image(top:bottom, left:right, c);
                                new(i,j,c) = sum(submatrix(:)) / numel(submatrix);
                            end
                        end
                    end
                    
                    obj.Image = new;

                end
        
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Shearing
        %Not Implemented
        
        %__ Translation
        %Not Implemented

        %__ Edge Detection
        %____ Sobel
        %____ Canny
        %____ Laplacian
        %Not Implemented

        %__ Histogram
        %Not Implemented

        %__ Histogram Equalization
        %Not Implemented

        %__ Draw
        %____ Line
        %____ Circle
        %____ Square
        %____ Rectangle
        %Not Implemented

    end
end