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
                image = im2double(obj.Image);
                R = image(:,:,1);
                G = image(:,:,2);
                B = image(:,:,3);
    
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
        % Funcția RGB2HSV transforma imaginea RGB
        % intr-o imagine HSV.
        
        function [H, S, V] = RGB2HSV(obj)
            try

                image = im2double(obj.Image);
                R = image(:,:,1);
                G = image(:,:,2);
                B = image(:,:,3);
                
                Max = max(max(R,G),B);

                % V
                V = Max; 
                Min = min(min(R,G),B);
                delta = Max - Min;
                
                % S
                S = zeros(size(Max));
                S(Max ~= 0) = delta(Max ~= 0) ./ Max(Max ~= 0); 
                
                % H
                H = zeros(size(Max));
                mask = (Max == R) & (delta ~= 0);
                H(mask) = 60 * (0 + (G(mask) - B(mask)) ./ delta(mask));

                mask = (Max == G) & (delta ~= 0);
                H(mask) = 60 * (2 + (B(mask) - R(mask)) ./ delta(mask));

                mask = (Max == B) & (delta ~= 0);
                H(mask) = 60 * (4 + (R(mask) - G(mask)) ./ delta(mask));

                H(delta == 0) = 0; % Normalize
                H = H / 360; 

            catch Er
                disp("Error: " + Er.message)
            end
        end

        %__ RGB to LAB
        % Funcția RGB2LAB transforma imaginea RGB
        % intr-o imagine LAB (CIELAB).

        function [L, a, b] = RGB2LAB(obj)

            function new = f(t)
                new = t .^ (1/3);
                mask = t <= 0.008856;
                new(mask) = 7.787 * t(mask) + 16/116;
            end

            try
                R = double(obj.Image(:,:,1));
                G = double(obj.Image(:,:,2));
                B = double(obj.Image(:,:,3));

                Rn = R / 255;
                Gn = G / 255;
                Bn = B / 255;

                Rp = zeros(size(R));
                Gp = zeros(size(G));
                Bp = zeros(size(B));


                mask = (Rn <= 0.04045);
                Rp(mask) = Rn(mask) / 12.92;
                Rp(~mask) = ((Rn(~mask) + 0.055)/1.055).^2.4;
                
                mask = (Gn <= 0.04045);
                Gp(mask) = Gn(mask) / 12.92;
                Gp(~mask) = ((Gn(~mask) + 0.055)/1.055).^2.4;
                
                mask = (Bn <= 0.04045);
                Bp(mask) = Bn(mask) / 12.92;
                Bp(~mask) = ((Bn(~mask) + 0.055)/1.055).^2.4;


                M = [0.4124564 0.3575761 0.1804375;
                     0.2126729 0.7151522 0.0721750;
                     0.0193339 0.1191920 0.9503041];
                
                [h, w] = size(Rp);
                XYZ = zeros(h, w, 3);
                
                for i = 1:h
                    for j = 1:w
                        rgb = [Rp(i,j); Gp(i,j); Bp(i,j)];
                        XYZ(i,j,:) = M * rgb;
                    end
                end


                X = XYZ(:,:,1);
                Y = XYZ(:,:,2);
                Z = XYZ(:,:,3);
                
                Xn = 95.047;
                Yn = 100.000;
                Zn = 108.883;
                
                x = X/Xn;
                y = Y/Yn;
                z = Z/Zn;
                
                L = 116 * f(y) - 16;
                a = 500 * (f(x) - f(y));
                b = 200 * (f(y) - f(z));

            catch Er
                disp("Error: " + Er.message);
                L = []; a = []; b = [];
            end
        end

        %__ RGB to BGR
        % Funcția RGB2BGR transforma imaginea RGB
        % intr-o imagine BGR
    
        function [B, G, R] = RGB2BGR(obj)
            try
                image = im2double(obj.Image);
                R = image(:,:,1);
                G = image(:,:,2);
                B = image(:,:,3);
            catch Er
                disp("Error: " + Er.message);
                B = []; G = []; R = [];
            end
        end

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