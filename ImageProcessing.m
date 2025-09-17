classdef ImageProcessing < handle
  
    properties(Access = private)
        Image
    end

    methods(Access = public)

        function obj = ImageProcessing(inputImage)
            if(ischar(inputImage) || isstring(inputImage)) && isfile(inputImage)
                obj.Image = imread(inputImage);
            elseif isnumeric(inputImage) && (ndims(inputImage) == 3 && size(inputImage, 3) == 3)
                obj.Image = inputImage;
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
        
        %__ Translation
        % Funcția Translation mută conținutul imaginii la stânga/dreapta (pe axa X)
        % și/sau în sus/jos (pe axa Y) cu un anumit „stride" (deplasare)
        
        function Translation(obj, stride)
            try
                image = im2double(obj.Image);
                
                [h, w, c] = size(image);
                translation = zeros(h, w, c);
                
                if(stride(1) < 0)
                    start_y = abs(stride(1)):h;
                    dest_y = 1:(h + stride(1)+1);
                else
                    start_y = 1:(h - stride(1));
                    dest_y = (1 + stride(1)):h;
                end

                if(stride(2) < 0)
                    start_x = abs(stride(2)) : w;
                    dest_x = 1:(w + stride(2)+1);
                else
                    start_x = 1:(w - stride(2));
                    dest_x = (1 + stride(2)):w;
                end

                translation(dest_y, dest_x, :) = image(start_y, start_x, :);

                obj.Image = translation;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Resizing
        % Funcția Resizing redimensioneaza imaginea la noile dimensiuni
        % dorite, pastrand intregul continut al imaginii originale.

        function Resizing(obj, new_h, new_w)
            try
                
                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                new = zeros(new_h, new_w, c);
                for x = 1:new_h
                    for y = 1:new_w
                        
                        x_old = round((x - 0.5) * (h / new_h) + 0.5);
                        y_old = round((y - 0.5) * (w / new_w) + 0.5);
                
                        x_old = min(max(x_old, 1), h);
                        y_old = min(max(y_old, 1), w);
                
                        for k = 1:c
                            new(x,y,k) = image(x_old, y_old, k);
                        end
                
                    end
                end
                
                obj.Image = new;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Scaling
        % Mărește imaginea proporțional cu factorul "scale", prin interpolare.
        
        function Scaling(obj, scale, method)
            switch method
                case 'nearest'
                    image = obj.Nearest(scale);
                case 'linear'
                    image = obj.Linear(scale);
                case 'cubic'
                    image = obj.Cubic(scale);
                case 'lanczos4'
                    image = obj.Lanczos4(scale);
                otherwise
                    error("Error: Unknown method!")
            end
            obj.Image = image;
        end

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
        
        %__ Rotating
        % Funcția Rotate roteste imaginea, la un numar de grade 
        % selectat (in sensul acelor de ceasornic).
        
        function Rotate(obj, degrees)
            try
                image = im2double(obj.Image);


                [w, h, c] = size(image);

                cx = round(h/2);
                cy = round(w/2);


                degrees = degrees * pi/180;
                
                M = [cos(degrees) -sin(degrees);
                     sin(degrees) cos(degrees)];
            
                rotated_image = zeros(w, h, c);

                for x = 1:h
                    for y = 1:w
                       coords = [x - cx; y - cy];
                        
                        coords_new = M * coords;


                        x_new = round(coords_new(1) + cx);

                        y_new = round(coords_new(2) + cy);

                        
                        if x_new > 0 && x_new <= h && ...
                           y_new > 0 && y_new <= w
                            
                            rotated_image(y_new, x_new, :) = image(y, x, :);
                            
                        end
                    end
                end

                
                obj.Image = rotated_image;
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Crop
        % Funcția Crop primeste 2 puncte si "taie" un dreptunghi 
        % din imagine, apoi inlocuieste imaginea cu rezultatul
        
        function Crop(obj, first_point, second_point)
            try
                [w, h, ~] = size(obj.Image);

                if second_point(1) < first_point(1) 
                    [second_point(1), first_point(1)] = deal(first_point(1), second_point(1));
                end
                
                if second_point(2) < first_point(2) 
                    [second_point(2), first_point(2)] = deal(first_point(2), second_point(2));
                end
                
                if second_point(1) > w
                    second_point(1) = w;
                end

                if second_point(2) > h
                    second_point(2) = h;
                end

                if first_point(1) < 1 || first_point(2) < 1
                    error("Values must be greater than 0");
                end
                
                cropped = obj.Image(first_point(1):second_point(1), first_point(2): second_point(2), :);

                obj.Image = cropped;

            catch Er
                disp("Error: " + Er.message);
            end
        end

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
        % Funcția Shearing aplica o transformare afina asupra imaginii,
        % deplasand liniile orizontale sau verticale proportional cu
        % pozitia lor.
        
        function Shearing(obj, shearing_x_factor, shearing_y_factor)
            try
                image = im2double(obj.Image);
                [h,w,c] = size(image);
                M = [1 shearing_x_factor 0; 
                    shearing_y_factor 1 0;
                    0 0 1];
                
                new_h = round(w + abs(shearing_x_factor) * h);
                new_w = round(h + abs(shearing_y_factor) * w);
                new = zeros(new_h, new_w, c);
                
                for x=1:h
                    for y=1:w
                        old_coords = [x; y; 1];
                        new_coords = M * old_coords;
                
                        x_new = round(new_coords(1));
                        y_new = round(new_coords(2));
                
                        if x_new > 1 && x_new <= size(new, 1) && ...
                           y_new > 1 && y_new <= size(new, 2)
                        
                            for k=1:c
                                new(x_new, y_new, k) = image(x, y, k);
                            end
                        end
                    end
                end

                obj.Image = new;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Filters
        % Filtrele in procesarea imaginilor sunt metode care modifica
        % pixelii unei imagini pentru a evidentia sau reduce anumite
        % caracteristici, cum ar fi marginile, zgomotul sau textura.

        function Filters(obj, method)
            try
                if isstring(method) || ischar(method)
                    switch method
                        case 'Average'
                            obj.Image = obj.Average();
                        case 'Gaussian'
                            obj.Image = obj.Gaussian();
                        case 'Median'
                            obj.Image = obj.Median();
                        otherwise
                            error("Unknown method!");
                    end
                else
                    obj.Image = obj.Custom(method);
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Edge Detection
        % Reprezinta schimbarile bruste in intensitate sau culoare intr-o imagine.

        function new = Edge_Detection(obj, method)
            try
                if isstring(method) || ischar(method)
                    switch method
                        case 'Sobel'
                            new = obj.Sobel();
                        case 'Canny'
                            new = obj.Canny();
                        case 'Laplacian4'
                            new = obj.Laplacian('L4');
                        case 'Laplacian8'
                            new = obj.Laplacian('L8');
                        otherwise
                            error("Unknown method!");
                    end
                else
                    error("'method' must be a string!");
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Histogram
        % Reprezinta grafic distributia intensitatilor pixelilor din acea
        % imagine.

        function [x,y] = Histogram(obj)
            try
                [h, w, ~] = size(obj.Image);
                x = 0:255;
                y = zeros([1,256]);

                for i = 1:h
                    for j = 1:w
                        y(obj.Image(i,j,1) + 1) = 1 + y(obj.Image(i,j,1) + 1);
                        y(obj.Image(i,j,2) + 1) = 1 + y(obj.Image(i,j,2) + 1);
                        y(obj.Image(i,j,3) + 1) = 1 + y(obj.Image(i,j,3) + 1);
                    end
                end

                %bar(x,y);
                %xlabel('Pixel Intensity');
                %ylabel('Frequency');
                %title('Image Histogram');

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Histogram Equalization
        % Este o tehnica de redistributie a contrastului in imagine.

        function Histogram_Equalization(obj)
            try
                [h, w, c] = size(obj.Image);
                new = zeros(h,w,c, "uint8");

                [~, y] = obj.Histogram();

                y_R = y(1:3:end);
                y_G = y(2:3:end);
                y_B = y(3:3:end);

                CDF_R = cumsum(y_R) / sum(y_R);
                CDF_G = cumsum(y_G) / sum(y_G);
                CDF_B = cumsum(y_B) / sum(y_B);

                CDF_R_scaled = uint8(255 * CDF_R);
                CDF_G_scaled = uint8(255 * CDF_G);
                CDF_B_scaled = uint8(255 * CDF_B);

                R_index = min(floor(double(obj.Image(:,:,1)) / 3) + 1, 85);
                G_index = min(floor(double(obj.Image(:,:,2)) / 3) + 1, 85);
                B_index = min(floor(double(obj.Image(:,:,3)) / 3) + 1, 85);

                new(:,:,1) = CDF_R_scaled(R_index);
                new(:,:,2) = CDF_G_scaled(G_index);
                new(:,:,3) = CDF_B_scaled(B_index);

                obj.Image = new;
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Draw Rectangle
        % Deseneaza un patrulater pe imagine

        function Draw_Rectangle(obj, first_point, second_point, thikness, color, fill)
            try
                [h, w, c] = size(obj.Image);
                

                if second_point(1) < first_point(1) 
                    [second_point(1), first_point(1)] = deal(first_point(1), second_point(1));
                end
                
                if second_point(2) < first_point(2) 
                    [second_point(2), first_point(2)] = deal(first_point(2), second_point(2));
                end
                
                if second_point(1) > w
                    second_point(1) = w;
                end

                if second_point(2) > h
                    second_point(2) = h;
                end

                if first_point(1) < 1 || first_point(2) < 1
                    error("Values must be greater than 0");
                end

                if color(1) == 0
                    color(1) = 1;
                elseif color(2) == 0
                    color(2) = 1;
                elseif color(3) == 0
                    color(3) = 1;
                end
                
                mask_image = zeros(h, w, c, "uint8");

                for k = 1:c
                    mask_image(first_point(1):second_point(1), first_point(2):second_point(2), k) = color(k);
                    if fill ~= true
                        mask_image(first_point(1)+thikness:second_point(1)-thikness, first_point(2)+thikness:second_point(2)-thikness, k) = 0;
                    end
                    
                end
                
                mask = mask_image ~= 0;
                obj.Image(mask) = mask_image(mask);

            catch Er
                disp("Error: " + Er.message);
            end
        end


        %__ Display Image
            
        function Display_Image(obj)
            imshow(obj.Image);
        end

    end

    methods (Access = private)
        %__ Interpolation
        % Determina cum valorile pixelilor sunt selectati, 
        % cand dimensiunile imaginii sunt scazute sau crescute.

        %____ Nearest
        % Fiecarui pixel din imaginea redimensionata ii este atribuita
        % valoarea celui mai apropiat pixel din imaginea originala

        function new = Nearest(obj, scale)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);

                size_new = [scale*h, scale*w, c];
                new = zeros(size_new);

                for x=1:h
                    for y=1:w
                        for k=1:c
                            new(scale*x-(scale-1):scale*x, scale*y-(scale-1):scale*y, k) = image(x,y,k);
                        end
                    end
                end
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Linear
        % Foloseste media aritmetica a celor mai apropiati pixeli
        % pentru a calcula valoare noului pixel

        function new = Linear(obj, scale)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);

                size_new = [scale*h, scale*w, c];
                new = zeros(size_new);

                for x = 1:size_new(1) 
                    for y = 1:size_new(2)
                        
                        i = (x-1)/scale + 1;
                        j = (y-1)/scale + 1;
                
                        i1 = floor(i);
                        i2 = min(i1+1, h);
                        j1 = floor(j);
                        j2 = min(j1+1, w);
                
                        di = i - i1;
                        dj = j - j1;
                
                        for k = 1:c
                            L11 = image(i1,j1,k);
                            L21 = image(i2,j1,k);
                            L12 = image(i1,j2,k);
                            L22 = image(i2,j2,k);
                
                            new(x,y,k) = (1-di)*(1-dj)*L11 + di*(1-dj)*L21 + (1-di)*dj*L12 + di*dj*L22;
                        end
                    end
                end
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Cubic
        % Folosim cei mai apropiati 16 pixeli pentru a calcula valoarea 
        % noului pixel.

        function new = Cubic(obj, scale)
            
            %Cubic kernel
            function out = cubic_weight(x)
                a = -0.5;
                x = abs(x);
                if x <= 1
                    out = (a+2)*x^3 - (a+3)*x^2 + 1;
                elseif x < 2
                    out = a*x^3 - 5*a*x^2 + 8*a*x - 4*a;
                else
                    out = 0;
                end
            end

            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);

                size_new = [scale*h, scale*w, c];
                new = zeros(size_new);

                for x = 1:size_new(1) 
                    for y = 1:size_new(2)
                        
                        % coordonatele din imaginea originala
                        i = (x-1)/scale + 1;
                        j = (y-1)/scale + 1;
                
                        i1 = floor(i);
                        j1 = floor(j);
                
                        di = i - i1;
                        dj = j - j1;
                
                        for k = 1:c
                            val = 0;
                            total_w = 0;
                            
                            % vecinii pixelului 
                            for m = -1:2
                                for n = -1:2
                                    im = min(max(i1 + m, 1), h);
                                    jn = min(max(j1 + n, 1), w);
                
                                    wi = cubic_weight(m - di);
                                    wj = cubic_weight(n - dj);
                                    wtot = wi * wj;
                
                                    val = val + image(im, jn, k) * wtot;
                                    total_w = total_w + wtot;
                                end
                            end
                            if total_w ~= 0
                                new(x,y,k) = val / total_w;
                            end
                        end
                    end
                end
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Area
        % Impartim imaginea in blocuri de pixeli, apoi calculam
        % media aritmetica a tuturor pixelilor din bloc, pentru a
        % obtine un nou pixel
        
        function new = Area(obj, scale)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);

                size_new = [round(scale*h), round(scale*w), c];
                new = zeros(size_new);

                block_h = h / size_new(1);
                block_w = w / size_new(2);
                
                for x = 1:size_new(1) 
                    for y = 1:size_new(2)
                        start_x = floor((x-1) * block_h) + 1;
                        end_x = min(floor(x * block_h), h);
                
                        start_y = floor((y-1) * block_w) + 1;
                        end_y = min(floor(y * block_w), w);
                
                        for k = 1:c
                            block = image(start_x:end_x, start_y:end_y, k);
                            new(x,y,k) = mean(block(:));
                        end
                    end
                end
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Lanczos4
        % Folosește un kernel bazat pe funcția sinc trunchiat la ±4 pixeli,
        % adică pentru fiecare pixel nou combină valorile unui pătrat 8×8 
        % de vecini cu ponderi calculate prin sinc(x)*sinc(x/4).
        
        function new = Lanczos4(obj, scale)
            function out = mySinc(x)
                if x ~= 0
                    out = sin(pi*x) ./ (pi*x);
                else
                    out = 1;
                end
            end
            
            function out = Lanczos(x)
                if abs(x) < 4
                    out = mySinc(x) .* mySinc(x/4);
                else
                    out = 0;
                end
            end

            try

                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                size_new = [round(scale*h), round(scale*w), c];
                new = zeros(size_new);
                
                for x = 1:size_new(1) 
                    for y = 1:size_new(2)
                        
                        i = (x-0.5)/scale + 0.5;
                        j = (y-0.5)/scale + 0.5;
                
                        i1 = floor(i);
                        j1 = floor(j);
                
                        for k = 1:c
                            val = 0;
                            total_w = 0;
                            
                            for m = -3:4
                                for n = -3:4
                                    im = min(max(i1 + m, 1), h);
                                    jn = min(max(j1 + n, 1), w);
                
                                    wi = Lanczos(i - (i1 + m));
                                    wj = Lanczos(j - (j1 + n));
                                    wtot = wi * wj;
                
                                    val = val + image(im, jn, k) * wtot;
                                    total_w = total_w + wtot;
                                end
                            end
                            if total_w ~= 0
                                new(x,y,k) = val / total_w;
                            end
                        end
                    end
                end

            catch Er
                disp("Error: " + Er.message);
            end
        end
        
        %__ Filters
        % Filtrele in procesarea imaginilor sunt metode care modifica
        % pixelii unei imagini pentru a evidentia sau reduce anumite
        % caracteristici, cum ar fi marginile, zgomotul sau textura.

        %____ Average Blur
        % Inlocuieste fiecare pixel cu media aritmetica a vecinilor lui.

        function new = Average(obj)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                new = zeros(h, w, c);
                
                for x = 2:(h-1)
                    for y = 2:(w-1)
                        for k = 1:c
                            block = image((x-1):(x+1),(y-1):(y+1), k);
                            new_pixel = (sum(sum(block))) / 9;
                            new(x,y,k) = new_pixel;
                        end
                    end
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Gaussian Blur
        % Inlocuieste fiecare pixel cu o medie ponderata a vecinilor,
        % folosind o distributie Gaussiana.

        function new = Gaussian(obj)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                new = zeros(h, w, c);
                
                G = [1 2 1;
                     2 4 2;
                     1 2 1]; 
                
                G = G / sum(G(:));
                
                for x = 2:(h-1)
                    for y = 2:(w-1)
                        for k = 1:c
                            block = image((x-1):(x+1),(y-1):(y+1), k);
                            new_pixel = sum(sum(block .* G));
                            new(x,y,k) = new_pixel;
                        end
                    end
                end

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Median Blur
        % Inlocuieste fiecare pixel cu valoarea mediana a vecinilor.

        function new = Median(obj)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                new = zeros(h, w, c);
                
                for x = 2:(h-1)
                    for y = 2:(w-1)
                        for k = 1:c
                            block = image((x-1):(x+1),(y-1):(y+1), k);
                            new_pixel = median(block);
                            new(x,y,k) = new_pixel;
                        end
                    end
                end

            catch Er
                disp("Error: " + Er.message);
            end
        end
        
        %____ Custom
        % Aceasta functie aplica un filtru personalizat pe imagine, unde
        % fiecare pixel este recalculat ca suma ponderata a vecinilor sai
        % conform unei matrice de NxM definite.
        
        function new = Custom(obj, custom_filter)
            try
                image = im2double(obj.Image);
                [h, w, c] = size(image);
                
                [n, m] = size(custom_filter);
                pad_h = floor(n/2);
                pad_w = floor(m/2);

                new = zeros(h, w, c);
                
                for x = (1 + pad_h):(h - pad_h)
                    for y = (1 + pad_w):(w - pad_w)
                        for k = 1:c
                            block = image((x - pad_h):(x + pad_h),(y - pad_w):(y + pad_w), k);
                            new_pixel = sum(sum(block .* custom_filter));
                            new(x,y,k) = new_pixel;
                        end
                    end
                end

            catch Er
                disp("Error: " + Er.message);
            end
        end
        
        %__ Edge Detection
        % Reprezinta schimbarile bruste in intensitate sau culoare intr-o imagine. 

        %____ Sobel
        % Functia Sobel detecteaza marginile orizontale si verticale,
        % calculand gradientul.

        function new = Sobel(obj)
            try
                image_gray = obj.RGB2GrayScale();
                image_gray = image_gray.Image;

                [h, w, ~] = size(image_gray);
                new_h = zeros(h, w);
                new_v = zeros(h, w);

                Gx = [-1 0 1; 
                      -2 0 2; 
                      -1 0 1];
                
                Gy = [-1 -2 -1;
                       0  0  0;
                       1  2  1];

                for x = 2:(h-1)
                    for y = 2:(w-1)
                        block = image_gray((x-1):(x+1), (y-1):(y+1));
                        new_h(x,y) = sum(sum(double(block) .* Gx));
                        new_v(x,y) = sum(sum(double(block) .* Gy));
                    end
                end

                magnitude = sqrt(new_h.^2 + new_v.^2);
                magnitude = magnitude / max(magnitude(:));

                new = magnitude;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Canny
        % Detecteaza marginile combinand algoritmul Sobel si binarizare pe
        % baza de prag.
    
        function new = Canny(obj)
            try
                image = obj;

                %1. Aplicam Gaussian Blur
                image.Filters('Gaussian');

                %2. Transformam imaginea in RGB
                image = image.RGB2GrayScale();

                %3. Aplicam Sobel
                [h, w, ~] = size(image.Image);
                new_h = zeros(h, w);
                new_v = zeros(h, w);
                
                Gx = [-1 0 1; 
                      -2 0 2; 
                      -1 0 1];
                
                Gy = [-1 -2 -1;
                       0  0  0;
                       1  2  1];
                
                for x = 2:(h-1)
                    for y = 2:(w-1)
                        block = image.Image((x-1):(x+1), (y-1):(y+1));
                        new_h(x,y) = sum(sum(double(block) .* Gx));
                        new_v(x,y) = sum(sum(double(block) .* Gy));
                    end
                end

                magnitude = sqrt(new_h.^2 + new_v.^2);
                
                %4. Subtiem marginile
                tetha_rad = atan2(new_v, new_h);
                tetha_deg = tetha_rad * (180/pi);
                
                tetha_deg(tetha_deg < 0) = tetha_deg(tetha_deg < 0) + 100;
                
                nms = zeros(h, w);

                for i = 2:(h-1)
                    for j = 2:(w-1)
                        angle = tetha_deg(i, j);
                        mag = magnitude(i, j);
                
                        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
                            neighbors = [magnitude(i, j-1), magnitude(i,j+1)];
                        elseif (angle >= 22.5 && angle < 67.5)
                            neighbors = [magnitude(i-1, j+1), magnitude(i+1,j-1)];
                        elseif (angle >= 67.5 && angle < 112.5)
                            neighbors = [magnitude(i-1, j), magnitude(i+1,j)];
                        else
                            neighbors = [magnitude(i-1, j-1), magnitude(i+1,j+1)];
                        end
                
                        if mag >= max(neighbors)
                            nms(i,j) = mag;
                        else
                            nms(i,j) = 0;
                        end
                    end
                end
                
                %5. Detectam marginile
                highThreshold = 0.2 * max(nms(:));
                lowThreshold = 0.1 * max(nms(:));
                
                edge = zeros(h, w);
                
                strong = nms >= highThreshold;
                weak = (nms >= lowThreshold) & (nms < highThreshold);
                
                edge(strong) = 1;
                
                for i = 2:(h-1)
                    for j = 2:(w-1)
                        if weak(i,j)
                            if any(any(strong(i-1:i+1, j-1:j+1)))
                                edge(i,j) = 1;
                            end
                        end
                    end
                end
                
                new = edge;

            catch Er
                disp("Error: " + Er.message);
            end
        end

        %____ Laplacian
        % Functia Laplacian detecteaza marginile cautand regiunile cu
        % schimbari rapide.
        
        function new = Laplacian(obj, method)
            try
                image_gray = obj.RGB2GrayScale();
                image_gray = image_gray.Image;

                [h, w, ~] = size(image_gray);
                new = zeros(h, w);
                
                if method == "L4"
                    L = [0 1 0; 1 -4 1; 0 1 0];
                elseif method == "L8"
                    L = [1 1 1; 1 -8 1; 1 1 1];
                else
                    error("Unknown method!");
                end

                for x = 2:(h-1)
                    for y = 2:(w-1)
                        block = image_gray((x-1):(x+1), (y-1):(y+1));
                        new(x,y) = sum(sum(double(block) .* L));
                    end
                end
                
            catch Er
                disp("Error: " + Er.message);
            end
        end

    end
end