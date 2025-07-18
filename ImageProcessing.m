classdef ImageProcessing
  
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

        %__ Flip
        % Funcția Flip inversează orientarea unei imagini, fie pe orizontală, 
        % fie pe verticală, prin rearanjarea poziției pixelilor corespunzator 
        % direcției specificate.

        function new = Flip(obj, Direction)
            switch lower(Direction)
                case "horizontal"
                    new = obj.Image(1:end, end:-1:1, 1:3);
                case "vertical"
                    new = obj.Image(end:-1:1, 1:end, 1:3);
                otherwise
                    error("Direction can be 'horizontal' or 'vertical'")
            end
        end
        
        %__ Stacking
        % Funcția Stacking combină două imagini prin alăturare, 
        % fie pe verticală, fie pe orizontală, rezultând o 
        % singură imagine compusă.

        function new = Stacking(obj, img, Direction)
            try
                [first_rows, first_cols, ~] = size(obj.Image);
                [second_rows, second_cols, ~] = size(img);
                if Direction == "horizontal"
                    height = max(first_rows, second_rows);
                    width = first_cols + second_cols;
                    new = zeros(height, width, 3, 'like', obj.Image);
                    
                    new(1:first_rows, 1:first_cols, :) = obj.Image;
                    new(1:second_rows, first_cols+1:width, :) = img;
        
                elseif Direction == "vertical"
                    height = first_rows + second_rows;
                    width = max(first_cols, second_cols);
                    new = zeros(height, width, 3, 'like', img);
        
                    new(1:first_rows, 1:first_cols, :) = obj.Image;
                    new(first_rows+1:height, 1:second_cols, :) = img;
        
                else
                    error("Direction can be 'horizontal' or 'vertical'")
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Zoom
        % Funcția Zoom mărește o imagine prin replicarea fiecarui pixel într-un bloc 
        % de dimensiune proporțională cu un factor dat, rezultând o 
        % imagine de dimensiuni mai mari, dar cu acelasi conținut vizual.

        function new = Zoom(obj, factor)
            try 
                [rows, cols, ~] = size(obj.Image);
                new = zeros(rows * factor, cols * factor, 3, 'like', obj.Image);
        
                for i = 1:rows
                    for j = 1:cols
                        for x = 1:factor
                            for y = 1:factor
                                new((i-1) * factor + x, (j-1) * factor + y, 1:3) = obj.Image(i,j,1:3);
                            end
                        end
                    end
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Shape
        % Funcția Shape decupează o imagine conform unei 
        % forme geometrice specificate (pătrat sau cerc).

        function new = Shape(obj, r, shape)
            try
                [rows, cols, ~] = size(obj.Image);
                new = zeros(rows, cols, 3, 'like', obj.Image);
        
                Ox = round(rows/2);
                Oy = round(cols/2);
        
                switch lower(shape)
                    case "square"
                        new(Ox - r:r + Ox, Oy - r:r + Oy, 1:3) = obj.Image(Ox - r:r + Ox, Oy - r:r + Oy, 1:3);
                    case "circle"
                        for i = 1:rows
                            for j = 1:cols
                                if sqrt((Ox - i)^2 + (Oy - j)^2) <= r
                                    new(i,j,1:3) = obj.Image(i,j,1:3);
                                end
                            end
                        end
                    otherwise
                        error("The shape does not exist!");
                end
            catch Er
                disp("Error: " + Er.message);
            end
        end

        %__ Brightness_Contrast
        % Funcția Brightness_Contrast ajustează valorile de intensitate ale 
        % pixelilor unei imagini, fie prin adăugarea unei constante, 
        % fie prin inmultirea cu un factor.

        function new = Brightness_Contrast(obj, value_Brightness, value_Contrast)
            new = [];
            try
                if((value_Brightness >= -100) && (value_Brightness <= 100))
                    double_image = double(obj.Image);
                    percentage = (value_Brightness / 100) * 255;
                    double_image = double_image + percentage;
        
                    if ~(value_Contrast >= 0 && value_Contrast <= 5)
                        error("Contrast value must be between 0 and 5");
                    end
        
                    double_image = (double_image - 128) * value_Contrast + 128;
                    new = uint8(min(max(double_image,0),255));
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

        function new = Compression(obj, percentage)
            new = [];
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
                    
                    new = cat(3, com_R, com_G, com_B);
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

        function new = Estompare(obj, value_Blur)
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
        
                end
        
            catch Er
                disp("Error: " + Er.message);
                new = obj.Image;
            end
        end
    end
end