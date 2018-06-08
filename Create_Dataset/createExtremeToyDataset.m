%This function creates a matrix of triangle class images. A triangle is drawn on a black
%background, afterwards a circle or a diamond is added. Here the bounding
%box uniformly changes from 10*10 to 20*20, plus independant variation [-2
%+2] for height and width.
%2000 images are created.

clear all
close all
%As a reminder, 0 is black and 255 is white

%Creation of the matrix containing all the class1
numberOfImages=25;
sizeOfImages=60;
hardClass1=zeros(numberOfImages,sizeOfImages,sizeOfImages);



for i=1: numberOfImages
    %All shapes will fit in a (radius*2)*(radius*2) square
    radius=randi([5,10]);
    intensity=randi(255)/255;
    %%
    %Draw a triangle
    %Variation in width and height
    varH=randi([-1,1]);
    varW=randi([-1,1]);
    radiusH=radius+varH;
    radiusW=radius+varW;
    
    
    %smaller portion of the image where the center can be
    %If the first shape is in the center and the size of the figure is big,
    %the second shape will never have a place to insert itself in the image
    inCenter=true;
    %j=0;
    while(inCenter)
        %j=j+1
        centerX=randi([radiusW sizeOfImages-radiusW],1);
        centerY=randi([radiusH sizeOfImages-radiusH],1);
        inCenter=false;
        if(23<centerX && centerX<37 && 23<centerY && centerY<37)
            inCenter=true;
        end
    end
    
    RGB = insertShape(zeros(sizeOfImages),'FilledPolygon',[centerX-radiusW centerY+radiusH centerX+radiusW centerY+radiusH centerX centerY-radiusH],'Color', [1 1 1]);
    BW = cast(im2bw(RGB, 0.5),'uint8');
    hardClass1(i,:,:) = BW ;
    
    
    %%
    %Get Next position
    previousX=centerX;
    previousY=centerY;
    prevRadiusH=radiusH;
    prevRadiusW=radiusW;

    %Just a random number to star the calculation
    area=10;
    %Variation in width and height
    varH=randi([-1,1]);
    varW=randi([-1,1]);
    radiusH=radius+varH;
    radiusW=radius+varW;
    %Check if stuck
    %i=0
    while(area~=0)
        %i=i+1
        centerX=randi([radiusW sizeOfImages-radiusW],1);
        centerY=randi([radiusH sizeOfImages-radiusH],1);
        
        %This calculate the area of the intersection between the squares that
        %fit the shapes
        area = rectint([previousX-prevRadiusW previousY-prevRadiusH  prevRadiusW*2 prevRadiusH*2],[centerX-radiusW centerY-radiusH radiusW*2 radiusH*2]);
    end
    %%
    %Draw the second figure
    whichShape=randi([0 1],1);
    if(whichShape==0)
        %Draw a filled circle (x1, x2, radius). If radius=10, in a square of
        % 20*20
        smallestRadius=radiusW;
        if(radiusH<radiusW)
            smallestRadius=radiusH;
        end
        RGB = insertShape(squeeze(hardClass1(i,:,:)),'FilledCircle',[centerX centerY smallestRadius],'Color', [1 1 1]);
        
    else
        %Draw a losange
        RGB = insertShape(squeeze(hardClass1(i,:,:)),'FilledPolygon',[centerX-radiusW centerY centerX centerY+radiusH centerX+radiusW centerY centerX centerY-radiusH  ],'Color', [1 1 1]);
    end
    hardClass1(i,:,:) = im2bw(RGB, 0.5)*intensity;
    %hardClass1(i,:,:)=imnoise(hardClass1(i,:,:), 'gaussian');
    %%
    %Show
    subplot(5,5,i)
    imshow(squeeze(hardClass1(i,:,:)))
    
    
   
end
%suptitle('Extreme Toy Dataset')
%save('extremeClass1_2000.mat', 'hardClass1');
