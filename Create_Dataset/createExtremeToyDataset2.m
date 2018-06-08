%This function creates a matrix of rectangle class images. A rectangle is drawn on a black
%background, afterwards a circle or a diamond is added. Here the bounding
%box uniformly changes from 10*10 to 20*20, plus independant variation [-2
%+2] for height and width.
%2000 images are created.


clear all
close all
%As a reminder, 0 is black and 255 is white

%Creation of the matrix containing all the class1
numberOfImages=2000;
sizeOfImages=60;
hardClass2=zeros(numberOfImages,sizeOfImages,sizeOfImages);



for i=1: numberOfImages
    intensity=randi(255)/255;
    %All shapes will fit in a (radius*2)*(radius*2) square
    radius=randi([3,10]);
    %%
    %Draw a rectangle
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
    
    RGB = insertShape(zeros(sizeOfImages),'FilledRectangle',[centerX-radiusW centerY-radiusH radiusW*2 radiusH*2],'Color', [1 1 1]);
    BW = cast(im2bw(RGB, 0.5),'uint8');
    hardClass2(i,:,:) = BW;
    
    
    %%
    %Get Next position
    previousX=centerX;
    previousY=centerY;
    prevRadiusW=radiusW;
    prevRadiusH=radiusH;
    
    %Just a random number to star the calculation
    area=10;
    %Variation in width and height
    varH=randi([-1,1]);
    varW=randi([-1,1]);
    radiusH=radius+varH;
    radiusW=radius+varW;
    %j=0
    while(area~=0)
        %j=j+1
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
        RGB = insertShape(squeeze(hardClass2(i,:,:)),'FilledCircle',[centerX centerY smallestRadius],'Color', [1 1 1]);
        
    else
        %Draw a losange
        RGB = insertShape(squeeze(hardClass2(i,:,:)),'FilledPolygon',[centerX-radiusW centerY centerX centerY+radiusH centerX+radiusW centerY centerX centerY-radiusH  ],'Color', [1 1 1]);
    end
    hardClass2(i,:,:) = im2bw(RGB, 0.5)*intensity;
    %hardClass1(i,:,:)=imnoise(hardClass1(i,:,:), 'gaussian');
    %%
    %Show
    %subplot(5,5,i)
    %imshow(squeeze(hardClass2(i,:,:)))
        
end
save('extremeClass2_2000.mat', 'hardClass2')
    
   
