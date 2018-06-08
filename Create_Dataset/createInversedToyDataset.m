%This function creates a matrix of circle class images. A circle is drawn on a black
%background, afterwards a triangle or a rectangle is added.
%2000 images are created.


clear all
close all
%As a reminder, 0 is black and 255 is white

%Creation of the matrix containing all the class1
numberOfImages=2000;
sizeOfImages=60;
allClass1=zeros(numberOfImages,sizeOfImages,sizeOfImages);

%All shapes will fit in a (radius*2)*(radius*2) square
radius=10;

for i=1: numberOfImages
    %%
    centerX=randi([radius sizeOfImages-radius],1);
    centerY=randi([radius sizeOfImages-radius],1);
    %Draw a filled circle (x1, x2, radius). If radius=10, in a square of 20*20
    RGB = insertShape(zeros(sizeOfImages),'FilledCircle',[centerX centerY radius],'Color', [1 1 1]);
    BW = cast(im2bw(RGB, 0.5),'uint8');
    allClass1(i,:,:) = BW;
    
    
    %%
    %Get Next position
    previousX=centerX;
    previousY=centerY;

    %Just a random number to star the calculation
    area=10;
    while(area~=0)
        centerX=randi([radius sizeOfImages-radius],1);
        centerY=randi([radius sizeOfImages-radius],1);
        
        %This calculate the area of the intersection between the squares that
        %fit the shapes
        area = rectint([previousX-radius previousY-radius  radius*2 radius*2],[centerX-radius centerY-radius radius*2 radius*2]);
    end
    
    %%
    %Draw the second figure
    whichShape=randi([0 1],1);
    if(whichShape==0)
        %Draw a triangle
        RGB = insertShape(squeeze(allClass1(i,:,:)),'FilledPolygon',[centerX-radius centerY+radius centerX+radius centerY+radius centerX centerY-radius],'Color', [1 1 1]);
        
    else
        %Draw a square
        RGB = insertShape(squeeze(allClass1(i,:,:)),'FilledRectangle',[centerX-radius centerY-radius radius*2 radius*2],'Color', [1 1 1]);
    end
    allClass1(i,:,:) = im2bw(RGB, 0.5);
    
    %%
    %Show
    %subplot(5,5,i)
    %imshow(squeeze(allClass1(i,:,:)))
   
end
save('allClass1Inversed.mat', 'allClass1');
