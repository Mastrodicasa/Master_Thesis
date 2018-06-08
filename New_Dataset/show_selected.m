close all
for i=1:225
    subplot(15,15,i)
    imshow(squeeze(A_patch(i,:,:)))
end
suptitle("A, 1-225")
% figure
% for i=1:225
%     subplot(15,15,i)
%     imshow(squeeze(A_patch(i+225,:,:)))
% end
% suptitle("A, 226-450")

figure
for i=1:225
    subplot(15,15,i)
    imshow(squeeze(B1_patch(i,:,:)))
end
suptitle("B1, 1-225")
figure
for i=1:225
    subplot(15,15,i)
    imshow(squeeze(B1_patch(i+225,:,:)))
end
suptitle("B1, 226-450")

figure
for i=1:25
    subplot(5,5,i)
    imshow(squeeze(B2_patch(i,:,:)))
end
suptitle("B2, 1-25")