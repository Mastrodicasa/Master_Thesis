close all
clear all
%indices_pred
%indices_pred_inv
indices_pred_s


nbr_patch=36;
%size_patch=30;
size_patch=10;
images_batch=50;
nbr_patch_batch=nbr_patch*images_batch;

batch_teste=1;
max_patch=nbr_patch_batch*batch_teste
%%

%In one batch, differentiate one image from the other
%+1 is because python's vectors start at 0 and matlab's at 1
for j=1:images_batch
    indices_predictions(j,:)=indices_predictions(j,:)+1+nbr_patch*(j-1);
end


%%
%Select patch
%all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test.npy');
%all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test_i.npy');
all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test_s.npy');
%Select A
pred_patch=zeros(images_batch,size_patch,size_patch);
for i=1:images_batch
    pred_patch(i,:,:)=all_patch_test(indices_predictions(i),:,:);
end

%%
%Show 
%show_selected

for i=1:25
    subplot(5,5,i)
    imshow(squeeze(pred_patch(i,:,:)))
end
suptitle("Predicted Most Discriminative patches, 1-25")
figure
for i=1:25
    subplot(5,5,i)
    imshow(squeeze(pred_patch(i+25,:,:)))
end
suptitle("Predicted Most Discriminative patches, 25-50")