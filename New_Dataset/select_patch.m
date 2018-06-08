%Creation of the new dataset
close all
clear all
indices_patch
%indices_patch_withoutSF_withSC_TrainingAndTesting
%indices_inversed_patch
%indices_small_patches


nbr_patch=36;
%size_patch=30;
size_patch=10;
images_batch=15;
nbr_patch_batch=nbr_patch*images_batch;

batch_teste=5;
max_patch=nbr_patch_batch*batch_teste
%%
%Indices A
all_A=[];
for i=1:batch_teste
    
    b=strcat('indices_A_',num2str(i));
    var=eval(b);
    %In one batch, differentiate one image from the other
    %+1 is because python's vector starts at 0 and matlab's at 1
    for j=1:images_batch
        var(j,:)=var(j,:)+1+nbr_patch*(j-1);
    end
    
    %Differentiate one batch from another
    var=var+nbr_patch_batch*(i-1);
    all_A = vertcat(all_A,var);
end

all_A=reshape(all_A',[],1);
all_A=reshape(all_A,1,[]);

%%
%Indices B1
all_B1=[];
for i=1:batch_teste
    
    b=strcat('indices_B1_',num2str(i));
    var=eval(b);
    
    %Differentiate one batch from another
    var=var+1+nbr_patch_batch*(i-1);
    all_B1 = horzcat(all_B1,var);
end

%%
%Indices B2
all_B2=[];
for i=1:batch_teste
    
    b=strcat('indices_B2_',num2str(i));
    var=eval(b);
    
    %Differentiate one batch from another
    var=var+1+nbr_patch_batch*(i-1);
    all_B2 = horzcat(all_B2,var);
end


%%
%Select patch
% all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test.npy');
% label_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/label_patch_test.npy');
%all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test_i.npy');
%label_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/label_patch_test_i.npy');
all_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/all_patch_test_s.npy');
label_patch_test = readNPY('C:/Users/user/Documents/Londres/These/Python_Files/label_patch_test_s.npy');


%Select A
A_patch=zeros(length(all_A),size_patch,size_patch);
for i=1: length(all_A)
    A_patch(i,:,:)=all_patch_test(all_A(i),:,:);
end
A_label=zeros(1,length(all_A));
for i=1: length(all_A)
    A_label(i)=label_patch_test(all_A(i));
end

%Select B1
B1_patch=zeros(length(all_B1),size_patch,size_patch);
for i=1: length(all_B1)
    B1_patch(i,:,:)=all_patch_test(all_B1(i),:,:);
end
B1_label=ones(1,length(all_B1))*2;

%Select B2
B2_patch=zeros(length(all_B2),size_patch,size_patch);
for i=1: length(all_B2)
    B2_patch(i,:,:)=all_patch_test(all_B2(i),:,:);
end
B2_label=ones(1,length(all_B2))*2;

%%
%Show 
show_selected

%%
%Save
%Everything is double
%save('new_dataset_s.mat', 'A_label', 'A_patch', 'B1_label', 'B1_patch', 'B2_label', 'B2_patch')