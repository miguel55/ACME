config;

% 1. Extract venules for each capture
folder='../database/venules_init_pred';
if (~exist(folder,'dir'))
    mkdir(folder);
end
for i=1:length(sequences)
    disp(i);
    aux=strsplit(sequences{i},'/');
    d1=dir(['../database/volumes_init/' aux{end-1} '__' aux{end} '__*.mat']);
    if (~isempty(d1))
        load([d1(1).folder filesep d1(1).name]);
        segmented_aux=zeros(size(data,1),size(data,2),size(data,4),3,length(d1));
        for k=1:length(d1)
            % Get segmentations from the ACME network and combine them
            d=dir(['../database/venules_pred/' d1(k).name(1:end-4) '__*_pred.mat']);
            for j=1:length(d)
                load([d(j).folder '/' d(j).name]);
                z=strsplit(d(j).name,'__');
                z=strsplit(z{end},'_');
                z=str2double(z{1});
                if (size(data,4)<size(segm,3))
                    segmented_aux(:,:,:,:,k)=segmented_aux(:,:,:,:,k)+double(segm(:,:,z+1:z+size(segmented_aux,3),:));
                elseif(size(data,4)==size(segm,3))
                    segmented_aux(:,:,:,:,k)=segmented_aux(:,:,:,:,k)+double(segm);
                else
                    segmented_aux(:,:,z+1:z+size(segm,3),:,k)=segmented_aux(:,:,z+1:z+size(segm,3),:,k)+double(segm);
                end
            end
            % Compute softmax probabilities 
            vol=softmaxX(segmented_aux(:,:,:,:,k));
            [~,vol]=max(vol,[],4);
            % Venule is venule channel and cells (segmented cells are 
            % inside the venule)
            vol=vol>1;
            venule=vol;
            save([folder,filesep,d1(k).name],'venule')
        end
    end
end

% 2. Extract cells for each capture and track them
if (~exist('results','dir'))
    mkdir('results');    
end
% Load the classification results
load('../database/results_test.mat');
% Python indexing starts from 0
results(:,1:6)=results(:,1:6)+1;

for i=1:length(sequences)
    disp(i);
    % Obtain scores y segmentation
    aux=strsplit(sequences{i},'/');
    aux2=strsplit(aux{end},'_');
    ids=[str2double(aux{end-1}(end)),str2double(aux2{1}),str2double(aux2{2}(8:end)),0,0];
    dataRe=['../data/annotation/' aux{end-1} '/' aux{end}];
    names=dir([dataRe '/*.mat']);
    orig=load([names(1).folder '/' names(1).name]);
    orig=orig.data;
    segmented_aux=zeros(size(orig,1),size(orig,2),size(orig,4),2,length(names));
    segmented=zeros(size(orig,1),size(orig,2),size(orig,4),length(names));
    scores=zeros(size(orig,1),size(orig,2),size(orig,4),length(names));
    scores_full=zeros(size(orig,1),size(orig,2),size(orig,4),length(names));
    n_planes=zeros(size(orig,1),size(orig,2),size(orig,4),length(names));
    for k=1:length(names)
        % Get segmentations from the ACME network and combine them
        d=dir(['../database/soft_pred/' aux{end-1} '__' aux{end} '__' names(k).name(1:end-4) '__*_pred.mat']);
        ids(4)=str2double(names(k).name(2:end-4));
        for j=1:length(d)
            load([d(j).folder '/' d(j).name]);
            z=strsplit(d(j).name,'__');
            z=strsplit(z{end},'_');
            z=str2double(z{1});
            ids(5)=z;
            % Detection results
            neutrophils=results(sum(results(:,9:13)==ids,2)==5,:);
            neutrophils(:,1:6)=max(neutrophils(:,1:6),1);
            neutrophils(:,3)=min(neutrophils(:,3),size(segm,1));
            neutrophils(:,4)=min(neutrophils(:,4),size(segm,2));
            neutrophils(:,6)=min(neutrophils(:,6),size(segm,3));
            binary_map=softmaxX(segm);
            [~,binary_map]=max(binary_map,[],4);
            binary_map=double(binary_map==2);
            for n=1:size(neutrophils,1)
                binary_map(neutrophils(n,1):neutrophils(n,3),neutrophils(n,2):neutrophils(n,4),neutrophils(n,5):neutrophils(n,6))=binary_map(neutrophils(n,1):neutrophils(n,3),neutrophils(n,2):neutrophils(n,4),neutrophils(n,5):neutrophils(n,6))*neutrophils(n,7);
            end
            if (size(orig,4)<size(segm,3))
                segmented_aux(:,:,:,:,k)=segmented_aux(:,:,:,:,k)+double(segm(:,:,z+1:z+size(segmented_aux,3),:));
                scores(:,:,:,k)=scores(:,:,:,k)+double(binary_map(:,:,z+1:z+size(scores,3)));
                n_planes(:,:,:,k)=n_planes(:,:,:,k)+1;
            elseif(size(orig,4)==size(segm,3))
                segmented_aux(:,:,:,:,k)=segmented_aux(:,:,:,:,k)+double(segm);
                scores(:,:,:,k)=scores(:,:,:,k)+double(binary_map);
                n_planes(:,:,:,k)=n_planes(:,:,:,k)+1;
            else
                segmented_aux(:,:,z+1:z+size(segm,3),:,k)=segmented_aux(:,:,z+1:z+size(segm,3),:,k)+double(segm);
                scores(:,:,z+1:z+size(binary_map,3),k)=scores(:,:,z+1:z+size(binary_map,3),k)+double(binary_map);
                n_planes(:,:,z+1:z+size(binary_map,3),k)=n_planes(:,:,z+1:z+size(binary_map,3),k)+1;
            end
        end
        % Normalize scores with the number of contributing planes
        scores(:,:,:,k)=scores(:,:,:,k)./n_planes(:,:,:,k);
        % Compute softmax probabilities
        vol=softmaxX(segmented_aux(:,:,:,:,k));
        [~,vol]=max(vol,[],4);
        % Get the cells
        vol=vol==2;
        % Apply the scores to each segmented cell
        S_vol=regionprops3(vol,'VoxelIdxList');
        vol2=zeros(size(vol));
        scores2=scores(:,:,:,k);
        for s=1:height(S_vol)
            vol2(S_vol.VoxelIdxList{s})=mean(scores2(S_vol.VoxelIdxList{s}));
        end
        % 
        scores_full(:,:,:,k)=vol2;
        segmented(:,:,:,k)=vol;
    end
    if (~exist(['results/',aux{end-1},'/',aux{end},'_La'],'dir'))
        mkdir(['results/',aux{end-1},'/',aux{end},'_La']);
    end
    if (~exist(['results/',aux{end-1},'/',aux{end},'_Ha/handles.mat'],'file'))
        dataLa=['results/',aux{end-1},'/',aux{end},'_La'];
        % Track the cells with the three-pass method
        [handles]= three_pass_tracking(segmented,dataLa,dataRe,scores_full,segmented_aux);
    else
        load(['results/',aux{end-1},'/',aux{end},'_Ha/handles.mat']);
    end
    if (JPG)
        data_dir='../data/annotation/';
        for j=1:length(names)
            volume=[aux{end-1},filesep,aux{end}];
            % Load data
            data=load([data_dir,volume,filesep,'T',sprintf('%05d',j),'.mat']);
            data=data.data;
            if (size(data,3)==2)
                data=cat(3,data,zeros(size(data,1),size(data,2),1,size(data,4)));
            end
            data=max(data,[],4);
            % Load segmentation
            dataLa=['results/',aux{end-1},'/',aux{end},'_La'];
            load([dataLa,filesep,'T',sprintf('%05d',j),'.mat']);
            data=cat(2,im2uint8(data),label2rgb(max(dataL,[],3),'jet','k'));
            id_folder='../database/jpgs_results/';
            if (~exist([id_folder,aux{end-1},'/',aux{end}],'dir'))
                mkdir([id_folder,aux{end-1},'/',aux{end}]);
            end
            if (~exist([id_folder,aux{end-1},'/',aux{end}],'dir'))
                mkdir([id_folder,aux{end-1},'/',aux{end}]);
            end
            imwrite(data,[id_folder,aux{end-1},'/',aux{end},filesep,'T',sprintf('%05d',j),'.jpg']);
        end
    end
end

function s=softmaxX(x)
    %Compute softmax values for each sets of scores in x
    e_x=exp(x - repmat(max(x,[],4),[1,1,1,size(x,4)]));

    s=e_x./ repmat(sum(e_x,4),[1,1,1,size(x,4)]);
end