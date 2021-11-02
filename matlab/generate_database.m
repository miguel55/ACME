clear, close all;
config;
% Build the database directory
if (~exist('../database','dir'))
    mkdir('../database');
end
if (~exist('../database/volumes_init','dir'))
    mkdir('../database/volumes_init');
end
% Get the original captures
for i=1:length(sequences)
    d2=dir([sequences{i},'/','*.mat']);
    aux=strsplit(sequences{i},'/');
    for j=1:length(d2)
        name=[aux{end-1},'__',aux{end},'__',d2(j).name];
        load([d2(j).folder, '/', d2(j).name]);
        % Re-arrange channels
        if (size(data,3)==2)
            data=cat(3,data(:,:,cell_channels,:),zeros(size(data,1),size(data,2),1,size(data,4)),data(:,:,venule_channel,:));
        else
            data=data(:,:,[cell_channels venule_channel],:);
        end
        save(['../database/volumes_init/',name],'data');
    end
end

% Get the 3D segmentation network captures
if (~exist('../database/volumes','dir'))
    mkdir('../database/volumes');
end
for i=1:length(sequences)
    disp(i);
    d2=dir([sequences{i},'/','*.mat']);
    aux=strsplit(sequences{i},'/');
    for j=1:length(d2)
        name=[aux{end-1},'__',aux{end},'__',d2(j).name];
        load([d2(j).folder, '/', d2(j).name]);
        % Re-arrange channels
        if (size(data,3)==2)
            data=cat(3,data(:,:,cell_channels,:),zeros(size(data,1),size(data,2),1,size(data,4)),data(:,:,venule_channel,:));
        else
            data=data(:,:,[cell_channels venule_channel],:);
        end
        dataF=data;
        if (size(dataF,4)>DEPTH)
            for k=0:size(dataF,4)-DEPTH
                data=dataF(:,:,:,k+1:k+DEPTH);
                save(['../database/volumes/',name(1:end-4),'__',num2str(k),'.mat'],'data');
            end
        elseif (size(dataF,4)==DEPTH)
            data=dataF;
            save(['../database/volumes/',name(1:end-4),'__',num2str(0),'.mat'],'data');
        else
            for k=0:DEPTH-size(dataF,4)
                data=zeros(size(dataF,1),size(dataF,2),size(dataF,3),DEPTH);
                data(:,:,:,k+1:k+size(dataF,4))=dataF;
                save(['../database/volumes/',name(1:end-4),'__',num2str(k),'.mat'],'data');
            end
        end
    end
end

