clear, close all;
% List of extracted sequences (saved with FIJI)
d_seq=dir('../data/sequences/group*/*/*/*_t001_z001_c001.tif');
nbits=16;               % Number of bits for sequences
for seq=1:length(d_seq)
    folder=d_seq(seq).folder;
    d=dir([folder,filesep,'*.tif']);
    names=cell(length(d),1);
    for i=1:length(d)
        names{i}=d(i).name;
    end
    d=natsortfiles(names);
    aux=imread([folder,filesep,d{1}]);
    npix=size(aux,1);
    aux=strsplit(d{end},'_');
    tmax=str2double(aux{2}(2:end));
    zmax=str2double(aux{3}(2:end));
    cmax=str2double(aux{4}(2:end-4));
    
    % Build each capture (volume variable) and save tiff images
    volume=zeros(npix,npix,cmax,zmax,tmax);
    for i=1:length(d)
        aux=strsplit(d{i},'_');
        t=str2double(aux{2}(2:end));
        z=str2double(aux{3}(2:end));
        color=aux{4}(1:end-4);
        c=str2double(color(2:end));
        volume(:,:,c,z,t)=imread([folder,filesep,d{i}]);
    end
    % Normalize between 0 and 1
    volume=double(volume)/(2^16-1);
    temp=strsplit(folder,filesep);
    mkdir(['../data/annotation/',temp{end-2},filesep,temp{end-1},'_',temp{end}]);
    mkdir(['../data/annotation/',temp{end-2},filesep,temp{end-1},'_',temp{end},'_labels']);
    % Save the captures
    for i=1:size(volume,5)
        data=volume(:,:,:,:,i);
        save(['../data/annotation/',temp{end-2},filesep,temp{end-1},'_',temp{end},'/T' sprintf('%05d',i) '.mat'],'data');
    end
end