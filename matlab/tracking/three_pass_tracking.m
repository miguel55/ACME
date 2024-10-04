function [handles]= three_pass_tracking(dataL,dataLa,dataRe,scores,scores_segm)
%[handles]= three_pass_tracking(dataL,dataLa, dataRe,scores,scores_segm)
%--------------------------------------------------------------------------
% three_pass_tracking is the main routine for the cell tracking
%    
%       INPUT
%         dataL:                4D binary segmentation of the volume
%
%         dataLa:               route to store the segmentation
%
%         dataRe:               original data route
%
%         scores:               well-segmented cell probability scores
%
%         scores_segm:          segmentation probabilities of the network
%--------------------------------------------------------------------------
% Code based on PhagoSight
%--------------------------------------------------------------------------
%     Copyright (C) 2012  Constantino Carlos Reyes-Aldasoro
%
%     This file is part of the PhagoSight package.
%
%     The PhagoSight package is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, version 3 of the License.
%
%     The PhagoSight package is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with the PhagoSight package.  If not, see <http://www.gnu.org/licenses/>.
%
%--------------------------------------------------------------------------
%
% This m-file is part of the PhagoSight package used to analyse fluorescent phagocytes
% as observed through confocal or multiphoton microscopes.  For a comprehensive 
% user manual, please visit:
%
%           http://www.phagosight.org.uk
%
% Please feel welcome to use, adapt or modify the files. If you can improve
% the performance of any other algorithm please contact us so that we can
% update the package accordingly.
%
%--------------------------------------------------------------------------
%
% The authors shall not be liable for any errors or responsibility for the 
% accuracy, completeness, or usefulness of any information, or method in the content, or for any 
% actions taken in reliance thereon.
%
%--------------------------------------------------------------------------

handles.rows=size(dataL,1); 
handles.cols=size(dataL,2);    
handles.levs=size(dataL,3);    
handles.numFrames=size(dataL,4);
handles.minBlob=110;
handles.dataLa=dataLa;
handles.ChannelDistribution=[1;handles.levs;0;0;0;0];
handles.SOLVABLE=5;
if (nargin==5)
    [dataL,handles] = saveDataAndScores(handles,dataL,scores);
else
    [dataL,handles] = saveDataAndScores(handles,dataL);
end  

%% Pass 1. Kalman filter and collision detection
if (sum(scores(:)>0) > 0)
    if ~exist('dataL','var')
        dataL = dataIn;
    end
    disp('Pass 1. In process...');
    % Locate neutrophils and track them
    handles.nodeNetwork = positionNeutrophil(dataL);
    handles = TrackingKalman3D(scores>0,handles,true);

    collisions=handles.collisions;
    handles.furtherAnalysis=handles.collisions;
    % Complete handles collisions
    for j=1:size(collisions,2)
        ind=find(collisions(:,j));
        for k=1:length(ind)
            if (collisions(ind(k),j)>20000)
                % Collision
                if (mod(collisions(ind(k),j),2)==0)
                    paired=find(collisions==collisions(ind(k),j)-1);
                    if (k==length(ind))
                        if (~isempty(find(handles.finalNetwork(ind(k):end,j)==0)))
                            handles.furtherAnalysis(ind(k):find(handles.finalNetwork(ind(k):end,j)==0,1,'first')+ind(k)-1,j)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        else
                            if (length(handles.finalNetwork(ind(k):end,j))<=handles.SOLVABLE)
                                handles.furtherAnalysis(ind(k):end,j)=handles.finalNetwork(paired);
                                handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                            else
                                handles.furtherAnalysis(ind(k),j)=0;
                                handles.furtherAnalysis(paired)=0;
                            end
                        end
                    else
                        [~,id_aux]=min([abs((ind(k):ind(k+1)-1)-ind(k));abs((ind(k):ind(k+1)-1)-ind(k+1))],[],1);
                        if (~isempty(find(handles.finalNetwork(ind(k):ind(k+1)-1,j)==0)))
                            handles.furtherAnalysis(ind(k):find(handles.finalNetwork(ind(k):ind(k+1)-1,j)==0,1,'first')+ind(k)-1,j)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        else
                            aux=handles.furtherAnalysis(ind(k):ind(k+1)-1,j);
                            aux(id_aux'==1)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(ind(k):ind(k+1)-1,j)=aux;
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        end
                    end
                    collisions(ind(k),j)=0;
                    collisions(paired)=0;
                    if (handles.furtherAnalysis(ind(k),j)>20000)
                        handles.furtherAnalysis(ind(k),j)=0; 
                    end
                    if (handles.furtherAnalysis(paired)>20000)
                        handles.furtherAnalysis(paired)=0; 
                    end
                end
            else
                % Separation
                if (mod(collisions(ind(k),j),2)==0)
                    paired=find(collisions==collisions(ind(k),j)-1);
                    if (k==1)
                        if (~isempty(find(handles.finalNetwork(1:ind(k),j)==0)))
                            handles.furtherAnalysis(find(handles.finalNetwork(1:ind(k),j)==0,1,'last')+1:ind(k),j)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        else
                            if (length(handles.finalNetwork(1:ind(k),j))<=handles.SOLVABLE)
                                handles.furtherAnalysis(1:ind(k),j)=handles.finalNetwork(paired);
                                handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                            else
                                handles.furtherAnalysis(ind(k),j)=0;
                                handles.furtherAnalysis(paired)=0;
                            end
                        end
                    else
                        [~,id_aux]=min([abs((ind(k-1)+1:ind(k))-ind(k));abs((ind(k-1)+1:ind(k))-ind(k-1))],[],1);
                        if (~isempty(find(handles.finalNetwork(ind(k-1)+1:ind(k),j)==0)))
                            handles.furtherAnalysis(find(handles.finalNetwork(ind(k-1)+1:ind(k),j)==0,1,'last')+ind(k-1)+1:ind(k),j)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        else
                            aux=handles.furtherAnalysis(ind(k-1)+1:ind(k),j);
                            aux(id_aux'==1)=handles.finalNetwork(paired);
                            handles.furtherAnalysis(ind(k-1)+1:ind(k),j)=aux;
                            handles.furtherAnalysis(paired)=handles.finalNetwork(paired);
                        end
                    end
                    collisions(ind(k),j)=0;
                    collisions(paired)=0;
                    if (handles.furtherAnalysis(ind(k),j)>10000)
                        handles.furtherAnalysis(ind(k),j)=0; 
                    end
                    if (handles.furtherAnalysis(paired)>10000)
                        handles.furtherAnalysis(paired)=0; 
                    end
                end
            end
        end
    end
    
    % Collision handling
    collisions=unique(handles.furtherAnalysis(:));
    collisions=setdiff(collisions,0);
    
    for i=1:length(collisions)
        neuts=handles.finalNetwork(find(handles.furtherAnalysis(:)==collisions(i)));
        neuts=setdiff(neuts,0);
        neuts=sort(neuts);
        if (max(handles.collisions(find(handles.furtherAnalysis(:)==collisions(i))))>20000)
            % If it is a collision
            % Reference segmentation
            segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(1)),5))>0.5;
            neuts=setdiff(neuts,neuts(1));
            non_revertible=false;
            already_solved=false;
            j=1;
            while(j<=length(neuts) && non_revertible==false)
                time_id=handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5);
                segm_aux=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5));
                S_init=regionprops3(segm_ref,'Centroid','VoxelIdxList','BoundingBox');
                pos_cell=[S_init.Centroid(:,1),S_init.Centroid(:,2),S_init.Centroid(:,3)];
                ds_neu=sqrt(sum((handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),[2 1 3])-pos_cell).^2,2));
                [~,p]=min(ds_neu);
                [~,s]=sort(ds_neu);
                segm_ref=uint8(zeros(size(segm_ref)));
                segm_ref(S_init.VoxelIdxList{s(1)})=1; 
                % Get the 2 nearest regions
                if (length(s)>1)
                    segm_ref(S_init.VoxelIdxList{s(2)})=2;
                    bbox=[min(S_init.BoundingBox(s(1),1),S_init.BoundingBox(s(2),1)) min(S_init.BoundingBox(s(1),2),S_init.BoundingBox(s(2),2)) ...
                    min(S_init.BoundingBox(s(1),3),S_init.BoundingBox(s(2),3)) ...
                    max(S_init.BoundingBox(s(1),1)+S_init.BoundingBox(s(1),4),S_init.BoundingBox(s(2),1)+S_init.BoundingBox(s(2),4))+1 ...
                    max(S_init.BoundingBox(s(1),2)+S_init.BoundingBox(s(1),5),S_init.BoundingBox(s(2),2)+S_init.BoundingBox(s(2),5))+1 ...
                    max(S_init.BoundingBox(s(1),3)+S_init.BoundingBox(s(1),6),S_init.BoundingBox(s(2),3)+S_init.BoundingBox(s(2),6))+1];
                    bbox=min(max(bbox,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_ref,1) size(segm_ref,2) size(segm_ref,3)]);
                else
                    bbox=[S_init.BoundingBox(s(1),1:3),S_init.BoundingBox(s(1),1:3)+S_init.BoundingBox(s(1),4:6)+1];
                    bbox=min(max(bbox,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_ref,1) size(segm_ref,2) size(segm_ref,3)]);
                end
                neuts_aux=handles.nodeNetwork(handles.nodeNetwork(:,5)==handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5),:);
                id_aux=intersect(neuts_aux(:,6),neuts);
                centers_aux2 = handles.nodeNetwork(handles.nodeNetwork(:,6)==id_aux,1:3);
                S_aux=regionprops3(segm_aux>0.5,'Centroid','VoxelIdxList','BoundingBox');
                dS=sqrt(sum((S_aux.Centroid(:,[2,1,3])-centers_aux2).^2,2));
                [m,p_aux]=min(dS);
                segm_aux=false(size(segm_ref));
                segm_aux(S_aux.VoxelIdxList{p_aux})=true;
                centers_aux=S_aux.Centroid(p_aux,:);
                % Load the original data
                load([dataRe,filesep,'T',sprintf('%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)),'.mat']);
                data=data(:,:,end-1,:);
                data=squeeze(max(data,[],3));
                bbox2=[S_aux.BoundingBox(p_aux,1:3),S_aux.BoundingBox(p_aux,1:3)+S_aux.BoundingBox(p_aux,4:6)+1];
                bbox2=min(max(bbox2,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_aux,1) size(segm_aux,2) size(segm_aux,3)]);
                data_rec=data(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                segm_aux=segm_aux(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                segm_ref=segm_ref(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                % Cut the RoI in every matrix (data, reference segmentation 
                % and current segmentation)
                size_obj=max(size(segm_ref),size(segm_aux));
                size_obj=size_obj+(1-rem(size_obj,2));
                segm_aux_pad=zeros(size_obj);
                segm_ref_pad=uint8(zeros(size_obj));
                data_rec_pad=zeros(size_obj);
                segm_aux_pad(round((size(segm_aux_pad,1)-size(segm_aux,1))/2)+1:round((size(segm_aux_pad,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                    round((size(segm_aux_pad,2)-size(segm_aux,2))/2)+1:round((size(segm_aux_pad,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                    round((size(segm_aux_pad,3)-size(segm_aux,3))/2)+1:round((size(segm_aux_pad,3)-size(segm_aux,3))/2)+size(segm_aux,3))=segm_aux;
                segm_ref_pad(round((size(segm_ref_pad,1)-size(segm_ref,1))/2)+1:round((size(segm_ref_pad,1)-size(segm_ref,1))/2)+size(segm_ref,1), ...
                    round((size(segm_ref_pad,2)-size(segm_ref,2))/2)+1:round((size(segm_ref_pad,2)-size(segm_ref,2))/2)+size(segm_ref,2), ...
                    round((size(segm_ref_pad,3)-size(segm_ref,3))/2)+1:round((size(segm_ref_pad,3)-size(segm_ref,3))/2)+size(segm_ref,3))=segm_ref;
                data_rec_pad(round((size(data_rec_pad,1)-size(data_rec,1))/2)+1:round((size(data_rec_pad,1)-size(data_rec,1))/2)+size(data_rec,1), ...
                    round((size(data_rec_pad,2)-size(data_rec,2))/2)+1:round((size(data_rec_pad,2)-size(data_rec,2))/2)+size(data_rec,2), ...
                    round((size(data_rec_pad,3)-size(data_rec,3))/2)+1:round((size(data_rec_pad,3)-size(data_rec,3))/2)+size(data_rec,3))=data_rec;

                % Watershed method
                numS_ref=height(regionprops3(segm_ref_pad));
                img_dist=bwdist(double(~segm_aux_pad));
                D=-img_dist;
                L = watershed(D);
                L(~segm_aux_pad) = 0;
                if (length(unique(L(:)))<=max(L(:)))
                    id_absent=setdiff(unique(L(:)),1:max(L(:)));
                    for k=1:length(id_absent)
                        L(L==max(L(:)))=k;
                    end
                end
                numS_aux=max(L(:));
                if (numS_aux<numS_ref)
                    % Sub-segmentation
                    if (already_solved)
                        S=regionprops(segm_ref,'Centroid');
                        centroids=round(cat(1,S.Centroid)+[round((size_obj(1)-size(segm_aux,1))/2) round((size_obj(2)-size(segm_aux,2))/2) round((size_obj(3)-size(segm_aux,3))/2)]);
                        [X,Y,Z]=meshgrid(1:size(segm_ref_pad,2),1:size(segm_ref_pad,1),1:size(segm_ref_pad,3));
                        D=zeros([size(segm_ref_pad) size(centroids,1)]);
                        for k=1:size(centroids,1)
                            D(:,:,:,k)=sqrt((X-centroids(k,1)).^2+(Y-centroids(k,2)).^2+(Z-centroids(k,3)).^2);
                        end
                        [~,L]=min(D,[],4);
                        L(~segm_aux_pad)=0;
                        if (length(unique(L(:)))<=max(L(:)))
                            id_absent=setdiff(unique(L(:)),1:max(L(:)));
                            for k=1:length(id_absent)
                                L(L==max(L(:)))=k;
                            end
                        end
                        % Delete the padding
                        eroded_img=L(round((size(L,1)-size(segm_aux,1))/2)+1:round((size(L,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                            round((size(L,2)-size(segm_aux,2))/2)+1:round((size(L,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                            round((size(L,3)-size(segm_aux,3))/2)+1:round((size(L,3)-size(segm_aux,3))/2)+size(segm_aux,3));
                        % Save results
                        dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                        dataL=load(dataOutName1);
                        statsData=dataL.statsData;
                        numNeutrop=dataL.numNeutrop;
                        dataL=dataL.dataL;
                        data_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                        data_aux=data_aux(segm_aux>0);
                        id_dataL=unique(data_aux(data_aux(:)>0));
                        % Remap S_eroded
                        S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                        data_aux=zeros(size(eroded_img));
                        for m=1:height(S_eroded)
                            if (m<=length(id_dataL))
                                if (iscell(S_eroded.VoxelIdxList))
                                    data_aux(S_eroded.VoxelIdxList{m})=id_dataL(m);
                                else
                                    data_aux(S_eroded.VoxelIdxList(m))=id_dataL(m);
                                end
                            else
                                numNeutrop=numNeutrop+1;
                                if (iscell(S_eroded.VoxelIdxList))
                                    data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                                else
                                    data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                                end
                            end
                        end
                        orig_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                        for k=1:length(id_dataL)
                            orig_aux(orig_aux(:)==id_dataL(k))=0;
                        end
                        orig_aux(segm_aux>0)=data_aux(segm_aux>0);
                        dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=orig_aux;
                        if (length(id_dataL)>height(S_eroded))
                            for k=1:length(id_dataL)-height(S_eroded)
                                dataL(dataL==numNeutrop)=id_dataL(length(id_dataL)-k+1);
                                numNeutrop=numNeutrop-1;
                            end
                        end
                        while (length(unique(dataL(:)))<(max(dataL(:))+1))
                            id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                            for k=1:length(id_absent)
                                dataL(dataL==max(dataL(:)))=k;
                                numNeutrop=numNeutrop-1;
                            end
                        end
                        if (height(S_eroded)>1)
                            segm_ref=dataL;
                        else
                            % Recharge original
                            segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(end)),5))>0.5;
                        end
                        save(dataOutName1,'dataL','numNeutrop','statsData');
                        dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                        dataSc=load(dataOutName2);
                        dataSc=dataSc.dataSc;
                        dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=...
                                        dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6))).*double(eroded_img>0);
                        save(dataOutName2,'dataSc');  
                    else
                        non_revertible=true;
                    end
                else
                    % Over-segmentation
                    if (numS_aux>numS_ref)
                        L2=zeros([size(segm_ref_pad) numS_ref]);
                        for n=1:numS_ref
                            L2(:,:,:,n)=segm_ref_pad==n;
                        end
                        % Iterative region merging
                        k=1;
                        while (k<numS_aux)
                            region=L==k;
                            overlaps=zeros(1,numS_ref);
                            for n=1:numS_ref
                                auxL=L2(:,:,:,n);
                                overlaps(n)=sum(region(:).*auxL(:))/(sum(region(:))+sum(auxL(:))-sum(region(:).*auxL(:)));
                            end
                            [overlaps,p_ref]=max(overlaps);
                            auxL=L2(:,:,:,p_ref);
                            overlap_increments=zeros(numS_aux,1);
                            for n=1:numS_aux
                                if (n==k)
                                    overlap_increments(n,:)=0;
                                else
                                    new_region=region | L==n;
                                    overlap_increments(n)=sum(new_region(:).*auxL(:))/(sum(new_region(:))+sum(auxL(:))-sum(new_region(:).*auxL(:)))-overlaps;
                                end
                            end
                            % Analyze overlaps, if there is improvement, merge
                            [max_ov,p]=max(overlap_increments);
                            if (max_ov>0)
                                L(L==p)=k;
                                % RemapL
                                L(L==max(setdiff(1:numS_aux,k)))=p;
                                numS_aux=numS_aux-1;
                            else
                                k=k+1;
                            end
                        end
                        % Finally, group per GT
                        overlaps=zeros(numS_aux,numS_ref);
                        for k=1:numS_aux
                            region=L==k;
                            for n=1:numS_ref
                                auxL=L2(:,:,:,n);
                                overlaps(k,n)=sum(region(:).*auxL(:))/(sum(region(:))+sum(auxL(:))-sum(region(:).*auxL(:)));
                            end
                        end
                        [overlaps2,p]=max(overlaps,[],2);
                        overlaps=zeros(size(overlaps));
                        for k=1:numS_aux
                            overlaps(k,p(k))=overlaps2(k);
                        end
                        newL=L;
                        offset=0;
                        for k=1:numS_ref
                            ids=find(overlaps(:,k)>0);
                            if (isempty(ids))
                                offset=offset+1;
                            else
                                for n=1:length(ids)
                                    newL(L==ids(n))=k-offset;
                                end
                            end
                        end
                        % Merge elements
                        for k=1:numS_ref
                            aux=newL==k;
                            aux=imclose(imclose(aux,strel('line',3,0)),strel('line',3,90));
                            newL(aux)=k;
                        end
                        L=newL;
                    end
                    % Delete the padding
                    eroded_img=L(round((size(L,1)-size(segm_aux,1))/2)+1:round((size(L,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                        round((size(L,2)-size(segm_aux,2))/2)+1:round((size(L,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                        round((size(L,3)-size(segm_aux,3))/2)+1:round((size(L,3)-size(segm_aux,3))/2)+size(segm_aux,3));
                    % Save results
                    dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                    dataL=load(dataOutName1);
                    statsData=dataL.statsData;
                    numNeutrop=dataL.numNeutrop;
                    dataL=dataL.dataL;
                    data_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                    data_aux=data_aux(segm_aux>0);
                    id_dataL=unique(data_aux(data_aux(:)>0));
                    % Remap S_eroded
                    S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                    data_aux=zeros(size(eroded_img));
                    for m=1:height(S_eroded)
                        if (m<=length(id_dataL)) %id
                            if (iscell(S_eroded.VoxelIdxList))
                                data_aux(S_eroded.VoxelIdxList{m})=id_dataL(m);
                            else
                                data_aux(S_eroded.VoxelIdxList(m))=id_dataL(m);
                            end
                        else
                            numNeutrop=numNeutrop+1;
                            if (iscell(S_eroded.VoxelIdxList))
                                data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                            else
                                data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                            end
                        end
                    end
                    orig_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                    for k=1:length(id_dataL)
                        orig_aux(orig_aux(:)==id_dataL(k))=0;
                    end
                    orig_aux(segm_aux>0)=data_aux(segm_aux>0);
                    dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=orig_aux;
                    if (length(id_dataL)>height(S_eroded))
                        for k=1:length(id_dataL)-height(S_eroded)
                            dataL(dataL==numNeutrop)=id_dataL(length(id_dataL)-k+1);
                            numNeutrop=numNeutrop-1;
                        end
                    end
                    while (length(unique(dataL(:)))<(max(dataL(:))+1))
                        id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                        for k=1:length(id_absent)
                            dataL(dataL==max(dataL(:)))=k;
                            numNeutrop=numNeutrop-1;
                        end
                    end
                    if (height(S_eroded)>1)
                        segm_ref=dataL;
                    else
                        % Recharge original
                        segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(end)),5))>0.5;
                    end
                    save(dataOutName1,'dataL','numNeutrop','statsData');
                    dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                    dataSc=load(dataOutName2);
                    dataSc=dataSc.dataSc;
                    dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=...
                                    dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6))).*double(eroded_img>0);
                    save(dataOutName2,'dataSc');  
                    already_solved=true;
                end
                j=j+1;
            end
        else
            % Separation
            % Reference segmentation
            segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(end)),5))>0.5;
            neuts=setdiff(neuts,neuts(end));
            non_revertible=false;
            already_solved=false;
            j=length(neuts);
            while (j>=1 && non_revertible==false)
                time_id=handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5);
                segm_aux=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5));
                S_init=regionprops3(segm_ref,'Centroid','VoxelIdxList','BoundingBox');
                pos_cell=[S_init.Centroid(:,1),S_init.Centroid(:,2),S_init.Centroid(:,3)];
                ds_neu=sqrt(sum((handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),[2 1 3])-pos_cell).^2,2));
                [~,p]=min(ds_neu);
                [~,s]=sort(ds_neu);
                segm_ref=uint8(zeros(size(segm_ref)));
                segm_ref(S_init.VoxelIdxList{s(1)})=1; 
                % Get the 2 nearest regions
                if (length(s)>1)
                    segm_ref(S_init.VoxelIdxList{s(2)})=2;
                    bbox=[min(S_init.BoundingBox(s(1),1),S_init.BoundingBox(s(2),1)) min(S_init.BoundingBox(s(1),2),S_init.BoundingBox(s(2),2)) ...
                    min(S_init.BoundingBox(s(1),3),S_init.BoundingBox(s(2),3)) ...
                    max(S_init.BoundingBox(s(1),1)+S_init.BoundingBox(s(1),4),S_init.BoundingBox(s(2),1)+S_init.BoundingBox(s(2),4))+1 ...
                    max(S_init.BoundingBox(s(1),2)+S_init.BoundingBox(s(1),5),S_init.BoundingBox(s(2),2)+S_init.BoundingBox(s(2),5))+1 ...
                    max(S_init.BoundingBox(s(1),3)+S_init.BoundingBox(s(1),6),S_init.BoundingBox(s(2),3)+S_init.BoundingBox(s(2),6))+1];
                    bbox=min(max(bbox,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_ref,1) size(segm_ref,2) size(segm_ref,3)]);
                else
                    bbox=[S_init.BoundingBox(s(1),1:3),S_init.BoundingBox(s(1),1:3)+S_init.BoundingBox(s(1),4:6)+1];
                    bbox=min(max(bbox,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_ref,1) size(segm_ref,2) size(segm_ref,3)]);
                end 
                neuts_aux=handles.nodeNetwork(handles.nodeNetwork(:,5)==handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5),:);
                id_aux=intersect(neuts_aux(:,6),neuts);
                centers_aux2 = handles.nodeNetwork(handles.nodeNetwork(:,6)==id_aux,1:3);
                S_aux=regionprops3(segm_aux>0.5,'Centroid','VoxelIdxList','BoundingBox');
                dS=sqrt(sum((S_aux.Centroid(:,[2,1,3])-centers_aux2).^2,2));
                [m,p_aux]=min(dS);
                segm_aux=false(size(segm_ref));
                segm_aux(S_aux.VoxelIdxList{p_aux})=true;
                centers_aux=S_aux.Centroid(p_aux,:);
                % Load the original data
                load([dataRe,filesep,'T',sprintf('%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)),'.mat']);
                data=data(:,:,end-1,:);
                data=squeeze(max(data,[],3));
                bbox2=[S_aux.BoundingBox(p_aux,1:3),S_aux.BoundingBox(p_aux,1:3)+S_aux.BoundingBox(p_aux,4:6)+1];
                bbox2=min(max(bbox2,[1 1 1 0 0 0]),[Inf Inf Inf size(segm_aux,1) size(segm_aux,2) size(segm_aux,3)]);
                data_rec=data(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                segm_aux=segm_aux(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                segm_ref=segm_ref(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                % Cut the RoI in every matrix (data, reference segmentation 
                % and current segmentation)
                size_obj=max(size(segm_ref),size(segm_aux));
                size_obj=size_obj+(1-rem(size_obj,2));
                segm_aux_pad=zeros(size_obj);
                segm_ref_pad=uint8(zeros(size_obj));
                data_rec_pad=zeros(size_obj);
                segm_aux_pad(round((size(segm_aux_pad,1)-size(segm_aux,1))/2)+1:round((size(segm_aux_pad,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                    round((size(segm_aux_pad,2)-size(segm_aux,2))/2)+1:round((size(segm_aux_pad,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                    round((size(segm_aux_pad,3)-size(segm_aux,3))/2)+1:round((size(segm_aux_pad,3)-size(segm_aux,3))/2)+size(segm_aux,3))=segm_aux;
                segm_ref_pad(round((size(segm_ref_pad,1)-size(segm_ref,1))/2)+1:round((size(segm_ref_pad,1)-size(segm_ref,1))/2)+size(segm_ref,1), ...
                    round((size(segm_ref_pad,2)-size(segm_ref,2))/2)+1:round((size(segm_ref_pad,2)-size(segm_ref,2))/2)+size(segm_ref,2), ...
                    round((size(segm_ref_pad,3)-size(segm_ref,3))/2)+1:round((size(segm_ref_pad,3)-size(segm_ref,3))/2)+size(segm_ref,3))=segm_ref;
                data_rec_pad(round((size(data_rec_pad,1)-size(data_rec,1))/2)+1:round((size(data_rec_pad,1)-size(data_rec,1))/2)+size(data_rec,1), ...
                    round((size(data_rec_pad,2)-size(data_rec,2))/2)+1:round((size(data_rec_pad,2)-size(data_rec,2))/2)+size(data_rec,2), ...
                    round((size(data_rec_pad,3)-size(data_rec,3))/2)+1:round((size(data_rec_pad,3)-size(data_rec,3))/2)+size(data_rec,3))=data_rec;
                
                % Watershed method
                numS_ref=height(regionprops3(segm_ref_pad));
                img_dist=bwdist(double(~segm_aux_pad));
                D=-img_dist;
                L = watershed(D);
                L(~segm_aux_pad) = 0;
                if (length(unique(L(:)))<=max(L(:)))
                    id_absent=setdiff(unique(L(:)),1:max(L(:)));
                    for k=1:length(id_absent)
                        L(L==max(L(:)))=k;
                    end
                end
                numS_aux=max(L(:));
                if (numS_aux<numS_ref)
                    if (already_solved)
                        S=regionprops(segm_ref,'Centroid');
                        centroids=round(cat(1,S.Centroid)+[round((size_obj(1)-size(segm_aux,1))/2) round((size_obj(2)-size(segm_aux,2))/2) round((size_obj(3)-size(segm_aux,3))/2)]);
                        [X,Y,Z]=meshgrid(1:size(segm_ref_pad,2),1:size(segm_ref_pad,1),1:size(segm_ref_pad,3));
                        D=zeros([size(segm_ref_pad) size(centroids,1)]);
                        for k=1:size(centroids,1)
                            D(:,:,:,k)=sqrt((X-centroids(k,1)).^2+(Y-centroids(k,2)).^2+(Z-centroids(k,3)).^2);
                        end
                        [~,L]=min(D,[],4);
                        L(~segm_aux_pad)=0;
                        if (length(unique(L(:)))<=max(L(:)))
                            id_absent=setdiff(unique(L(:)),1:max(L(:)));
                            for k=1:length(id_absent)
                                L(L==max(L(:)))=k;
                            end
                        end
                        % Delete the padding
                        eroded_img=L(round((size(L,1)-size(segm_aux,1))/2)+1:round((size(L,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                            round((size(L,2)-size(segm_aux,2))/2)+1:round((size(L,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                            round((size(L,3)-size(segm_aux,3))/2)+1:round((size(L,3)-size(segm_aux,3))/2)+size(segm_aux,3));
                        % Save results
                        dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                        dataL=load(dataOutName1);
                        statsData=dataL.statsData;
                        numNeutrop=dataL.numNeutrop;
                        dataL=dataL.dataL;
                        data_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                        data_aux=data_aux(segm_aux>0);
                        id_dataL=unique(data_aux(data_aux(:)>0));
                        % Remap S_eroded
                        S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                        data_aux=zeros(size(eroded_img));
                        for m=1:height(S_eroded)
                            if (m<=length(id_dataL))
                                if (iscell(S_eroded.VoxelIdxList))
                                    data_aux(S_eroded.VoxelIdxList{m})=id_dataL(m);
                                else
                                    data_aux(S_eroded.VoxelIdxList(m))=id_dataL(m);
                                end
                            else
                                numNeutrop=numNeutrop+1;
                                if (iscell(S_eroded.VoxelIdxList))
                                    data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                                else
                                    data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                                end
                            end
                        end
                        orig_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                        for k=1:length(id_dataL)
                            orig_aux(orig_aux(:)==id_dataL(k))=0;
                        end
                        orig_aux(segm_aux>0)=data_aux(segm_aux>0);
                        dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=orig_aux;
                        if (length(id_dataL)>height(S_eroded))
                            for k=1:length(id_dataL)-height(S_eroded)
                                dataL(dataL==numNeutrop)=id_dataL(length(id_dataL)-k+1);
                                numNeutrop=numNeutrop-1;
                            end
                        end
                        while (length(unique(dataL(:)))<(max(dataL(:))+1))
                            id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                            for k=1:length(id_absent)
                                dataL(dataL==max(dataL(:)))=k;
                                numNeutrop=numNeutrop-1;
                            end
                        end
                        if (height(S_eroded)>1)
                            segm_ref=dataL;
                        else
                            % Recharge original
                            segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(end)),5))>0.5;
                        end
                        save(dataOutName1,'dataL','numNeutrop','statsData');
                        dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                        dataSc=load(dataOutName2);
                        dataSc=dataSc.dataSc;
                        dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=...
                                        dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6))).*double(eroded_img>0);
                        save(dataOutName2,'dataSc');  
                    else
                        non_revertible=true;
                    end
                else
                    if (numS_aux>numS_ref)
                        L2=zeros([size(segm_ref_pad) numS_ref]);
                        for n=1:numS_ref
                            L2(:,:,:,n)=segm_ref_pad==n;
                        end
                        % Iterative region merging
                        k=1;
                        while (k<numS_aux)
                            region=L==k;
                            overlaps=zeros(1,numS_ref);
                            for n=1:numS_ref
                                auxL=L2(:,:,:,n);
                                overlaps(n)=sum(region(:).*auxL(:))/(sum(region(:))+sum(auxL(:))-sum(region(:).*auxL(:)));
                            end
                            [overlaps,p_ref]=max(overlaps);
                            auxL=L2(:,:,:,p_ref);
                            overlap_increments=zeros(numS_aux,1);
                            for n=1:numS_aux
                                if (n==k)
                                    overlap_increments(n,:)=0;
                                else
                                    new_region=region | L==n;
                                    overlap_increments(n)=sum(new_region(:).*auxL(:))/(sum(new_region(:))+sum(auxL(:))-sum(new_region(:).*auxL(:)))-overlaps;
                                end
                            end
                            % Analyze overlaps, if there is improvement, merge
                            [max_ov,p]=max(overlap_increments);
                            if (max_ov>0)
                                L(L==p)=k;
                                % RemapL
                                L(L==max(setdiff(1:numS_aux,k)))=p;
                                numS_aux=numS_aux-1;
                            else
                                k=k+1;
                            end
                        end
                        % Finally, group per GT
                        overlaps=zeros(numS_aux,numS_ref);
                        for k=1:numS_aux
                            region=L==k;
                            for n=1:numS_ref
                                auxL=L2(:,:,:,n);
                                overlaps(k,n)=sum(region(:).*auxL(:))/(sum(region(:))+sum(auxL(:))-sum(region(:).*auxL(:)));
                            end
                        end
                        [overlaps2,p]=max(overlaps,[],2);
                        overlaps=zeros(size(overlaps));
                        for k=1:numS_aux
                            overlaps(k,p(k))=overlaps2(k);
                        end
                        newL=L;
                        offset=0;
                        for k=1:numS_ref
                            ids=find(overlaps(:,k)>0);
                            if (isempty(ids))
                                offset=offset+1;
                            else
                                for n=1:length(ids)
                                    newL(L==ids(n))=k-offset;
                                end
                            end
                        end
                        % Merge elements
                        for k=1:numS_ref
                            aux=newL==k;
                            aux=imclose(imclose(aux,strel('line',3,0)),strel('line',3,90));
                            newL(aux)=k;
                        end
                        L=newL;
                    end
                    % Delete the padding
                    eroded_img=L(round((size(L,1)-size(segm_aux,1))/2)+1:round((size(L,1)-size(segm_aux,1))/2)+size(segm_aux,1), ...
                        round((size(L,2)-size(segm_aux,2))/2)+1:round((size(L,2)-size(segm_aux,2))/2)+size(segm_aux,2), ...
                        round((size(L,3)-size(segm_aux,3))/2)+1:round((size(L,3)-size(segm_aux,3))/2)+size(segm_aux,3));
                    % Save results
                    dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                    dataL=load(dataOutName1);
                    statsData=dataL.statsData;
                    numNeutrop=dataL.numNeutrop;
                    dataL=dataL.dataL;
                    data_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                    data_aux=data_aux(segm_aux>0);
                    id_dataL=unique(data_aux(data_aux(:)>0));
                    % Remap S_eroded
                    S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                    data_aux=zeros(size(eroded_img));
                    for m=1:height(S_eroded)
                        if (m<=length(id_dataL))
                            if (iscell(S_eroded.VoxelIdxList))
                                data_aux(S_eroded.VoxelIdxList{m})=id_dataL(m);
                            else
                                data_aux(S_eroded.VoxelIdxList(m))=id_dataL(m);
                            end
                        else
                            numNeutrop=numNeutrop+1;
                            if (iscell(S_eroded.VoxelIdxList))
                                data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                            else 
                                data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                            end
                        end
                    end
                    orig_aux=dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)));
                    for k=1:length(id_dataL)
                        orig_aux(orig_aux(:)==id_dataL(k))=0;
                    end
                    orig_aux(segm_aux>0)=data_aux(segm_aux>0);
                    dataL(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=orig_aux;
                    if (length(id_dataL)>height(S_eroded))
                        for k=1:length(id_dataL)-height(S_eroded)
                            dataL(dataL==numNeutrop)=id_dataL(length(id_dataL)-k+1);
                            numNeutrop=numNeutrop-1;
                        end
                    end
                    while (length(unique(dataL(:)))<(max(dataL(:))+1))
                        id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                        for k=1:length(id_absent)
                            dataL(dataL==max(dataL(:)))=k;
                            numNeutrop=numNeutrop-1;
                        end
                    end
                    if (height(S_eroded)>1)
                        segm_ref=dataL;
                    else
                        % Recharge original
                        segm_ref=scores_segm(:,:,:,2,handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(end)),5))>0.5;
                    end
                    save(dataOutName1,'dataL','numNeutrop','statsData');
                    dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(j)),5)));
                    dataSc=load(dataOutName2);
                    dataSc=dataSc.dataSc;
                    dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6)))=...
                                    dataSc(floor(bbox2(2)):floor(bbox2(5)),floor(bbox2(1)):floor(bbox2(4)),floor(bbox2(3)):floor(bbox2(6))).*double(eroded_img>0);
                    save(dataOutName2,'dataSc');  
                    already_solved=true;
                end
                j=j-1;
            end
        end
    end           
    
dataL=dataLa;
% Reload scores 
labels=zeros(size(scores));
for i=1:handles.numFrames
    xx=load(strcat(handles.dataLa,'/T',sprintf( '%05d',i)));
    labels(:,:,:,i)=xx.dataL;
end
disp('Pass 1. Done.');


%% Pass 2: without collisions. Kalman filter and morphological post-processing
disp('Pass 2. In process...');
handles = rmfield(handles,{'nodeNetwork'});
handles.nodeNetwork = positionNeutrophil(dataL);
handles = TrackingKalman3D(labels,handles, false);
    
% Morphological post-processing
for i=1:size(handles.finalNetwork,2)
    [~,neuts]=intersect(handles.nodeNetwork(:,6),handles.finalNetwork(:,i),'stable');
    if (length(neuts)>10)
        % Detect the outliers
        vol=handles.nodeNetwork(neuts,7);
        area=handles.nodeNetwork(neuts,11).*handles.nodeNetwork(neuts,12);
        isout=abs(vol/mean(vol))>5 | abs(vol/mean(vol))<1/5 | isoutlier(area,'median','ThresholdFactor',10);
        [~,L,~,C]=isoutlier(vol,'median','ThresholdFactor',3);
        id_out=find(isout);
        for j=1:length(id_out)
            if (vol(id_out(j))>L) % Agglomeration
                % Try to split, if possible
                time_id=handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(id_out(j))),5);
                segm=scores_segm(:,:,:,2,time_id);
                % Get the nearest reference plane
                valids=setdiff(1:length(isout),id_out);
                valids_id=handles.nodeNetwork(neuts(valids),5);
                [~,v_aux]=min(abs(valids_id-time_id));
                segm_aux=scores_segm(:,:,:,2,valids_id(v_aux));
                S_init=regionprops3(segm>0.5,'Centroid','VoxelIdxList','BoundingBox');
                pos_cell=[S_init.Centroid(:,1),S_init.Centroid(:,2),S_init.Centroid(:,3)];
                ds_neu=sqrt(sum((handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(id_out(j))),[2 1 3])-pos_cell).^2,2));
                [~,p]=min(ds_neu);
                neuts_aux=handles.nodeNetwork(handles.nodeNetwork(:,5)==valids_id(v_aux),:);
                id_aux=intersect(neuts_aux(:,6),neuts);
                centers_aux2 = handles.nodeNetwork(handles.nodeNetwork(:,6)==id_aux,1:3);
                S_aux=regionprops3(segm_aux>0.5,'Centroid','VoxelIdxList');
                dS=sqrt(sum((S_aux.Centroid(:,[2,1,3])-centers_aux2).^2,2));
                [m,p_aux]=min(dS);
                segm_aux=false(size(segm));
                segm_aux(S_aux.VoxelIdxList{p_aux})=true;
                centers_aux=S_aux.Centroid(p_aux,:);
                % Load original data
                load([dataRe,filesep,'T',sprintf('%05d',time_id),'.mat']);
                data=data(:,:,end-1,:);
                data=squeeze(max(data,[],3));
                % Cut the RoIs
                bbox=S_init.BoundingBox(p,:);
                bbox=min(max([bbox(1:3) bbox(1:3)+bbox(4:6)],[1 1 1 0 0 0]),[Inf Inf Inf size(segm,1) size(segm,2) size(segm,3)]);
                data_rec=data(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                segm_aux=segm_aux(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                centers_aux = centers_aux-floor([bbox(1),bbox(2),bbox(3)]);
                img_dist=bwdist(~(data_rec>graythresh(data_rec)));
                % Splitting method: compute overlaps and centroid distance
                % from erosion
                ths=unique(img_dist);
                overlaps=zeros(length(ths)-1,1);
                centroids=zeros(length(ths)-1,1);
                for id_er=1:length(ths)-1
                    id_erosion=ths(id_er);
                    img_aux=img_dist>id_erosion;
                    S_eroded= regionprops3(img_aux,'Centroid','VoxelIdxList');
                    [id,dS] = dsearchn(S_eroded.Centroid,centers_aux);
                    selected=false(size(img_aux));
                    if (iscell(S_eroded.VoxelIdxList))
                        selected(S_eroded.VoxelIdxList{id})=true;
                    else
                        selected(S_eroded.VoxelIdxList(id))=true;
                    end
                    selected2d=max(selected,[],3);
                    segm_aux2d=max(segm_aux,[],3);
                    overlaps(id_er)=sum(selected2d(:).*segm_aux2d(:))/(sum(selected2d(:))+sum(segm_aux2d(:))-sum(selected2d(:).*segm_aux2d(:)));
                    centroids(id_er,:)=sum((S_eroded.Centroid(id,:)-centers_aux).^2,2);
                end
                % Look for the point where the overlap is a local maximum 
                % owing to the splitting of regions.
                p=find(islocalmax(overlaps));
                p=setdiff(p,1);
                if (~isempty(p))
                    if (length(p)>1)
                        if (overlaps(p)>0.1)
                            [~,pp]=max(overlaps(p));
                            eroded_img=img_dist>ths(p(pp));
                        else
                            eroded_img=img_dist>ths(1);
                        end
                    else
                        if (overlaps(p)>0.1)
                            eroded_img=img_dist>ths(p);
                        else
                            eroded_img=img_dist>ths(1);
                        end
                    end
                else
                    % If not, look for a drop in centroid distance
                    if (~isempty(centroids))
                        changes=diff([centroids(1); centroids]);
                        [~,~,~,C]=isoutlier(changes,'median','ThresholdFactor',3);
                        % 3 pixels under the median: deviation
                        std_emp=3;
                        isoutC=((changes-C)/std_emp)<-3;
                        p=find(islocalmin(changes) & isoutC);
                        if (~isempty(p))
                            eroded_img=img_dist>ths(p(1));
                        else
                            [m,p]=max(overlaps);
                            if (m>0)
                                eroded_img=img_dist>ths(p);
                            else
                                eroded_img=img_dist>ths(1);
                            end
                        end
                    else
                        eroded_img=false(size(img_dist));
                    end
                end
                % Save the results
                dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',time_id));
                dataL=load(dataOutName1);
                statsData=dataL.statsData;
                numNeutrop=dataL.numNeutrop;
                dataL=dataL.dataL;
                data_aux=dataL(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                id_dataL=unique(data_aux(data_aux(:)>0));
                % Remap eroded_img
                S_eroded= regionprops3(eroded_img,'VoxelIdxList','Volume');
                % Filter by volume
                eroded_img = false(size(eroded_img));
                for k=1:height(S_eroded)
                    if (S_eroded.Volume(k)>110)
                        eroded_img(S_eroded.VoxelIdxList{k})=true;
                    end
                end
                S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                data_aux=zeros(size(eroded_img));
                for m=1:height(S_eroded)
                    if (m<=length(id_dataL))
                        if (iscell(S_eroded.VoxelIdxList))
                            data_aux(S_eroded.VoxelIdxList{m})=id_dataL(m);
                        else
                            data_aux(S_eroded.VoxelIdxList(m))=id_dataL(m);
                        end
                    else
                        numNeutrop=numNeutrop+1;
                        if (iscell(S_eroded.VoxelIdxList))
                            data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                        else
                            data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                        end
                    end
                end
                dataL(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)))=data_aux;
                if (length(id_dataL)>height(S_eroded))
                    for k=1:length(id_dataL)-height(S_eroded)
                        dataL(dataL==numNeutrop)=id_dataL(length(id_dataL)-k+1);
                        numNeutrop=numNeutrop-1;
                    end
                end
                S_dataL=regionprops3(dataL);
                if (~isempty(S_dataL))
                    S_dataL=sum(~isnan(S_dataL.Centroid(:,1)));
                    id_dataL=setdiff(unique(dataL(:)),0);
                    if (S_dataL<max(dataL(:)))
                        numNeutrop=S_dataL;
                        for k=S_dataL:-1:1
                            dataL(dataL==id_dataL(end-k+1))=k;
                        end
                    end
                else
                    numNeutrop=0;
                    statsData=S_dataL;
                end
                while (length(unique(dataL(:)))<(max(dataL(:))+1))
                    id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                    for k=1:length(id_absent)
                        dataL(dataL==max(dataL(:)))=k;
                        numNeutrop=numNeutrop-1;
                    end
                end
                save(dataOutName1,'dataL','numNeutrop','statsData');
                dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',time_id));
                dataSc=load(dataOutName2);
                dataSc=dataSc.dataSc;
                dataSc(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)))=...
                                dataSc(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6))).*eroded_img;
                save(dataOutName2,'dataSc');  
            elseif (vol(id_out(j))<C) % Separation
                % Try to join with other region, if possible
                time_id=handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(id_out(j))),5);
                % Get the nearest reference plane
                valids=setdiff(1:length(isout),id_out);
                valids_id=handles.nodeNetwork(neuts(valids),5);
                [~,v_aux]=min(abs(valids_id-time_id));
                segm=scores_segm(:,:,:,2,time_id);
                segm_aux=scores_segm(:,:,:,2,valids_id(v_aux));
                S_init=regionprops3(segm_aux>0.5,'Centroid','VoxelIdxList','BoundingBox');
                pos_cell=[S_init.Centroid(:,1),S_init.Centroid(:,2),S_init.Centroid(:,3)];
                ds_neu=sqrt(sum((handles.nodeNetwork(find(handles.nodeNetwork(:,6)==neuts(valids(v_aux))),[2 1 3])-pos_cell).^2,2));
                [~,p]=min(ds_neu);
                neut_full=false(size(segm_aux));
                neut_full(S_init.VoxelIdxList{p})=true;
                centers_aux=S_init.Centroid(p,:);
                S_sep=regionprops3(segm>0.5,'Centroid','VoxelIdxList');
                % Get nearest regions
                segm=false(size(segm));
                for m=1:height(S_sep)
                    neut_sep=false(size(segm));
                    neut_sep(S_sep.VoxelIdxList{m})=true; 
                    neut_full2d=max(neut_full,[],3);
                    neut_sep2d=max(neut_sep,[],3);
                    if (sum(neut_sep2d(:).*neut_full2d(:))/(sum(neut_sep2d(:))+sum(neut_full2d(:))-sum(neut_sep2d(:).*neut_full2d(:)))>0)
                        segm(S_sep.VoxelIdxList{m})=true;
                    end
                end
                % Load original data
                load([dataRe,filesep,'T',sprintf('%05d',time_id),'.mat']);
                data=data(:,:,end-1,:);
                data=squeeze(max(data,[],3));
                % Detect if there are 2 close regions 
                is_sep=sum(segm(neut_full(:)))>0;
                if (is_sep)
                    bbox=[find(sum(sum(segm,3),1),1,'first')-1 find(sum(sum(segm,3),2),1,'first')-1 find(sum(sum(segm,1),2),1,'first')-1 ...
                        find(sum(sum(segm,3),1),1,'last')-find(sum(sum(segm,3),1),1,'first')+1 ...
                        find(sum(sum(segm,3),2),1,'last')-find(sum(sum(segm,3),2),1,'first')+1 ... 
                        find(sum(sum(segm,1),2),1,'last')-find(sum(sum(segm,1),2),1,'first')+1];
                else
                    bbox=[find(sum(sum(neut_full,3),1),1,'first')-1 find(sum(sum(neut_full,3),2),1,'first')-1 find(sum(sum(neut_full,1),2),1,'first')-1 ...
                        find(sum(sum(neut_full,3),1),1,'last')-find(sum(sum(neut_full,3),1),1,'first')+1 ...
                        find(sum(sum(neut_full,3),2),1,'last')-find(sum(sum(neut_full,3),2),1,'first')+1 ... 
                        find(sum(sum(neut_full,1),2),1,'last')-find(sum(sum(neut_full,1),2),1,'first')+1];
                end
                bbox=min(max([bbox(1:3) bbox(1:3)+bbox(4:6)],[1 1 1 0 0 0]),[Inf Inf Inf size(segm,1) size(segm,2) size(segm,3)]);
                data_rec=data(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                neut_full=neut_full(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                centers_aux = centers_aux-floor([bbox(1),bbox(2),bbox(3)]);
                img_dist=bwdist(~(data_rec>graythresh(data_rec)));
                ths=unique(img_dist);
                % Method: compute overlaps and centroid distance
                % from erosion
                overlaps=zeros(length(ths)-1,1);
                num_regions=zeros(length(ths)-1,1);
                for id_er=1:length(ths)-1
                    id_erosion=ths(id_er);
                    img_aux=img_dist>id_erosion;
                    S_eroded= regionprops3(img_aux,'Centroid','VoxelIdxList');
                    [id,dS] = dsearchn(S_eroded.Centroid,centers_aux);
                    selected=false(size(img_aux));
                    if (iscell(S_eroded.VoxelIdxList))
                        selected(S_eroded.VoxelIdxList{id})=true;
                    else
                        selected(S_eroded.VoxelIdxList(id))=true;
                    end
                    selected2d=max(selected,[],3);
                    neut_full2d=max(neut_full,[],3);
                    overlaps(id_er)=sum(selected2d(:).*neut_full2d(:))/(sum(selected2d(:))+sum(neut_full2d(:))-sum(selected2d(:).*neut_full2d(:)));
                    num_regions(id_er)=height(S_eroded);
                end
                % Look for the point where the two regions join.
                overlaps=overlaps(num_regions==1);
                if (~isempty(overlaps))
                    [m,p]=max(overlaps);
                    if (m>0)
                        eroded_img=img_dist>ths(p);
                    else
                        eroded_img=img_dist>0;
                    end 
                else
                    eroded_img=img_dist>0;
                end
                eroded_img=imfill(eroded_img,'holes');
                CC = bwconncomp(eroded_img);
                numPixels = cellfun(@numel,CC.PixelIdxList);
                [~,idx] = max(numPixels);
                eroded_img = false(size(eroded_img));
                if (~isempty(idx))
                    eroded_img(CC.PixelIdxList{idx}) = true;
                end
                % Save the results
                dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',time_id),'.mat');
                dataL=load(dataOutName1);
                numNeutrop=dataL.numNeutrop;
                dataL=dataL.dataL;
                data_aux=dataL(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)));
                id_dataL=mode(data_aux(data_aux(:)>0));
                if (is_sep)
                    % The number of regions remain
                    S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                    data_aux=zeros(size(eroded_img));
                    for m=1:height(S_eroded)
                        if (iscell(S_eroded.VoxelIdxList))
                            data_aux(S_eroded.VoxelIdxList{m})=id_dataL;
                        else
                            data_aux(S_eroded.VoxelIdxList(m))=id_dataL;
                        end
                    end
                else
                    numNeutrop=numNeutrop+1;
                    % Add a region
                    S_eroded= regionprops3(eroded_img,'Centroid','VoxelIdxList');
                    data_aux=zeros(size(eroded_img));
                    for m=1:height(S_eroded)
                        if (iscell(S_eroded.VoxelIdxList))
                            data_aux(S_eroded.VoxelIdxList{m})=numNeutrop;
                        else
                            data_aux(S_eroded.VoxelIdxList(m))=numNeutrop;
                        end
                    end
                end
                dataL(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)))=data_aux;
                dataL=bwlabeln(dataL>0);
                numNeutrop=max(dataL(:));
                statsData=regionprops(dataL,'Area','Centroid','BoundingBox');
                while (length(unique(dataL(:)))<(max(dataL(:))+1))
                    id_absent=setdiff(1:max(dataL(:)),unique(dataL(:)));
                    for k=1:length(id_absent)
                        dataL(dataL==max(dataL(:)))=k;
                        numNeutrop=numNeutrop-1;
                    end
                end
                save(dataOutName1,'dataL','numNeutrop','statsData');
                dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',time_id),'.mat');
                dataSc=load(dataOutName2);
                dataSc=dataSc.dataSc;
                dataSc(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)))=...
                                max(max(max(dataSc(floor(bbox(2)):floor(bbox(5)),floor(bbox(1)):floor(bbox(4)),floor(bbox(3)):floor(bbox(6)))))).*eroded_img;
                save(dataOutName2,'dataSc');  
            end
        end
    end
end
else
%There are no nodes to process
disp(...
    strcat('It was NOT possible to detect any node in the data,',...
           'verify the threshold levels!'));
handles = [];
return;
end

dataL=dataLa;
% Reload scores 
labels=zeros(size(scores));
for i=1:handles.numFrames
    xx=load(strcat(handles.dataLa,'/T',sprintf( '%05d',i)));
    labels(:,:,:,i)=xx.dataL;
end
disp('Pass 2. Done.');
%% Pass 3: without outliers. Final kalman filter.
disp('Pass 3. In process...');
handles = rmfield(handles,{'nodeNetwork'});
handles.nodeNetwork= positionNeutrophil(dataL);
handles = TrackingKalman3D(labels,handles, false);
% Compute distance network from trajectories
handles.distanceNetwork = getDistanceNet(handles.finalNetwork,handles.nodeNetwork);
dataHa = strcat(dataLa(1:end-2),'Ha');
mkdir(dataHa)
save(strcat(dataHa,'/handles.mat'),'handles');
disp('Pass 3. Done.');

end
