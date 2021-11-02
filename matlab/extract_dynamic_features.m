% Load data
clear, close all;
config;
% Load the segmentation and tracking results
d=dir('results/*/*/T*.mat');
% Load data descriptions for Z-axis and time normalization
filename = '../data/data_description.xlsx';
corrections = xlsread(filename);
corrections = corrections(:,[8 9 1 6 7]); 

for vol=1:length(sequences)
    fprintf('VOLUME: %d\n',vol);
    aux=strsplit(sequences{vol},'/');
    volumes_unet=dir(['results/' aux{end-1} '/' aux{end} '_La/T*.mat']);
    id_neu=0;
    if (exist(['results/' aux{end-1} '/' aux{end} '_Ha/handles.mat'],'file'))
        load(['results/' aux{end-1} '/' aux{end} '_Ha/handles.mat']);
        d=dir(['../database/venules_init_pred/',aux{end-1},'__',aux{end},'__','*.mat']);
        % Correct the Z-axis and time 
        group=str2double(aux{end-1}(end));
        aux2=strsplit(aux{end},'_');
        volume=str2double(aux2{1});
        capture=str2double(aux2{2}(8:end));
        FACTORS=corrections(sum(corrections(:,1:3)==[group,volume,capture],2)==3,4:5);
        ZFACTOR=FACTORS(1)/v_size(1);
        TFACTOR=FACTORS(2);
        % Obtain the venule across time: microscopy movement is negligible
        venule_full=zeros(handles.rows,handles.cols,handles.levs);
        for i=1:length(d)
            % Estimated venule data
            venule=load([d(i).folder filesep d(i).name]);
            venule=venule.venule;
            venule_full=venule_full+double(venule);
        end
        venule=venule_full>handles.numFrames/2;
        if (INTERP)
            venule = imresize3(double(venule),[size(venule,1),size(venule,2),round(size(venule,3)*ZFACTOR)])>0.5;
            handles.nodeNetwork(:,3)=handles.nodeNetwork(:,3)*ZFACTOR;
        end
        % Venule segmentation: we delete small regions
        venule=imfill(venule,'holes');
        S_venule=regionprops3(venule,'Volume','VoxelIdxList');
        venule=false(size(venule));
        [~,p]=max(S_venule.Volume(:));
        venule(S_venule.VoxelIdxList{p})=true;
        % Obtain the venule boundary
        venule_boundary = imdilate(venule, true(3,3,3)) - venule;
        S_extern=regionprops3(venule_boundary,'VoxelList','VoxelIdxList');
        plane_xy=max(venule,[],3);
        % Get the skeleton (venule center) in 2D, as an ordered
        % contour
        skel2D=false(size(plane_xy));
        xdir=false;
        if (sum(plane_xy(:,1))>sum(plane_xy(1,:)))
            for i=1:size(plane_xy,2)
                center=round(median(find(plane_xy(:,i))));
                if (~isnan(center))
                    skel2D(center,i)=true;
                end
            end
            xdir=true;
        else
            for i=1:size(plane_xy,1)
                center=round(median(find(plane_xy(i,:))));
                if (~isnan(center))
                    skel2D(i,center)=true;
                end
            end
        end
        if plane_xy(1,1)
            initH=max(find(plane_xy(1,:)>0));
            initV=max(find(plane_xy(:,1)>0));
            skel2D(1:initV,1:initH)=false;
            if (xdir)
                skel2D(:,1:initH)=false;
            else
                skel2D(1:initV,:)=false;
            end
            m=round((initH+initV)/2);
            if (initH>initV)
                p2=[1+(m-initV),1];
            else
                p2=[1,1+(m-initH)];
            end
            S_skel=regionprops(skel2D,'PixelList');
            p1=cat(1,S_skel(:).PixelList);
            p=dsearchn(p1,[1,1]);
            p1=p1(p,:);
            points=[linspace(p1(1),p2(1),1000);linspace(p1(2),p2(2),1000)]';
            points=unique(round(points),'rows');
            skel2D(sub2ind(size(skel2D),points(:,2),points(:,1)))=true;
        end
        if plane_xy(1,size(plane_xy,2))
            initH=min(find(plane_xy(1,:)>0));
            initV=max(find(plane_xy(:,size(plane_xy,2))>0));
            skel2D(1:initV,initH:end)=false;
            if (xdir)
                skel2D(:,initH:end)=false;
            else
                skel2D(1:initV,:)=false;
            end
            m=round((size(plane_xy,2)-initH+initV)/2);
            if ((size(plane_xy,2)-initH)>initV)
                p2=[size(plane_xy,2)-(m-initV),1];
            else
                p2=[size(plane_xy,2),1+(m-(size(plane_xy,2)-initH))];
            end
            S_skel=regionprops(skel2D,'PixelList');
            p1=cat(1,S_skel(:).PixelList);
            p=dsearchn(p1,[size(plane_xy,2),1]);
            p1=p1(p,:);
            points=[linspace(p1(1),p2(1),1000);linspace(p1(2),p2(2),1000)]';
            points=unique(round(points),'rows');
            skel2D(sub2ind(size(skel2D),points(:,2),points(:,1)))=true;
        end
        if plane_xy(size(plane_xy,1),size(plane_xy,2))
            initH=min(find(plane_xy(size(plane_xy,1),:)>0));
            initV=min(find(plane_xy(:,size(plane_xy,2))>0));
            skel2D(initV:end,initH:end)=false;
            if (xdir)
                skel2D(:,initH:end)=false;
            else
                skel2D(initV:end,:)=false;
            end
            m=round((size(plane_xy,2)-initH+size(plane_xy,1)-initV)/2);
            if ((size(plane_xy,2)-initH)>(size(plane_xy,1)-initV))
                p2=[size(plane_xy,1)-(m-(size(plane_xy,1)-initV)),size(plane_xy,2)];
            else
                p2=[size(plane_xy,1),size(plane_xy,2)-(m-(size(plane_xy,2)-initH))];
            end
            S_skel=regionprops(skel2D,'PixelList');
            p1=cat(1,S_skel(:).PixelList);
            p=dsearchn(p1,[size(plane_xy,1),size(plane_xy,2)]);
            p1=p1(p,:);
            points=[linspace(p1(1),p2(1),1000);linspace(p1(2),p2(2),1000)]';
            points=unique(round(points),'rows');
            skel2D(sub2ind(size(skel2D),points(:,2),points(:,1)))=true;
        end
        if plane_xy(size(plane_xy,1),1)
            initH=max(find(plane_xy(size(plane_xy,1),:)>0));
            initV=min(find(plane_xy(:,1)>0));
            skel2D(initV:end,1:initH)=false;
            if (xdir)
                skel2D(:,1:initH)=false;
            else
                skel2D(initV:end,:)=false;
            end
            m=round((size(plane_xy,1)-initV+initH)/2);
            if (initH>(size(plane_xy,1)-initV))
                p2=[1+(m-(size(plane_xy,1)-initV)),size(plane_xy,1)];
            else
                p2=[1,size(plane_xy,1)-(m-initH)];
            end
            S_skel=regionprops(skel2D,'PixelList');
            p1=cat(1,S_skel(:).PixelList);
            p=dsearchn(p1,[1,size(plane_xy,1)]);
            p1=p1(p,:);
            points=[linspace(p1(1),p2(1),1000);linspace(p1(2),p2(2),1000)]';
            points=unique(round(points),'rows');
            skel2D(sub2ind(size(skel2D),points(:,2),points(:,1)))=true;
        end  
        skel=false(size(venule));
        if (sum(plane_xy(:,1))>sum(plane_xy(1,:)))
            pos_init=[0,0,0];
            process=true;
            for i=1:size(skel,2)
                plane_z=sum(squeeze(venule(:,i,:)),1);
                [m,~]=max(plane_z);
                z_id=round(median(find(plane_z==m)));
                skel(:,i,z_id)=skel2D(:,i);
                if (process && ~(isempty(find(skel2D(:,i),1,'first'))))
                    pos_init=[find(skel2D(:,i),1,'first'),i,z_id];
                    process=false;
                end
            end
            % Obtain the ordered contour from the initial point
            ordered=order_contour3D(skel,sub2ind(size(skel),pos_init(1),pos_init(2),pos_init(3)));
        else
            pos_init=[0,0,0];
            process=true;
            for i=1:size(skel,1)
                plane_z=sum(squeeze(venule(i,:,:)),1);
                [m,~]=max(plane_z);
                z_id=round(median(find(plane_z==m)));
                skel(i,:,z_id)=skel2D(i,:);
                if (process && ~(isempty(find(skel2D(i,:),1,'first'))))
                    pos_init=[i,find(skel2D(i,:),1,'first'),z_id];
                    process=false;
                end
            end
            % Obtain the ordered contour from the initial point
            ordered=order_contour3D(skel,sub2ind(size(skel),pos_init(1),pos_init(2),pos_init(3)));
        end
        ordered_sub=zeros(length(ordered),3);
        [ordered_sub(:,2),ordered_sub(:,1),ordered_sub(:,3)]=ind2sub(size(skel),ordered);
        
        % Load instantaneous features
        load(['results/' aux{end-1} '/' aux{end} '_Ha/' id_features '.mat'],'features_simple');
        % Dynamic features
        features_track=zeros(size(features_simple,1),N_feat_dyn);
        
        % For each track
        for i=1:size(handles.finalNetwork,2)
            for j=1:sum(handles.finalNetwork(:,i)>0)
                % Cells in each track
                id_neutrophil=find(features_simple(:,25)==handles.finalNetwork(j,i));  
                if (~isempty(id_neutrophil))
                    % Trajectory features
                    features_track(id_neutrophil,1)=handles.distanceNetwork.numHops(i);
                    features_track(id_neutrophil,2)=handles.distanceNetwork.totPerTrack(i);
                    features_track(id_neutrophil,3)=handles.distanceNetwork.maxPerTrack(i)/TFACTOR;
                    features_track(id_neutrophil,4)=handles.distanceNetwork.avPerTrack(i)/TFACTOR;
                    if (~isnan(handles.distanceNetwork.meanderRatio(i)) && ~isnan(handles.distanceNetwork.meanderRatio(i)))
                        features_track(id_neutrophil,5)=handles.distanceNetwork.meanderRatio(i);
                    else
                        features_track(id_neutrophil,5)=-1;
                    end
                    if (~isnan(handles.distanceNetwork.tortuosity(i)) && ~isinf(handles.distanceNetwork.tortuosity(i)))
                        features_track(id_neutrophil,6)=handles.distanceNetwork.tortuosity(i);
                    else
                        features_track(id_neutrophil,6)=-1;
                    end
                    features_track(id_neutrophil,7)=handles.distanceNetwork.angleTrack(i);
                    % Locate the cells in handles
                    [~,neuts]=intersect(features_simple(:,25),handles.finalNetwork(:,i),'stable');
                    ds=abs(handles.nodeNetwork(neuts,5)-handles.nodeNetwork(id_neutrophil,5));
                    [~,ids]=sort(ds,'ascend');
                    time_id=handles.nodeNetwork(id_neutrophil,5);
                    % Normalized Hann window
                    L=length(neuts);
                    if (L>=DIST)
                        W=hann(2*DIST+1);
                        W=padarray(W,[L-DIST,0]);
                    else
                        W=hann(2*DIST+1);
                        W=W(DIST-L+1:length(W)-(DIST-L));
                    end
                    center=(length(W)-1)/2+1;
                    W=W(center-(abs(min(ids)-time_id)):center+(max(ids)-time_id));
                    W=W./sum(W);
                    % Extract angle from venule
                    trajectory=round(handles.nodeNetwork(neuts,1:3));
                    trajectory=trajectory(sum(isnan(trajectory),2)==0,:);
                    plane=zeros(256,256);
                    for k=1:size(trajectory,1)
                        plane(trajectory(k,1),trajectory(k,2))=1;
                    end
                    if (xdir)
                        [~,id_min] = min(trajectory(:,2));
                        [~,id_max] = max(trajectory(:,2));
                        [id_1,~] = dsearchn(ordered_sub(:,[2 1 3]),trajectory(id_min,:));
                        [id_2,~] = dsearchn(ordered_sub(:,[2 1 3]),trajectory(id_max,:));
                        diffBetPoints = ordered_sub(id_1,1:2)-ordered_sub(id_2,1:2);
                        distFromStart = sqrt((sum(diffBetPoints.^2,2)));
                        angleVenule = acos(diffBetPoints(:,1)./(distFromStart+1e-30));
                    else
                        [~,id_min] = min(trajectory(:,1));
                        [~,id_max] = max(trajectory(:,1));
                        [id_1,~] = dsearchn(ordered_sub(:,[2 1 3]),trajectory(id_min,:));
                        [id_2,~] = dsearchn(ordered_sub(:,[2 1 3]),trajectory(id_max,:));
                        diffBetPoints = ordered_sub(id_1,1:2)-ordered_sub(id_2,1:2);
                        distFromStart = sqrt((sum(diffBetPoints.^2,2)));
                        angleVenule = acos(diffBetPoints(:,1)./(distFromStart+1e-30));
                    end
                    % Update the orientation feature: differential
                    % orientation between the cell and the venule
                    features_track(id_neutrophil,7)=features_track(id_neutrophil,7)-angleVenule; 
                    % Filter outliers: if necessary (not in our case)
                    isout=zeros(size(features_simple(neuts,1)));
                    neuts_valids=neuts(~isout);
                    features_track(id_neutrophil,8:21)=sum(features_simple(neuts(~isout),1:14).*repmat(W,[1,14]));
                    features_track(id_neutrophil,22:35)=std(features_simple(neuts(~isout),1:14),W,1);
                    % If captures are not isometric, interpolate and
                    % recalculate trajectory features
                    if (INTERP)
                        distances_track=sqrt(sum(diff(trajectory).^2,2));
                        features_track(id_neutrophil,2)=sum(distances_track);
                        features_track(id_neutrophil,3)=max(distances_track)/TFACTOR;
                        features_track(id_neutrophil,4)=mean(distances_track)/TFACTOR;
                        dist_line=sqrt(sum((handles.nodeNetwork(neuts_valids(1),1:3)-handles.nodeNetwork(neuts_valids(end),1:3)).^2,2));
                        features_track(id_neutrophil,5)=sum(distances_track)/dist_line;
                        features_track(id_neutrophil,6)=dist_line/sum(distances_track);
                    end
                    % Polar position features
                    % Position
                    if (sum(isnan(features_simple(neuts,18)))==0)
                        features_track(id_neutrophil,36:37)=sum(features_simple(neuts(~isout),18:19).*repmat(W,[1,2]),1);
                        features_track(id_neutrophil,38:39)=std(features_simple(neuts(~isout),18:19),W,1);
                    else
                        features_track(id_neutrophil,36:39)=-1;
                    end
                    % Orientation
                    if (sum(sum(isnan(features_simple(neuts,15:17))))==0)
                        features_track(id_neutrophil,40:42)=sum(features_simple(neuts(~isout),15:17).*repmat(W,[1,3]),1);
                        features_track(id_neutrophil,43:45)=std(features_simple(neuts(~isout),15:17),W,1);
                    else
                        features_track(id_neutrophil,40:45)=-1;
                    end
                    % Cell to blood vessel distance features
                    features_track(id_neutrophil,46)=sum(features_simple(neuts(~isout),20).*W,1);
                    features_track(id_neutrophil,47)=sum(features_simple(neuts(~isout),21).*W,1);
                    features_track(id_neutrophil,48)=min(features_simple(neuts(~isout),21),[],1);
                    % Well-segmented cell probability features
                    features_track(id_neutrophil,49)=sum(features_simple(neuts(~isout),28).*W,1);
                    features_track(id_neutrophil,50)=max(features_simple(neuts(~isout),28),[],1);
                    features_track(id_neutrophil,51)=min(features_simple(neuts(~isout),28),[],1);
                    features_track(id_neutrophil,52)=max(features_simple(neuts(~isout),28),[],1)-min(features_simple(neuts(~isout),28),[],1);
                    features_track(id_neutrophil,53)=isout(find(neuts==id_neutrophil));
                end
           end
        end
        save(['results/' aux{end-1} '/' aux{end} '_Ha/' id_features '_track.mat'],'features_track');
    end
end
disp('end');

function[ordered]=order_contour3D(edges, index_init)
    indexes=find(edges(:));
    [y,x,z]=ind2sub(size(edges),indexes);
    % Ordenar el contorno
    ordered=zeros(length(y),1);
    ordered(1)=find(indexes==index_init);
    processed=zeros(size(ordered));
    processed(find(indexes==index_init))=1;
    for j=2:length(ordered)
        ds=((z-z(ordered(j-1))).^2+(y-y(ordered(j-1))).^2+(x-x(ordered(j-1))).^2);
        ds(processed==1)=Inf;
        [~,next]=min(ds);
        processed(next)=1;
        ordered(j)=next;
    end
    ordered=indexes(ordered);
end