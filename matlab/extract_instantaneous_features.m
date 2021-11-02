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
    % Temporal slices
    volumes_net=dir(['results/' aux{end-1} '/' aux{end} '_La/T*.mat']);
    features_simple=zeros(0,N_feat_inst);
    id_neu=0;
    % Load the tracking results
    if (exist(['results/' aux{end-1} '/' aux{end} '_Ha/handles.mat'],'file'))
        load(['results/' aux{end-1} '/' aux{end} '_Ha/handles.mat']);
        % Correct the Z-axis and time 
        group=str2double(aux{end-1}(end));
        aux2=strsplit(aux{end},'_');
            volume=str2double(aux2{1});
            capture=str2double(aux2{2}(8:end));
        FACTORS=corrections(sum(corrections(:,1:3)==[group,volume,capture],2)==3,4:5);
        ZFACTOR=FACTORS(1)/v_size(1);
        TFACTOR=FACTORS(2);
        % Interpolate the Z-axis
        if (INTERP)
            % Interpolation if necessary
            handles.nodeNetwork(:,3)=handles.nodeNetwork(:,3)*ZFACTOR;
        end
        for seq=1:length(volumes_net)
            % Segmentation from ACME
            segmented=load([volumes_net(seq).folder '/' volumes_net(seq).name]);
            segmented=segmented.dataL;

            % Load segmentation scores
            if (exist([strrep(volumes_net(seq).folder,'La','Sc') '/' volumes_net(seq).name],'file'))
                dataSc=load([strrep(volumes_net(seq).folder,'La','Sc') '/' volumes_net(seq).name]);
                dataSc=dataSc.dataSc;
            else
                dataSc=ones(size(segmented));
            end
            % Load venule segmentations
            venule=load(['../database/venules_init_pred/',aux{end-1},'__',aux{end},'__',volumes_net(seq).name]);
            venule=venule.venule;
            % Venule segmentation: we delete small regions
            venule=imfill(venule,'holes');
            S_venule=regionprops3(venule,'Volume','VoxelIdxList');
            if (height(S_venule)>0)
                venule=false(size(venule));
                [~,p]=max(S_venule.Volume(:));
                venule(S_venule.VoxelIdxList{p})=true;
                if (INTERP)
                    % Interpolation if necessary
                    segmented = imresize3(double(segmented),[size(segmented,1),size(segmented,2),round(size(segmented,3)*ZFACTOR)],'nearest');
                    dataSc = imresize3(dataSc,[size(dataSc,1),size(dataSc,2),round(size(dataSc,3)*ZFACTOR)]);
                    venule = imresize3(double(venule),[size(venule,1),size(venule,2),round(size(venule,3)*ZFACTOR)])>0.5;
                end
                % Obtain the venule boundary
                venule_boundary = imdilate(venule, true(3,3,3)) - venule;
                S_extern=regionprops3(venule_boundary,'VoxelList','VoxelIdxList');
                % Get the skeleton (venule center) in 2D, as an ordered
                % contour
                plane_xy=max(venule,[],3);
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
                % Current time instant
                tt=str2double(volumes_net(seq).name(2:end-4));
                % 3D properties
                S=regionprops3(segmented,'all');
                segmented=segmented>0;
                % Cells in this instant of time
                time_neu=handles.nodeNetwork(handles.nodeNetwork(:,5)==tt,[2 1 3 6]);
                for i=1:size(time_neu)
                    if (S.Volume(i)~=0)
                        % Locate the cell index in the handles
                        pos_cell=[S.Centroid(:,1),S.Centroid(:,2),S.Centroid(:,3)];
                        ds_neu=sqrt(sum((time_neu(i,1:3)-pos_cell).^2,2));
                        [~,p]=min(ds_neu);
                        % Cell in the original volume
                        cell_pred=false(size(segmented));
                        cell_pred(S.VoxelIdxList{p})=true;
                        id_neu=id_neu+1;
                        % New row for the cell
                        features_simple=[features_simple;zeros(1,N_feat_inst)];
                        % We must index the cell: file, capture, index
                        features_simple(id_neu,22)=str2double(aux{end-1}(end));
                        aux2=strsplit(aux{end},'_');
                        features_simple(id_neu,23)=str2double(aux2{1});
                        features_simple(id_neu,24)=str2double(aux2{2}(8:end));
                        features_simple(id_neu,25)=time_neu(i,4);
                        % Instantaneous feature computation
                        % Shape features, as in manual approach (CNIC)
                        features_simple(id_neu,1)=S.Volume(i);
                        features_simple(id_neu,2)=S.SurfaceArea(i);
                        spher=(pi^(1/3)*(6*S.Volume(i))^(2/3))/S.SurfaceArea(i);
                        features_simple(id_neu,6)=spher;
                        % From Imaris Manual
                        % If: a<=b<=c then when:
                            % a=0 It is an ellipse
                            % a=b=c It is a sphere
                            % a~=b~=c It is a scalene ellipsoid (three unequal sides)
                        % If two of these sides are equal, the ellipsoid is a spheroid:
                        % a = b < c  it is a prolate Spheroid (cigar-shaped)
                        % a < b = c it is an oblate Spheroid (disk-shaped)
                        % In any case, we can calculate
                        eig_values=S.EigenValues(i);
                        a=S.PrincipalAxisLength(i,3);
                        b=S.PrincipalAxisLength(i,2);
                        c=S.PrincipalAxisLength(i,1);
                        e_prolate=2*a^2/(a^2+b^2)*(1-(a^2+b^2)/(2*c^2));
                        e_ablate=2*b^2/(b^2+c^2)*(1-(2*a^2)/(b^2+c^2));
                        % Ellipticities
                        features_simple(id_neu,7)=e_prolate;
                        features_simple(id_neu,8)=e_ablate;
                        % Principal axes length
                        features_simple(id_neu,9)=S.PrincipalAxisLength(i,1); 
                        features_simple(id_neu,10)=S.PrincipalAxisLength(i,2); 
                        features_simple(id_neu,11)=S.PrincipalAxisLength(i,3);
                        % Segmentation features
                        features_simple(id_neu,12)=S.Extent(i);
                        features_simple(id_neu,13)=S.Solidity(i);
                        % Equivalent diameter
                        features_simple(id_neu,14)=S.EquivDiameter(i);

                        % Other features
                        % Centroid distance to venule surface
                        [id,dS] = dsearchn(S_extern.VoxelList{1},S.Centroid(i,:));
                        cut_centroid=S_extern.VoxelList{1}(id,:);
                        centroid=S.Centroid(i,:);
                        features_simple(id_neu,20)=min(dS);
                        % Minimum distance to the venule surface
                        [id,dS] = dsearchn(S_extern.VoxelList{1},S.VoxelList{i});
                        [min_dist,]=min(dS);
                        id_int=dS==min_dist;
                        id_ext=id(id_int);
                        id_int=find(id_int);
                        % Get the median one
                        features_simple(id_neu,21)=min_dist;
                        cut_cell=round(median(S.VoxelList{i}(id_int,:),1));
                        [id,dS] = dsearchn(S_extern.VoxelList{1},cut_cell);
                        cut_p=S_extern.VoxelList{1}(id,:);

                        % Rotate the capture in bloodstream direction in the nearest point to the cell
                        [id,dS] = dsearchn(ordered_sub,centroid);
                        p1=ordered_sub(max(1,id-range_orientation),:);
                        p2=ordered_sub(min(length(ordered_sub),id+range_orientation),:);
                        p11=ordered(max(1,id-range_orientation),:);
                        p12=ordered(min(length(ordered_sub),id+range_orientation),:);
                        skel_min=zeros(size(skel));
                        skel_min(ordered(max(1,id-range_orientation):min(length(ordered_sub),id+range_orientation)))=true;
                        % Obtain bloodstream direction in the cell center
                        % plane
                        SS=regionprops3(skel_min,'Orientation');
                        angle_xy=90+SS.Orientation(1);
                        venule_boundary_rot = imrotate3(double(venule_boundary),-angle_xy,[0,0,1])>0.5;
                        cell_pred_rot = imrotate3(double(cell_pred),-angle_xy,[0,0,1])>0.5;
                        centroid_rot=[cosd(angle_xy) -sind(angle_xy); sind(angle_xy) cosd(angle_xy)]*(centroid(1:2)-size(venule_boundary,1)/2)'+(size(venule_boundary_rot,1)-size(venule_boundary,1))/2+size(venule_boundary,1)/2;
                        % Differential orientation features 
                        features_simple(id_neu,15)=S.Orientation(i,1)-SS.Orientation(1); 
                        features_simple(id_neu,16)=S.Orientation(i,2)-SS.Orientation(2);
                        features_simple(id_neu,17)=S.Orientation(i,3)-SS.Orientation(3);

                        % Obtain the plane where the cell center is over
                        % the rotated venule. Obtain the elliptical cross -
                        % section by least squares
                        mask=squeeze(venule_boundary_rot(:,round(centroid_rot(1)),:))';
                        indexes=find(mask(:));
                        [y,x]=ind2sub(size(mask),indexes);
                        if (~isempty(x))
                            ellipse=funcEllipseFit_BFisher([x,y]);
                            if (~isempty(ellipse))
                                if (ellipse(3)<ellipse(4))
                                    ellipse(3:4)=ellipse(4:-1:3);
                                    if (ellipse(5)>0)
                                        ellipse(5)=ellipse(5)-pi/2;
                                    else
                                        ellipse(5)=ellipse(5)+pi/2;
                                    end
                                end
                            else
                                ellipse=[0 0 0 0 0];
                            end
                        else
                            ellipse=[0 0 0 0 0];
                        end
                        % Obtain the cell polar position
                        coords=getPolarCoordinatesAngle(size(mask),[ellipse(2),ellipse(1)],ellipse(3),ellipse(4),ellipse(5));
                        features_simple(id_neu,18)=coords(round(centroid(3)),max(round(centroid_rot(2)),1),1);
                        features_simple(id_neu,19)=coords(round(centroid(3)),max(round(centroid_rot(2)),1),2);

                        % Height calculation with 3D capture normals
                        [y,x,z]=meshgrid(1:size(venule,2),1:size(venule,1),1:size(venule,3));
                        % Compute the isosurface for the venule boundary
                        fv = isosurface(x,y,z,venule_boundary,1-eps);
                        coord_std=zeros(3,1);
                        coord_max=zeros(3,1);
                        for t=1:3
                            cunique=unique(fv.vertices(:,t));
                            aux_v=zeros(length(cunique),1);
                            for s=1:length(cunique)
                                aux_v(s)=sum(fv.vertices(:,t)==cunique(s));
                            end
                            coord_std(t)=std(aux_v);
                            coord_max(t)=max(aux_v);
                        end
                        [~,p]=min(coord_std);
                        xs=zeros(size(venule,p),coord_max(p));
                        ys=zeros(size(venule,p),coord_max(p));
                        zs=zeros(size(venule,p),coord_max(p));
                        valids=true(size(xs));
                        for s=1:size(venule,p)
                            aux_v=fv.vertices(fv.vertices(:,p)==s,:);
                            if (size(aux_v,1)<coord_max(p))
                                xs(s,:)=[aux_v(:,2)' zeros(1,coord_max(p)-size(aux_v,1))];
                                ys(s,:)=[aux_v(:,1)' zeros(1,coord_max(p)-size(aux_v,1))];
                                zs(s,:)=[aux_v(:,3)' zeros(1,coord_max(p)-size(aux_v,1))];
                                valids(:,size(aux_v,1)+1:end)=false;
                            else
                                xs(s,:)=aux_v(:,2);
                                ys(s,:)=aux_v(:,1);
                                zs(s,:)=aux_v(:,3);
                            end
                        end
                        % Normals: get the minimum distance from one normal
                        % to the centroid
                        [Nx,Ny,Nz]=surfnorm(xs,ys,zs);
                        ds=zeros(size(xs));
                        for s=1:size(xs,1)
                            for t=1:size(xs,2)
                                if (valids(s,t))
                                    [ds(s,t), Q] = point_to_line(centroid,[xs(s,t),ys(s,t),zs(s,t)],[xs(s,t)+10*Nx(s,t),ys(s,t)+10*Ny(s,t),zs(s,t)+10*Nz(s,t)]);
                                    ds(s,t)=ds(s,t)+norm([xs(s,t),ys(s,t),zs(s,t)]-Q);
                                else
                                    ds(s,t) = Inf;
                                end
                            end
                        end
                        [mind,id]=min(ds(:));
                        % Cutting point of the normal which joins the cell
                        % center with the venule
                        xp=xs(id);
                        yp=ys(id);
                        zp=zs(id);
                        % Height: calculated with the centroid and the
                        % cutting point
                        d_el_x=(xp-centroid(1));
                        d_el_y=(yp-centroid(2));
                        d_el_z=(zp-centroid(3));
                        points=[xp-(0:n_pixels_interp*n_div_interp-1)*d_el_x/n_div_interp; yp-(0:n_pixels_interp*n_div_interp-1)*d_el_y/n_div_interp; zp-(0:n_pixels_interp*n_div_interp-1)*d_el_z/n_div_interp]';
                        points=unique(round(points),'rows');
                        neut_x=interp3(double(cell_pred),points(:,1),points(:,2),points(:,3));
                        height_cell_polar=sum(neut_x(~isnan(neut_x)));
                        features_simple(id_neu,3)=height_cell_polar;

                        % Rotate the volume to obtain the maximum cell
                        % length in the perpendicular plane to the normal
                        if (zp-centroid(3)==0)
                            angle_of_line=0;
                        else
                            angle_of_line=atand((yp-centroid(2))/(zp-centroid(3)));
                        end
                        cell_aux=imrotate3(double(cell_pred_rot),-angle_of_line,[1,0,0])>0.5;
                        S_aux=regionprops3(cell_aux,'VoxelList');
                        if (isempty(S_aux))
                            % Cannot compute height and maximum length 
                            % (cell located in the capture extrema)
                            features_simple(id_neu,4)=NaN;
                            features_simple(id_neu,5)=NaN;
                        else
                            if (iscell(S_aux.VoxelList))
                                min_values=min(S_aux.VoxelList{1},[],1);
                                max_values=max(S_aux.VoxelList{1},[],1);
                                % Interesting planes: cell extrema
                                indX_min=find(S_aux.VoxelList{1}(:,1)==min_values(1));
                                indX_max=find(S_aux.VoxelList{1}(:,1)==max_values(1));
                                indY_min=find(S_aux.VoxelList{1}(:,2)==min_values(2));
                                indY_max=find(S_aux.VoxelList{1}(:,2)==max_values(2));
                                y_s=zeros(length(indX_min),1);
                                z_s=zeros(length(indX_min),1);
                                for s=1:length(indX_min)
                                    y_s=S_aux.VoxelList{1}(indX_min(s),2); 
                                    z_s=S_aux.VoxelList{1}(indX_min(s),3);
                                end
                                leftmost1=[min_values(1),mean(y_s),mean(z_s)];
                                y_s=zeros(length(indX_max),1);
                                z_s=zeros(length(indX_max),1);
                                for s=1:length(indX_max)
                                    y_s=S_aux.VoxelList{1}(indX_max(s),2); 
                                    z_s=S_aux.VoxelList{1}(indX_max(s),3);
                                end
                                rightmost1=[max_values(1),mean(y_s),mean(z_s)];
                                x_s=zeros(length(indY_min),1);
                                z_s=zeros(length(indY_min),1);
                                for s=1:length(indY_min)
                                    x_s=S_aux.VoxelList{1}(indY_min(s),1); 
                                    z_s=S_aux.VoxelList{1}(indY_min(s),3);
                                end
                                leftmost2=[mean(x_s),min_values(2),mean(z_s)];
                                x_s=zeros(length(indY_max),1);
                                z_s=zeros(length(indY_max),1);
                                for s=1:length(indY_max)
                                    x_s=S_aux.VoxelList{1}(indY_max(s),1); 
                                    z_s=S_aux.VoxelList{1}(indY_max(s),3);
                                end
                                rightmost2=[mean(x_s),max_values(2),mean(z_s)];
                                max_length_polar=max([norm(rightmost1-leftmost1),norm(rightmost2-leftmost2)]);
                            else
                                max_length_polar=1;
                            end
                            ratio=height_cell_polar/max_length_polar;
                            features_simple(id_neu,4)=max_length_polar;
                            features_simple(id_neu,5)=ratio;
                        end
                        % Well-segmented cell probability
                        score_neut=mean(dataSc(cell_pred));
                        features_simple(id_neu,28)=score_neut;
                    end
                end
                fprintf('SEQUENCE: %d\n',seq);
            end
            save(['results/' aux{end-1} '/' aux{end} '_Ha/' id_features '.mat'],'features_simple');
        end
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

function a = funcEllipseFit_BFisher(data)
% FITELLIPSE  Least-squares fit of ellipse to 2D points.
% Note:  Consistent with funcEllipseFit_direct with additional conversion
%        from conic coefficients to ellipse parameters
% http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FITZGIBBON/ELLIPSE/
%        A = FITELLIPSE(X,Y) returns the parameters of the best-fit
%        ellipse to 2D points (X,Y).
%        The returned vector A contains the center, radii, and orientation
%        of the ellipse, stored as (Cx, Cy, Rx, Ry, theta_radians)
%
% Authors: Andrew Fitzgibbon, Maurizio Pilu, Bob Fisher
% Reference: "Direct Least Squares Fitting of Ellipses", IEEE T-PAMI, 1999
%
% This is a more bulletproof version than that in the paper, incorporating
% scaling to reduce roundoff error, correction of behaviour when the input 
% data are on a perfect hyperbola, and returns the geometric parameters
% of the ellipse, rather than the coefficients of the quadratic form.
%
%  Example:  Run fitellipse without any arguments to get a demo
%% Test example
if nargin == 0
  % Create an ellipse
  t = linspace(0,2);
  
  Rx = 300;
  Ry = 200;
  Cx = 250;
  Cy = 150;
  Rotation = .4; % Radians
  
  x = Rx * cos(t); 
  y = Ry * sin(t);
  nx = x*cos(Rotation)-y*sin(Rotation) + Cx; 
  ny = x*sin(Rotation)+y*cos(Rotation) + Cy;
  % Draw it
  plot(nx,ny,'o');
  % Fit it
  funcEllipseFit_BFisher([nx',ny'])
  % Note it returns (Rotation - pi/2) and swapped radii, this is fine.
  return
end

%% Compute fitted ellipse coefficients of conic section
X=data(:,1);
Y=data(:,2);
% normalize data
mx = mean(X);
my = mean(Y);
sx = (max(X)-min(X))/2;
sy = (max(Y)-min(Y))/2; 

x = (X-mx)/sx;
y = (Y-my)/sy;

% Force to column vectors
x = x(:);
y = y(:);

% Build design matrix
D = [ x.*x  x.*y  y.*y  x  y  ones(size(x)) ];

% Build scatter matrix
S = D'*D;

% Build 6x6 constraint matrix
C(6,6) = 0; C(1,3) = -2; C(2,2) = 1; C(3,1) = -2;

% Solve eigensystem
if (sum(isnan(S(:)))+sum(isnan(C(:)))>0)
    a=[];
else
    [gevec, geval] = eig(S,C);

    % Find the negative eigenvalue
    I = find(real(diag(geval)) < 1e-8 & ~isinf(diag(geval)));

    % Extract eigenvector corresponding to negative eigenvalue
    % NOTE: this is ellipse coefficients of fitted conic section @Zhenyu
    A = real(gevec(:,I));

    % unnormalize
    par = [
      A(1)*sy*sy,   ...
          A(2)*sx*sy,   ...
          A(3)*sx*sx,   ...
          -2*A(1)*sy*sy*mx - A(2)*sx*sy*my + A(4)*sx*sy*sy,   ...
          -A(2)*sx*sy*mx - 2*A(3)*sx*sx*my + A(5)*sx*sx*sy,   ...
          A(1)*sy*sy*mx*mx + A(2)*sx*sy*mx*my + A(3)*sx*sx*my*my   ...
          - A(4)*sx*sy*sy*mx - A(5)*sx*sx*sy*my   ...
          + A(6)*sx*sx*sy*sy   ...
          ]';

    %% Convert conic coefficients to ellipse parameters
    % Convert to geometric radii, and centers
    thetarad = 0.5*atan2(par(2),par(1) - par(3));
    cost = cos(thetarad);
    sint = sin(thetarad);
    sin_squared = sint.*sint;
    cos_squared = cost.*cost;
    cos_sin = sint .* cost;

    Ao = par(6);
    Au =   par(4) .* cost + par(5) .* sint;
    Av = - par(4) .* sint + par(5) .* cost;
    Auu = par(1) .* cos_squared + par(3) .* sin_squared + par(2) .* cos_sin;
    Avv = par(1) .* sin_squared + par(3) .* cos_squared - par(2) .* cos_sin;

    % ROTATED = [Ao Au Av Auu Avv]

    tuCentre = - Au./(2.*Auu);
    tvCentre = - Av./(2.*Avv);
    wCentre = Ao - Auu.*tuCentre.*tuCentre - Avv.*tvCentre.*tvCentre;

    uCentre = tuCentre .* cost - tvCentre .* sint;
    vCentre = tuCentre .* sint + tvCentre .* cost;

    Ru = -wCentre./Auu;
    Rv = -wCentre./Avv;

    Ru = sqrt(abs(Ru)).*sign(Ru);
    Rv = sqrt(abs(Rv)).*sign(Rv);

    a = [uCentre, vCentre, Ru, Rv, thetarad];
end
end

function [pcoords]=getPolarCoordinatesAngle(mapSize,center,a,bs,angIn)
% mapSize: [x,y]
% centerSize: [y,x]
% angIn: radians
pcoords=zeros(mapSize);

for x=0:mapSize(1)-1
    for y=0:mapSize(2)-1
        R=(((x-center(1))*cos(-angIn)-(y-center(2))*sin(-angIn))*((x-center(1))*cos(-angIn)-(y-center(2))*sin(-angIn)))/(bs*bs)...
            +(((x-center(1))*sin(-angIn)+(y-center(2))*cos(-angIn))*((x-center(1))*sin(-angIn)+(y-center(2))*cos(-angIn)))/(a*a);
        pcoords(x+1,y+1,1)=R;
        ang=atan2(-(x-center(1)),(y-center(2)))-angIn;
        if (ang<0)
            pcoords(x+1,y+1,2)=2*pi+ang;
        else
            pcoords(x+1,y+1,2)=ang;
        end
    end
end
end

function [dist,Q] = point_to_line(P, V1, V2)
if (norm(V2-V1)==0)
    dist = norm(P-V1);
    Q = V1;
else
    v = (V2-V1)/norm(V2-V1); %// normalized vector from V1 to V2
    Q = dot(P-V1,v)*v+V1; %// projection of P onto line from V1 to V2
    dist = norm(P-Q);
end
end