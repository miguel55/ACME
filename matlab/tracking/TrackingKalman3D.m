function handles=TrackingKalman3D(label,handles,collide)
% Code based on Matlab script
% https://es.mathworks.com/help/vision/ug/motion-based-multiple-object-tracking.html
% 3D Kalman tracking system
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
frameCount=size(label,4);
handles.finalLabel2=double.empty(frameCount,0);
handles.collisions=double.empty(frameCount,0);
tracks_number=0;
tracks_prev=initializeTracks;
% Analyze collisions if provided
if (collide)
    id_neut1=10001;
    id_neut2=20001;
end
% Detect moving objects, and track them across video frames.
for f=1:frameCount
    % Objects are already detected by 3D segmentation CNN
    mask=label(:,:,:,f);
    S=regionprops3(mask,'all');
    % Centroids, bounding boxes and volume properties
    centroids=S.Centroid;
    bboxes=S.BoundingBox;
    props=[S.Volume S.EquivDiameter S.SurfaceArea];
    % Kalman prediction step
    predictNewLocationsOfTracks();
    % Hard- and soft- assignment step
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    % Update the tracks
    updateAssignedTracks();
    updateUnassignedTracks();
    % Align with the node network provided
    time_neu=handles.nodeNetwork(handles.nodeNetwork(:,5)==f,[2 1 3 6]);
    id_centroids=zeros(height(S),1);
    for m=1:size(time_neu)
        % Locate the cell index in the handles
        pos_cell=[S.Centroid(:,1),S.Centroid(:,2),S.Centroid(:,3)];
        ds_neu=sqrt(sum((time_neu(m,1:3)-pos_cell).^2,2));
        [~,p]=min(ds_neu);
        id_centroids(p,1)=time_neu(m,4);
    end
    % Prune unassignedDetections
    for k=length(unassignedDetections):-1:1
        if (id_centroids(unassignedDetections(k))==0)
            unassignedDetections(k)=[];
        end
    end
    % Create new tracks and perform the assigment in the handles struct
    createNewTracks();
    for k=1:length(unassignedDetections)
        if (id_centroids(unassignedDetections(k))>0)
            handles.finalLabel2=[handles.finalLabel2 zeros(frameCount,1)];
            handles.collisions=[handles.collisions zeros(frameCount,1)];
            handles.finalLabel2(f,tracks_number+1)=id_centroids(unassignedDetections(k),1);
            tracks_number=tracks_number+1;
        end
    end
    for k=1:size(assignments,1)
        handles.finalLabel2(f,assignments(k,1))=id_centroids(assignments(k,2));
    end
    if (collide)
        % Analyze collisions
        for k=1:length(tracks)
            % New track: separation
            % 1: track separated
            % 2: track remaining
            if (tracks(k).totalVisibleCount==1)
                for j=1:length(tracks_prev)
                    if (k~=j && tracks_prev(j).consecutiveInvisibleCount==0)
                        if (compute_iou_3D(tracks(k).bbox,tracks_prev(j).bbox)>0)
                            if ((handles.collisions(f-1,j)==0) && handles.collisions(f,k)==0)
                                handles.collisions(f,k)=id_neut1;
                                id_neut1=id_neut1+1;
                                handles.collisions(f-1,j)=id_neut1;
                                id_neut1=id_neut1+1;
                            end
                        end
                    end
                end
            end
            % Tracks disappearance: collision
            % 1: track disappeared
            % 2: track remaining
            if (tracks(k).consecutiveInvisibleCount==1)
                for j=1:length(tracks)
                    if (k~=j && tracks(j).consecutiveInvisibleCount==0)
                        if (compute_iou_3D(tracks_prev(k).bbox,double(tracks(j).bbox))>0)
                            % If the point is already taken, we will solve 
                            % the problem later
                            if ((handles.collisions(f-1,k)==0) && (handles.collisions(f,j)==0))
                                handles.collisions(f-1,k)=id_neut2;
                                id_neut2=id_neut2+1;
                                handles.collisions(f,j)=id_neut2;
                                id_neut2=id_neut2+1;
                            end
                        end
                    end
                end
            end
        end
    end
    tracks_prev=tracks;
end
non_valids=find(sum(handles.finalLabel2,1)==0);
for j=length(non_valids):-1:1
    handles.finalLabel2(:,non_valids(j))=[];
    handles.collisions(:,non_valids(j))=[];
end
handles.finalNetwork=handles.finalLabel2;
handles.finalLabel=1:size(handles.finalLabel2,2);
disp('end');

%% Initialize Tracks
% The |initializeTracks| function creates an array of tracks, where each
% track is a structure representing a moving object in the video. The
% purpose of the structure is to maintain the state of a tracked object.
% The state consists of information used for detection to track assignment,
% track termination, and display. 
%
% The structure contains the following fields:
%
% * |id| :                  the integer ID of the track
% * |bbox| :                the current bounding box of the object; used
%                           for display
% * |kalmanFilter| :        a Kalman filter object used for motion-based
%                           tracking
% * |age| :                 the number of frames since the track was first
%                           detected
% * |totalVisibleCount| :   the total number of frames in which the track
%                           was detected (visible)
% * |consecutiveInvisibleCount| : the number of consecutive frames for 
%                                  which the track was not detected (invisible).
%
% Noisy detections tend to result in short-lived tracks. For this reason,
% the example only displays an object after it was tracked for some number
% of frames. This happens when |totalVisibleCount| exceeds a specified 
% threshold.    
%
% When no detections are associated with a track for several consecutive
% frames, the example assumes that the object has left the field of view 
% and deletes the track. This happens when |consecutiveInvisibleCount|
% exceeds a specified threshold. A track may also get deleted as noise if 
% it was tracked for a short time, and marked invisible for most of the 
% frames.        

function tracks = initializeTracks()
    % create an empty array of tracks
    tracks = struct(...
        'id', {}, ...
        'bbox', {}, ...
        'prop', {}, ...
        'kalmanFilter', {}, ...
        'age', {}, ...
        'totalVisibleCount', {}, ...
        'consecutiveInvisibleCount', {});
end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

function predictNewLocationsOfTracks()
    for i = 1:length(tracks)
        bbox = tracks(i).bbox;

        % Predict the current location of the track.
        predictedCentroid = predict(tracks(i).kalmanFilter);

        predictedCentroid = predictedCentroid(1:3);
        % Shift the bounding box so that its center is at 
        % the predicted location.
        predictedCentroid = int32(int32(predictedCentroid) - int32(bbox(4:6)) / 2);
        tracks(i).bbox = [predictedCentroid, bbox(4:6)];
    end
end

%% Assign Detections to Tracks
% Assigning object detections in the current frame to existing tracks is
% done by minimizing cost. The cost is defined as the negative
% log-likelihood of a detection corresponding to a track.  
%
% The algorithm involves two steps: 
%
% Step 1: Compute the cost of assigning every detection to each track using
% the |distance| method of the |vision.KalmanFilter| System object(TM). The 
% cost takes into account the Euclidean distance between the predicted
% centroid of the track and the centroid of the detection. It also includes
% the confidence of the prediction, which is maintained by the Kalman
% filter. The results are stored in an MxN matrix, where M is the number of
% tracks, and N is the number of detections.   
%
% Step 2: Solve the assignment problem represented by the cost matrix using
% the |assignDetectionsToTracks| function. The function takes the cost 
% matrix and the cost of not assigning any detections to a track.  
%
% The value for the cost of not assigning a detection to a track depends on
% the range of values returned by the |distance| method of the 
% |vision.KalmanFilter|. This value must be tuned experimentally. Setting 
% it too low increases the likelihood of creating a new track, and may
% result in track fragmentation. Setting it too high may result in a single 
% track corresponding to a series of separate moving objects.   
%
% The |assignDetectionsToTracks| function uses the Munkres' version of the
% Hungarian algorithm to compute an assignment which minimizes the total
% cost. It returns an M x 2 matrix containing the corresponding indices of
% assigned tracks and detections in its two columns. It also returns the
% indices of tracks and detections that remained unassigned. 

function [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment()

    nTracks = length(tracks);
    nDetections = size(centroids, 1);

    % Compute the cost of assigning each detection to each track and
    % hard-assign them
    cost = zeros(nTracks, nDetections);
    for i = 1:nTracks
        cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
    end
    for i = 1:nTracks
        for z=1:nDetections
            if (tracks(i).consecutiveInvisibleCount > 1) || (double(tracks(i).bbox(4)*tracks(i).bbox(5)*tracks(i).bbox(6))/(bboxes(4)*bboxes(5)*bboxes(6))>10) || (double(tracks(i).bbox(4)*tracks(i).bbox(5)*tracks(i).bbox(6))/(bboxes(z,4)*bboxes(z,5)*bboxes(z,6))<1e-2) 
                cost(i, :)=Inf;
            end
            if (~is_inside(centroids(z,:),double(tracks(i).bbox)))
                cost(i,z)=Inf;
            end
        end
    end
    % Solve the hard-assignment problem.
    costOfNonAssignment = 20;
    [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(cost, costOfNonAssignment);
    
    % Reassign not assigned with a soft condition
    cost = zeros(length(unassignedTracks), length(unassignedDetections));
    for i = 1:length(unassignedTracks)
        cost(i, :) = distance(tracks(unassignedTracks(i)).kalmanFilter, centroids(unassignedDetections,:));
    end
    for i = 1:length(unassignedTracks)
        for z=1:length(unassignedDetections)
            if (tracks(unassignedTracks(i)).consecutiveInvisibleCount > 1) || (double(tracks(unassignedTracks(i)).bbox(4)*tracks(unassignedTracks(i)).bbox(5)*tracks(unassignedTracks(i)).bbox(6))/(bboxes(4)*bboxes(5)*bboxes(6))>10) || (double(tracks(unassignedTracks(i)).bbox(4)*tracks(unassignedTracks(i)).bbox(5)*tracks(unassignedTracks(i)).bbox(6))/(bboxes(unassignedDetections(z),4)*bboxes(unassignedDetections(z),5)*bboxes(unassignedDetections(z),6))<1e-2) 
                cost(i, :)=Inf;
            end
            if (sum(props(unassignedDetections(z),:)>2*tracks(unassignedTracks(i)).prop | props(unassignedDetections(z),:)<0.5*tracks(unassignedTracks(i)).prop)>0)
                cost(i,z)=Inf;
            end
        end
        
    end
    % Solve the assignment problem.
    costOfNonAssignment = 20;
    [assignments2, unassignedTracks2, unassignedDetections2] = ...
        assignDetectionsToTracks(cost, costOfNonAssignment);
    assignments2(:,1)=unassignedTracks(assignments2(:,1));
    assignments2(:,2)=unassignedDetections(assignments2(:,2));
    unassignedTracks=unassignedTracks(unassignedTracks2);
    unassignedDetections=unassignedDetections(unassignedDetections2);
    assignments=[assignments; assignments2];
end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0. 

function updateAssignedTracks()
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
        centroid = centroids(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);
        prop = props(detectionIdx, :);

        % Correct the estimate of the object's location
        % using the new detection.
        correct(tracks(trackIdx).kalmanFilter, centroid);

        % Replace predicted bounding box with detected
        % bounding box.
        tracks(trackIdx).bbox = bbox;
        tracks(trackIdx).prop = prop;

        % Update track's age.
        tracks(trackIdx).age = tracks(trackIdx).age + 1;

        % Update visibility.
        tracks(trackIdx).totalVisibleCount = ...
            tracks(trackIdx).totalVisibleCount + 1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
    end
end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

function updateUnassignedTracks()
    for i = 1:length(unassignedTracks)
        ind = unassignedTracks(i);
        tracks(ind).age = tracks(ind).age + 1;
        tracks(ind).consecutiveInvisibleCount = ...
            tracks(ind).consecutiveInvisibleCount + 1;
    end
end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

function deleteLostTracks()
    if isempty(tracks)
        return;
    end

    invisibleForTooLong = 42;
    ageThreshold = 42;

    % Compute the fraction of the track's age for which it was visible.
    ages = [tracks(:).age];
    totalVisibleCounts = [tracks(:).totalVisibleCount];
    visibility = totalVisibleCounts ./ ages;

    % Find the indices of 'lost' tracks.
    lostInds = (ages < ageThreshold & visibility < 0.6) | ...
        [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

    % Delete lost tracks.
    tracks = tracks(~lostInds);
end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

function createNewTracks()
    centroids = centroids(unassignedDetections, :);
    bboxes = bboxes(unassignedDetections, :);
    props = props(unassignedDetections, :);

    for i = 1:size(centroids, 1)

        centroid = centroids(i,:);
        bbox = bboxes(i, :);
        prop = props(i,:);
        % Create a Kalman filter object.
        kalmanFilter = configureKalmanFilter('ConstantAcceleration', ...
            centroid, [20, 10, 5], [3, 1,1], 3);%[200, 50], [100, 25], 100
        % Create a new track.
        newTrack = struct(...
            'id', nextId, ...
            'bbox', bbox, ...
            'prop',prop, ...
            'kalmanFilter', kalmanFilter, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0);

        % Add it to the array of tracks.
        tracks(end + 1) = newTrack;

        % Increment the next id.
        nextId = nextId + 1;
    end
end

function[iou]=compute_iou_3D(box, boxes)
    %"""Calculates IoU of the given box with the array of the given boxes.
    %box: 1D vector [x1, y1, z1, x2, y2, z2] (typically gt box)
    %boxes: [boxes_count, (x1, y1, z1, x2, y2, z2)]
    %box_area: float. the area of 'box'
    %boxes_area: array of length boxes_count.

    %Note: the areas are passed in rather than calculated here for
    %      efficency. Calculate once in the caller to avoid duplicate work.
    %"""
    %# Calculate intersection areas
    box=[box(1) box(2) box(3) box(1)+box(4) box(2)+box(5) box(3)+box(6)];
    boxes=[boxes(:,1) boxes(:,2) boxes(:,3) boxes(:,1)+boxes(:,4) boxes(:,2)+boxes(:,5) boxes(:,3)+boxes(:,6)];

    y1 = max(box(2), boxes(:,2));
    y2 = min(box(5), boxes(:,5));
    x1 = max(box(1), boxes(:,1));
    x2 = min(box(4), boxes(:,4));
    z1 = max(box(3), boxes(:,3));
    z2 = min(box(6), boxes(:,6));
    intersection = max(x2 - x1, 0) .* max(y2 - y1, 0) .* max(z2 - z1, 0);
    box_volume=box(4)*box(5)*box(6);
    boxes_volume=boxes(:,4).*boxes(:,5).*boxes(:,6);
    union = box_volume + boxes_volume(:) - intersection(:);
    iou = intersection ./ union;
end
function[is_in]=is_inside(point, box)
    %"""Calculates IoU of the given box with the array of the given boxes.
    %box: 1D vector [x1, y1, z1, x2, y2, z2] (typically gt box)
    %boxes: [boxes_count, (x1, y1, z1, x2, y2, z2)]
    %box_area: float. the area of 'box'
    %boxes_area: array of length boxes_count.

    %Note: the areas are passed in rather than calculated here for
    %      efficency. Calculate once in the caller to avoid duplicate work.
    %"""
    %# Calculate intersection areas
    box=[box(1) box(2) box(3) box(1)+box(4) box(2)+box(5) box(3)+box(6)];

    is_in=point(1)>=box(1) && point(1)<=box(4) && point(2)>=box(2) && point(2)<=box(5) ;%&& point(3)>=box(3) && point(3)<=box(6)
end
end
